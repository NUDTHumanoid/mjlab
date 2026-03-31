from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


#Modified by czy:增添tracking rough四阶段地形reset采样事件
class TerrainSamplingStage(TypedDict):
  step: int
  max_terrain_level: int
  terrain_type_probs: dict[str, float]


def _resolve_tracking_terrain_stage(
  stages: list[TerrainSamplingStage],
  common_step_counter: int,
) -> TerrainSamplingStage:
  active_stage = stages[0]
  for stage in stages:
    if common_step_counter >= stage["step"]:
      active_stage = stage
  return active_stage


def _terrain_type_columns(terrain_generator_cfg) -> dict[str, list[int]]:
  names = list(terrain_generator_cfg.sub_terrains.keys())
  proportions = [sub_cfg.proportion for sub_cfg in terrain_generator_cfg.sub_terrains.values()]
  total = sum(proportions)
  if total <= 0.0:
    raise ValueError("Terrain proportions must sum to a positive value.")

  cumulative: list[float] = []
  running = 0.0
  for proportion in proportions:
    running += proportion / total
    cumulative.append(running)

  terrain_type_columns = {name: [] for name in names}
  for index in range(terrain_generator_cfg.num_cols):
    threshold = index / terrain_generator_cfg.num_cols + 0.001
    for sub_index, upper in enumerate(cumulative):
      if threshold < upper:
        terrain_type_columns[names[sub_index]].append(index)
        break

  return terrain_type_columns


class staged_tracking_terrain_sampling:
  #Modified by czy:增添带状态的tracking rough阶段采样器，支持自动捕获rough起始offset
  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._captured_stage_step_offset: int | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    stages: list[TerrainSamplingStage],
    rough_stage_step_offset: int | None = None,
    auto_capture_offset: bool = True,
    within_patch_xy_range: tuple[float, float] | None = None,
  ) -> None:
    """Sample tracking terrains by training stage on each reset."""
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
      env_ids = env_ids.to(device=env.device, dtype=torch.long)

    if env_ids.numel() == 0:
      return

    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None or terrain.env_origins is None:
      return

    terrain_generator_cfg = terrain.cfg.terrain_generator
    if terrain_generator_cfg is None:
      return

    #Modified by czy:修改tracking rough阶段判断逻辑，优先使用手动offset，否则自动捕获rough起点
    if rough_stage_step_offset is not None:
      resolved_offset = int(rough_stage_step_offset)
      self._captured_stage_step_offset = resolved_offset
    elif auto_capture_offset:
      if self._captured_stage_step_offset is None:
        self._captured_stage_step_offset = int(env.common_step_counter)
      resolved_offset = self._captured_stage_step_offset
    else:
      resolved_offset = 0

    effective_step = max(env.common_step_counter - resolved_offset, 0)
    active_stage = _resolve_tracking_terrain_stage(stages, effective_step)
    terrain_type_columns = _terrain_type_columns(terrain_generator_cfg)

    valid_type_names = [
      name
      for name, weight in active_stage["terrain_type_probs"].items()
      if weight > 0.0 and len(terrain_type_columns.get(name, [])) > 0
    ]
    if not valid_type_names:
      raise ValueError("No active terrain types available for staged terrain sampling.")

    type_weights = torch.tensor(
      [active_stage["terrain_type_probs"][name] for name in valid_type_names],
      device=env.device,
      dtype=torch.float32,
    )
    type_weights /= torch.sum(type_weights)

    sampled_type_ids = torch.multinomial(
      type_weights,
      num_samples=env_ids.numel(),
      replacement=True,
    )
    sampled_terrain_types = torch.empty_like(env_ids)

    for local_type_id, terrain_type_name in enumerate(valid_type_names):
      assigned_mask = sampled_type_ids == local_type_id
      if not torch.any(assigned_mask):
        continue
      candidate_columns = torch.tensor(
        terrain_type_columns[terrain_type_name],
        device=env.device,
        dtype=torch.long,
      )
      sampled_columns = candidate_columns[
        torch.randint(
          0,
          candidate_columns.numel(),
          (int(torch.sum(assigned_mask).item()),),
          device=env.device,
        )
      ]
      sampled_terrain_types[assigned_mask] = sampled_columns

    max_level = min(active_stage["max_terrain_level"], terrain.terrain_origins.shape[0] - 1)
    sampled_terrain_levels = torch.randint(
      0,
      max_level + 1,
      (env_ids.numel(),),
      device=env.device,
    )

    terrain.terrain_levels[env_ids] = sampled_terrain_levels
    terrain.terrain_types[env_ids] = sampled_terrain_types
    terrain.env_origins[env_ids] = terrain.terrain_origins[
      terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]

    if within_patch_xy_range is not None:
      lower, upper = within_patch_xy_range
      offsets_xy = sample_uniform(lower, upper, (env_ids.numel(), 2), device=env.device)
      terrain.env_origins[env_ids, :2] += offsets_xy
