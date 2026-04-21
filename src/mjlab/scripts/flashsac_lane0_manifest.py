"""Generate shared Lane 0 provenance manifests for FlashSAC tracking work."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tyro

import mjlab
from mjlab.flashsac.config import FLASHSAC_TRACKING_NUM_ENVS
from mjlab.utils.training_steps import interaction_steps_from_total_env_steps

DEFAULT_BASELINE_SHA = "2fc0c19dfc4b87187d6372bf97965b3d40bda6d0"
DEFAULT_TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1"
DEFAULT_EVAL_ENVS = 1024
DEFAULT_OUTPUT_DIR = ".omx/specs/flashsac_lane0"
DEFAULT_SMOKE_SEEDS = (42,)
DEFAULT_ACCEPTANCE_SEEDS = (7, 42)

SOURCE_OF_TRUTH_FILES = {
  "spec": ".omx/specs/deep-interview-flashsac-rootcause-parallel-refactor.md",
  "prd": ".omx/plans/prd-flashsac-tracking-rootcause-team.md",
  "test_spec": ".omx/plans/test-spec-flashsac-tracking-rootcause-team.md",
}

FLASHSAC_TRAIN_CONTRACT_FILES = (
  "src/mjlab/scripts/train.py",
  "src/mjlab/flashsac/trainer.py",
  "src/mjlab/flashsac/config.py",
)
FLASHSAC_EVAL_CONTRACT_FILES = (
  "src/mjlab/tasks/tracking/scripts/evaluate.py",
  "src/mjlab/flashsac/runtime.py",
)
FLASHSAC_PLAY_CONTRACT_FILES = (
  "src/mjlab/scripts/play.py",
  "src/mjlab/flashsac/runtime.py",
)
PPO_TRAIN_CONTRACT_FILES = ("src/mjlab/scripts/train.py",)
PPO_EVAL_CONTRACT_FILES = (
  "src/mjlab/tasks/tracking/scripts/evaluate.py",
  "src/mjlab/scripts/play.py",
)

LANE_SPECS = (
  {
    "key": "laneA-parity",
    "label": "Lane A upstream parity",
    "backend": "flashsac",
    "run_name": "laneA-parity",
    "notes": "Highest-priority algorithm parity lane.",
  },
  {
    "key": "laneB-bridge",
    "label": "Lane B adapter/env bridge",
    "backend": "flashsac",
    "run_name": "laneB-bridge",
    "notes": "Adapter/runtime contract lane.",
  },
  {
    "key": "laneC-trainer-systems",
    "label": "Lane C trainer/systems",
    "backend": "flashsac",
    "run_name": "laneC-trainer-systems",
    "notes": "Trainer cadence/device/instrumentation lane.",
  },
  {
    "key": "laneD-env-contract-control",
    "label": "Lane D env-contract control",
    "backend": "flashsac",
    "run_name": "laneD-env-contract-control",
    "notes": "Control lane; cannot become mainline with cosmetic relaxations.",
  },
  {
    "key": "laneV-ppo-control",
    "label": "Lane V PPO control",
    "backend": "ppo",
    "run_name": "laneV-ppo-control",
    "notes": "Independent PPO control under the same eval/play protocol.",
  },
)

FAIL_FAST_GATES = (0, 50_000, 100_000, 250_000)
LONG_RUN_GATES = (1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000)
COMPARISON_COLUMNS = (
  "lane",
  "backend",
  "hypothesis",
  "baseline_sha",
  "head_sha",
  "seed",
  "milestone_env_steps",
  "milestone_interaction_steps",
  "train_command",
  "eval_command",
  "play_command",
  "config_hash",
  "eval_hash",
  "train_contract_hash",
  "eval_contract_hash",
  "play_contract_hash",
  "success_rate",
  "mpkpe",
  "r_mpkpe",
  "joint_vel_error",
  "ee_pos_error",
  "ee_ori_error",
  "video_path",
  "notes",
)


@dataclass(frozen=True)
class ManifestConfig:
  """Inputs for Lane 0 shared-manifest generation."""

  leader_cwd: str | None = None
  output_dir: str = DEFAULT_OUTPUT_DIR
  task_id: str = DEFAULT_TASK_ID
  baseline_sha: str = DEFAULT_BASELINE_SHA
  eval_num_envs: int = DEFAULT_EVAL_ENVS
  smoke_seeds: list[int] = field(default_factory=lambda: list(DEFAULT_SMOKE_SEEDS))
  acceptance_seeds: list[int] = field(
    default_factory=lambda: list(DEFAULT_ACCEPTANCE_SEEDS)
  )


def _resolved_repo_root(leader_cwd: str | None) -> Path:
  if leader_cwd is not None:
    return Path(leader_cwd).expanduser().resolve()
  env_value = os.environ.get("OMX_TEAM_LEADER_CWD")
  if env_value:
    return Path(env_value).expanduser().resolve()
  return Path(__file__).resolve().parents[3]


def _resolved_output_dir(cfg: ManifestConfig, repo_root: Path) -> Path:
  output_dir = Path(cfg.output_dir).expanduser()
  if output_dir.is_absolute():
    return output_dir.resolve()
  return (repo_root / output_dir).resolve()


def _sha256_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _git_sha(repo_root: Path, ref: str = "HEAD") -> str:
  result = subprocess.run(
    ["git", "-C", str(repo_root), "rev-parse", ref],
    check=True,
    capture_output=True,
    text=True,
  )
  return result.stdout.strip()


def _git_file_bytes(repo_root: Path, ref: str, relative_path: str) -> bytes:
  result = subprocess.run(
    ["git", "-C", str(repo_root), "show", f"{ref}:{relative_path}"],
    check=True,
    capture_output=True,
  )
  return result.stdout


def _build_hash_bundle(
  repo_root: Path,
  relative_paths: tuple[str, ...],
  *,
  git_ref: str | None = None,
) -> dict[str, Any]:
  files: list[dict[str, str]] = []
  bundle_lines: list[str] = []
  for relative_path in relative_paths:
    absolute_path = (repo_root / relative_path).resolve()
    if git_ref is None:
      digest = _sha256_file(absolute_path)
    else:
      digest = _sha256_bytes(_git_file_bytes(repo_root, git_ref, relative_path))
    files.append(
      {
        "relative_path": relative_path,
        "absolute_path": str(absolute_path),
        "sha256": digest,
      }
    )
    bundle_lines.append(f"{relative_path}:{digest}")
  bundle: dict[str, Any] = {
    "sha256": _sha256_bytes("\n".join(bundle_lines).encode("utf-8")),
    "files": files,
  }
  if git_ref is not None:
    bundle["git_ref"] = git_ref
  return bundle


def _source_of_truth_bundle(repo_root: Path) -> dict[str, Any]:
  docs: dict[str, Any] = {}
  for key, relative_path in SOURCE_OF_TRUTH_FILES.items():
    absolute_path = (repo_root / relative_path).resolve()
    exists = absolute_path.is_file()
    docs[key] = {
      "relative_path": relative_path,
      "absolute_path": str(absolute_path),
      "exists": exists,
      "sha256": _sha256_file(absolute_path) if exists else None,
    }
  return docs


def _artifact_root_var() -> str:
  return "$OMX_TEAM_LEADER_CWD/.omx/artifacts/flashsac_tracking"


def _lane_bundle_root(lane_key: str, seed: int | str) -> str:
  return f"{_artifact_root_var()}/{lane_key}/seed-{seed}"


def _checkpoint_placeholder(lane_key: str, seed: int) -> str:
  return f"<{lane_key}-seed-{seed}-checkpoint-file>"


def _wandb_placeholder(lane_key: str, seed: int) -> str:
  return f"<{lane_key}-seed-{seed}-wandb-run-path>"


def _lane_commands(
  *,
  lane_key: str,
  backend: str,
  run_name: str,
  task_id: str,
  eval_num_envs: int,
  smoke_seed: int,
) -> dict[str, str]:
  bundle_root = _lane_bundle_root(lane_key, smoke_seed)
  if backend == "flashsac":
    train_command = (
      'cd "$OMX_TEAM_LEADER_CWD" && uv run train '
      f"{task_id} --backend flashsac --registry-name <motion-artifact> "
      f"--agent.seed {smoke_seed} --agent.run-name {run_name}-seed-{smoke_seed}"
    )
    eval_command = (
      'cd "$OMX_TEAM_LEADER_CWD" && uv run evaluate-tracking '
      f"{task_id} --backend flashsac --checkpoint-file "
      f"{_checkpoint_placeholder(lane_key, smoke_seed)} --motion-file <absolute-motion-npz> "
      f"--num-envs {eval_num_envs} --output-file {bundle_root}/eval/eval.json"
    )
    play_command = (
      'cd "$OMX_TEAM_LEADER_CWD" && uv run play '
      f"{task_id} --backend flashsac --checkpoint-file "
      f"{_checkpoint_placeholder(lane_key, smoke_seed)} --motion-file <absolute-motion-npz> "
      f"--num-envs 1 --viewer viser --video"
    )
  else:
    train_command = (
      'cd "$OMX_TEAM_LEADER_CWD" && uv run train '
      f"{task_id} --registry-name <motion-artifact> --agent.seed {smoke_seed} "
      f"--agent.run-name {run_name}-seed-{smoke_seed}"
    )
    eval_command = (
      'cd "$OMX_TEAM_LEADER_CWD" && uv run evaluate-tracking '
      f"{task_id} --wandb-run-path {_wandb_placeholder(lane_key, smoke_seed)} "
      f"--num-envs {eval_num_envs} --output-file {bundle_root}/eval/eval.json"
    )
    play_command = (
      'cd "$OMX_TEAM_LEADER_CWD" && uv run play '
      f"{task_id} --wandb-run-path {_wandb_placeholder(lane_key, smoke_seed)} "
      "--num-envs 1 --viewer viser --video"
    )
  return {
    "smoke_train": train_command,
    "smoke_eval": eval_command,
    "smoke_play": play_command,
  }


def _gate_rows() -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for gate in FAIL_FAST_GATES:
    rows.append(
      {
        "name": "pretrain" if gate == 0 else f"env-{gate}",
        "env_steps": gate,
        "interaction_steps": interaction_steps_from_total_env_steps(
          gate, FLASHSAC_TRACKING_NUM_ENVS
        ),
        "required_evidence": [
          "train command recorded",
          "config hash recorded",
          "eval hash recorded",
          "comparison row updated",
        ]
        + (
          []
          if gate == 0
          else [
            "evaluation JSON",
            "short play/video artifact",
            "Q/alpha/entropy/action-spread summary",
          ]
        ),
        "promotion_rule": (
          "Publish manifest + hashes before any long run."
          if gate == 0
          else "Advance only with credible movement or contract-level evidence."
        ),
      }
    )
  return rows


def _long_run_rows() -> list[dict[str, Any]]:
  return [
    {
      "env_steps": gate,
      "interaction_steps": interaction_steps_from_total_env_steps(
        gate, FLASHSAC_TRACKING_NUM_ENVS
      ),
    }
    for gate in LONG_RUN_GATES
  ]


def generate_lane0_manifests(cfg: ManifestConfig) -> dict[str, Path]:
  if not cfg.smoke_seeds:
    raise ValueError("smoke_seeds must contain at least one seed.")

  repo_root = _resolved_repo_root(cfg.leader_cwd)
  output_dir = _resolved_output_dir(cfg, repo_root)
  output_dir.mkdir(parents=True, exist_ok=True)

  observed_head_sha = _git_sha(repo_root)
  pinned_baseline_sha = _git_sha(repo_root, cfg.baseline_sha)
  source_of_truth = _source_of_truth_bundle(repo_root)
  baseline = {
    "required_sha": cfg.baseline_sha,
    "pinned_sha": pinned_baseline_sha,
    "observed_head_sha": observed_head_sha,
    "head_matches_baseline": observed_head_sha == cfg.baseline_sha,
    "repo_root": str(repo_root),
    "output_dir": str(output_dir),
    "source_of_truth": source_of_truth,
  }

  contract_hashes = {
    "flashsac_train": _build_hash_bundle(
      repo_root,
      FLASHSAC_TRAIN_CONTRACT_FILES,
      git_ref=cfg.baseline_sha,
    ),
    "flashsac_eval": _build_hash_bundle(
      repo_root,
      FLASHSAC_EVAL_CONTRACT_FILES,
      git_ref=cfg.baseline_sha,
    ),
    "flashsac_play": _build_hash_bundle(
      repo_root,
      FLASHSAC_PLAY_CONTRACT_FILES,
      git_ref=cfg.baseline_sha,
    ),
    "ppo_train": _build_hash_bundle(
      repo_root,
      PPO_TRAIN_CONTRACT_FILES,
      git_ref=cfg.baseline_sha,
    ),
    "ppo_eval": _build_hash_bundle(
      repo_root,
      PPO_EVAL_CONTRACT_FILES,
      git_ref=cfg.baseline_sha,
    ),
  }
  comparison_template_value = str((output_dir / "comparison-template.csv").resolve())
  commands_by_lane = {
    spec["key"]: {
      "label": spec["label"],
      "backend": spec["backend"],
      "notes": spec["notes"],
      **_lane_commands(
        lane_key=spec["key"],
        backend=spec["backend"],
        run_name=spec["run_name"],
        task_id=cfg.task_id,
        eval_num_envs=cfg.eval_num_envs,
        smoke_seed=cfg.smoke_seeds[0],
      ),
    }
    for spec in LANE_SPECS
  }
  artifact_schema = {
    "root": _artifact_root_var(),
    "lane_bundle_template": f"{_artifact_root_var()}/{{lane}}/seed-{{seed}}",
    "required_entries": [
      "commands/train-command.txt",
      "commands/eval-command.txt",
      "commands/play-command.txt",
      "params/env.yaml",
      "params/agent.yaml",
      "hashes.json",
      "eval/eval.json",
      "play/video-or-note.txt",
      "summary/metrics.json",
      "comparison-row.json",
    ],
    "comparison_table_template": comparison_template_value,
  }

  shared_manifest = {
    "manifest_version": 1,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "task_id": cfg.task_id,
    "baseline": baseline,
    "seed_policy": {
      "smoke": cfg.smoke_seeds,
      "acceptance": cfg.acceptance_seeds,
      "rule": (
        "All lanes use smoke seeds for fail-fast gates and acceptance seeds for any "
        "claim beyond the fail-fast stage."
      ),
    },
    "hash_rules": {
      "config_hash_rule": {
        "formula": (
          "sha256('\\n'.join([f'params/env.yaml:{sha256(env_yaml)}', "
          "f'params/agent.yaml:{sha256(agent_yaml)}']))"
        ),
        "required_files": ["params/env.yaml", "params/agent.yaml"],
      },
      "eval_hash_rule": {
        "formula": (
          "sha256('\\n'.join([f'eval-contract:{eval_contract_hash}', "
          "f'play-contract:{play_contract_hash}', f'command:{eval_command}', "
          "f'command:{play_command}']))"
        ),
        "required_inputs": [
          "eval_contract_hash",
          "play_contract_hash",
          "exact eval command",
          "exact play command",
        ],
      },
    },
    "artifact_schema": artifact_schema,
    "fail_fast_gates": _gate_rows(),
    "long_run_milestones": _long_run_rows(),
    "commands": {
      "lane0_refresh": (
        'cd "$OMX_TEAM_LEADER_CWD" && uv run flashsac-lane0-manifest '
        f"--leader-cwd {repo_root} --output-dir {output_dir}"
      ),
      "by_lane": commands_by_lane,
    },
    "contract_hashes": {key: value["sha256"] for key, value in contract_hashes.items()},
    "contract_files": contract_hashes,
    "comparison_columns": list(COMPARISON_COLUMNS),
    "gpu_allocation_table_template": [
      {"lane": spec["key"], "gpu": "<assign-before-long-run>", "notes": spec["notes"]}
      for spec in LANE_SPECS
    ],
  }

  baseline_provenance = {
    "manifest_version": 1,
    "generated_at": shared_manifest["generated_at"],
    "baseline": baseline,
    "pinned_hashes": {
      "source_of_truth": {
        key: value["sha256"] for key, value in source_of_truth.items()
      },
      "flashsac_train_contract_hash": contract_hashes["flashsac_train"]["sha256"],
      "flashsac_eval_contract_hash": contract_hashes["flashsac_eval"]["sha256"],
      "flashsac_play_contract_hash": contract_hashes["flashsac_play"]["sha256"],
    },
  }

  ppo_control_manifest = {
    "manifest_version": 1,
    "generated_at": shared_manifest["generated_at"],
    "task_id": cfg.task_id,
    "baseline_sha": cfg.baseline_sha,
    "seed_policy": shared_manifest["seed_policy"],
    "lane": "laneV-ppo-control",
    "backend": "ppo",
    "commands": commands_by_lane["laneV-ppo-control"],
    "contract_hashes": {
      "ppo_train": contract_hashes["ppo_train"]["sha256"],
      "ppo_eval": contract_hashes["ppo_eval"]["sha256"],
      "shared_play": contract_hashes["flashsac_play"]["sha256"],
    },
    "artifact_bundle_root": _lane_bundle_root("laneV-ppo-control", cfg.smoke_seeds[0]),
    "comparison_table_template": artifact_schema["comparison_table_template"],
    "notes": (
      "PPO remains the minimum qualitative and quantitative control. Use the same "
      "evaluation JSON schema, video naming, and comparison-table row shape as the "
      "FlashSAC lanes."
    ),
  }

  shared_manifest_path = output_dir / "shared-manifest.json"
  baseline_provenance_path = output_dir / "baseline-provenance.json"
  ppo_control_manifest_path = output_dir / "ppo-control-manifest.json"
  comparison_template_path = output_dir / "comparison-template.csv"

  shared_manifest_path.write_text(
    json.dumps(shared_manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
  )
  baseline_provenance_path.write_text(
    json.dumps(baseline_provenance, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
  )
  ppo_control_manifest_path.write_text(
    json.dumps(ppo_control_manifest, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
  )
  with comparison_template_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(COMPARISON_COLUMNS)

  return {
    "shared_manifest_path": shared_manifest_path,
    "baseline_provenance_path": baseline_provenance_path,
    "ppo_control_manifest_path": ppo_control_manifest_path,
    "comparison_template_path": comparison_template_path,
  }


def main() -> None:
  cfg = tyro.cli(ManifestConfig, config=mjlab.TYRO_FLAGS)
  outputs = generate_lane0_manifests(cfg)
  print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
  main()
