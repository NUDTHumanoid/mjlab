"""Script to train RL agent with RSL-RL."""

import logging

import os
import sys
import re  # 新增用于正则匹配
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path  # 删除: get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
# 删除: from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder


# 新增: 运动文件解析辅助函数
def resolve_motion_file(motion_spec: str | None) -> str | None:
    """Resolve motion file path. If motion_spec is None, return None.
       If motion_spec is an existing file, return its absolute path.
       Otherwise, assume it's a name and look for source/motions/{name}.npz.
    """
    if motion_spec is None:
        return None
    path = Path(motion_spec)
    if path.exists():
        return str(path.absolute())
    # Try default location: source/motions/{motion_spec}.npz
    default_path = Path("source/motions") / f"{motion_spec}.npz"
    if default_path.exists():
        return str(default_path.absolute())
    raise FileNotFoundError(
        f"Motion file not found: {motion_spec} (tried as absolute and in source/motions/)"
    )


@dataclass(frozen=True)
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: RslRlBaseRunnerCfg
    # 修改: 保留 registry_name 但标记为弃用（可选），新增 motion_file 用于本地文件
    registry_name: str | None = None
    # 新添: motion_file 参数，用于指定本地 NPZ 运动文件（支持名称或路径）
    motion_file: str | None = None
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    enable_nan_guard: bool = False
    torchrunx_log_dir: str | None = None
    # 删除: wandb_run_path 和 wandb_checkpoint_name，改用本地加载
    # 原 wandb_run_path: str | None = None
    # 原 wandb_checkpoint_name: str | None = None
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

    @staticmethod
    def from_task(task_id: str) -> "TrainConfig":
        env_cfg = load_env_cfg(task_id)
        agent_cfg = load_rl_cfg(task_id)
        return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        device = "cpu"
        seed = cfg.agent.seed
        rank = 0
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        # Set EGL device to match the CUDA device.
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        device = f"cuda:{local_rank}"
        # Set seed to have diversity in different processes.
        seed = cfg.agent.seed + local_rank

    configure_torch_backends()

    cfg.agent.seed = seed
    cfg.env.seed = seed

    print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

    # 修改: 处理跟踪任务的数据源
    is_tracking_task = "motion" in cfg.env.commands and isinstance(
        cfg.env.commands["motion"], MotionCommandCfg
    )

    if is_tracking_task:
        motion_cmd = cfg.env.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)

        # 修改: 优先使用本地 motion_file（支持智能解析），如果未提供则尝试从 registry_name 构造路径
        if cfg.motion_file is not None:
            # 新添: 调用解析函数
            resolved_motion = resolve_motion_file(cfg.motion_file)
            motion_cmd.motion_file = resolved_motion
            print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")
        elif cfg.registry_name:
            # 修改: 保留 registry_name 的兼容性，但仅用于解析本地文件路径（假设文件已存在于 source/motions 下）
            # 原逻辑是从 WandB 下载，现改为从本地 source/motions 目录加载
            # 假设 registry_name 格式为 "your-org/motions/motion-name"，提取 motion-name
            match = re.search(r"/([^/]+)(?::latest)?$", cfg.registry_name)
            if match:
                motion_name = match.group(1)
                # 默认路径：source/motions/{motion_name}.npz
                local_path = Path("source/motions") / f"{motion_name}.npz"
                if local_path.exists():
                    motion_cmd.motion_file = str(local_path.absolute())
                    print(f"[INFO] Using local motion file derived from registry_name: {motion_cmd.motion_file}")
                else:
                    raise FileNotFoundError(
                        f"Registry name '{cfg.registry_name}' maps to expected local file {local_path}, "
                        "but the file does not exist. Please place the NPZ file there or use --motion-file directly."
                    )
            else:
                raise ValueError(f"Could not parse motion name from registry_name: {cfg.registry_name}")
        else:
            raise ValueError(
                "For tracking tasks, provide either:\n"
                "  --motion-file /path/to/motion.npz (local file)\n"
                "  --registry-name your-org/motions/motion-name (deprecated, expects file in source/motions/)"
            )

    # Enable NaN guard if requested.
    if cfg.enable_nan_guard:
        cfg.env.sim.nan_guard.enabled = True
        print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

    if rank == 0:
        print(f"[INFO] Logging experiment in directory: {log_dir}")

    env = ManagerBasedRlEnv(
        cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
    )

    log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

    # 修改: 移除 WandB checkpoint 加载，仅保留本地加载
    resume_path: Path | None = None
    if cfg.agent.resume:
        # 删除: 原先的 wandb_run_path 分支，现直接使用本地加载
        # 若需要加载特定 run 的 checkpoint，需通过 agent.load_run 和 agent.load_checkpoint 指定
        resume_path = get_checkpoint_path(
            log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
        )
        if resume_path is not None:
            print(f"[INFO] Loading checkpoint from local path: {resume_path}")
        else:
            print("[WARNING] No checkpoint found to resume. Starting from scratch.")

    # Only record videos on rank 0 to avoid multiple workers writing to the same files.
    if cfg.video and rank == 0:
        env = VideoRecorder(
            env,
            video_folder=Path(log_dir) / "videos" / "train",
            step_trigger=lambda step: step % cfg.video_interval == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )
        print("[INFO] Recording videos during training.")

    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

    agent_cfg = asdict(cfg.agent)
    env_cfg = asdict(cfg.env)

    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
        runner_cls = MjlabOnPolicyRunner

    # 删除: runner_kwargs 中的 registry_name，不再传递
    runner_kwargs = {}
    # 原代码: if is_tracking_task: runner_kwargs["registry_name"] = registry_name

    # Write config files before runner creation, since the runner mutates agent_cfg
    # in-place (e.g., injecting non-serializable objects).
    if rank == 0:
        dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
        dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

    runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

    # 删除: add_wandb_tags 调用
    # add_wandb_tags(cfg.agent.wandb_tags)
    # runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(str(resume_path))

    runner.learn(
        num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
    )

    env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
    args = args or TrainConfig.from_task(task_id)

    # Create log directory once before launching workers.
    log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
    log_root_path.resolve()
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.agent.run_name:
        log_dir_name += f"_{args.agent.run_name}"
    log_dir = log_root_path / log_dir_name

    # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
    selected_gpus, num_gpus = select_gpus(args.gpu_ids)

    # Set environment variables for all modes.
    if selected_gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    os.environ["MUJOCO_GL"] = "egl"

    if num_gpus <= 1:
        # CPU or single GPU: run directly without torchrunx.
        run_train(task_id, args, log_dir)
    else:
        # Multi-GPU: use torchrunx.
        import torchrunx

        # torchrunx redirects stdout to logging.
        logging.basicConfig(level=logging.INFO)

        # Configure torchrunx logging directory.
        # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
        if "TORCHRUNX_LOG_DIR" not in os.environ:
            if args.torchrunx_log_dir is not None:
                # User specified a value via flag (could be "" to disable).
                os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
            else:
                # Default: put logs in training directory.
                os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

        print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
        torchrunx.Launcher(
            hostnames=["localhost"],
            workers_per_host=num_gpus,
            backend=None,  # Let rsl_rl handle process group initialization.
            copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
        ).run(run_train, task_id, args, log_dir)


def main():
    # Parse first argument to choose the task.
    # Import tasks to populate the registry.
    import mjlab.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    args = tyro.cli(
        TrainConfig,
        args=remaining_args,
        default=TrainConfig.from_task(chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    del remaining_args

    launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
    main()
