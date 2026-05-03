# --- Monkey-patch: prevent NaN/negative std crashes in Normal distribution ---
import torch as _torch
_OrigNormal = _torch.distributions.Normal
class _SafeNormal(_OrigNormal):
    def __init__(self, loc, scale, validate_args=None):
        scale = _torch.clamp(scale, min=1e-6)
        loc = _torch.nan_to_num(loc, nan=0.0, posinf=1e6, neginf=-1e6)
        super().__init__(loc, scale, validate_args=validate_args)
_torch.distributions.Normal = _SafeNormal
# --- End monkey-patch ---

from isaaclab.app import AppLauncher
import argparse
import sys
import cli_args  # isort: skip

# ============================================================
# Phase 2 train.py — added VAE-only inject path
#
# 与原 train.py 的唯一差别:
#   原: 看到 cmoe_cfg 就 inject_cmoe (完整 MoE 套件)
#   新: 看 cmoe_cfg 内容决定:
#       - 只有 estimator (无 moe, 无 contrastive) → inject_vae_only
#       - 有 moe → inject_cmoe (原行为)
# ============================================================

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None,
                    help="RL Policy training iterations. When --resume, this is interpreted as ADDITIONAL iters.")
parser.add_argument("--train_to", type=int, default=None,
                    help="train UNTIL this absolute iteration (overrides --max_iterations semantics when used with --resume).")
parser.add_argument("--init_terrain_level", type=int, default=None,
                    help="override TerrainImporter.max_init_terrain_level.")
parser.add_argument("--warm_start_ckpt", type=str, default=None,
                    help="Path to a checkpoint from a different experiment (e.g. Phase 2a) to warm-start. "
                         "Different from --resume: --resume continues same experiment dir, "
                         "--warm_start_ckpt loads weights from another experiment.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

logger = logging.getLogger(__name__)

import cmoe.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Train with RSL-RL agent."""
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, args_cli.agent)

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    if args_cli.init_terrain_level is not None:
        if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "terrain"):
            env_cfg.scene.terrain.max_init_terrain_level = args_cli.init_terrain_level
            print(f"[INFO] max_init_terrain_level overridden to {args_cli.init_terrain_level}")

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    agent_dict = agent_cfg.to_dict()
    cmoe_cfg = agent_dict.get("algorithm", {}).pop("cmoe_cfg", None)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_dict, log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_dict, log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    # ============================================================
    # Phase 2b smart inject routing (cmoe_cfg.phase = "2b") + warm-start
    # ============================================================
    # 检查 cmoe_cfg 内容决定走哪条 inject 路径:
    #   - phase == "2b" → Phase 2b (VAE + actor extension)
    #   - 只有 estimator → Phase 2a (VAE-only)
    #   - 有 moe → 完整 CMoE (Phase 4+)
    # ============================================================
    if cmoe_cfg is not None:
        has_moe = "moe" in cmoe_cfg
        has_contrastive = "contrastive" in cmoe_cfg
        has_estimator = "estimator" in cmoe_cfg
        phase_tag = cmoe_cfg.get("phase", None)

        # IMPORTANT: phase_tag 优先!
        # 早期版本是 has_moe/has_contrastive 优先, 但这会让 Phase 4
        # (有 moe + contrastive 字段) 走错路径调到老的 inject_cmoe
        # (它不知道 per-term 975-dim layout).
        if phase_tag == "5.1":
            # Phase 5.1 = Phase 4 architecture + 8 paper terrains (env-only change)
            # Reuse Phase 4 inject directly — actor/critic/MoE/SwAV 与 Phase 4 完全相同.
            print("[INFO] Detected phase=5.1, reusing inject_moe_swav (Phase 4 architecture)")
            from cmoe.tasks.velocity.cmoe_phase4.phase4_inject import (
                inject_moe_swav, inject_phase4_runner_patches,
            )
            inject_moe_swav(runner.alg, cmoe_cfg, device=agent_cfg.device)
            inject_phase4_runner_patches(runner)
        elif phase_tag == "4":
            print("[INFO] Detected phase=4, using inject_moe_swav (Phase 4)")
            from cmoe.tasks.velocity.cmoe_phase4.phase4_inject import (
                inject_moe_swav, inject_phase4_runner_patches,
            )
            inject_moe_swav(runner.alg, cmoe_cfg, device=agent_cfg.device)
            inject_phase4_runner_patches(runner)
        elif phase_tag == "3":
            print("[INFO] Detected phase=3, using inject_vae_ae_actor (Phase 3)")
            from cmoe.tasks.velocity.cmoe_phase3.phase3_inject import (
                inject_vae_ae_actor, inject_phase3_runner_patches,
            )
            inject_vae_ae_actor(runner.alg, cmoe_cfg, device=agent_cfg.device)
            inject_phase3_runner_patches(runner)
        elif phase_tag == "2b":
            print("[INFO] Detected phase=2b, using inject_vae_actor (Phase 2b)")
            from cmoe.tasks.velocity.cmoe_phase2b.phase2b_inject import (
                inject_vae_actor, inject_phase2b_runner_patches,
            )
            inject_vae_actor(runner.alg, cmoe_cfg, device=agent_cfg.device)
            inject_phase2b_runner_patches(runner)
        elif has_moe or has_contrastive:
            # 仅在没有 phase_tag 时回落到 old inject_cmoe (legacy path)
            print("[INFO] Detected full CMoE config (no phase_tag), using legacy inject_cmoe")
            from cmoe.custom_classes.models.cmoe_inject import inject_cmoe, inject_cmoe_runner_patches
            inject_cmoe(runner.alg, cmoe_cfg, device=agent_cfg.device)
            inject_cmoe_runner_patches(runner)
        elif has_estimator:
            print("[INFO] Detected estimator-only config, using inject_vae_only (Phase 2a)")
            from cmoe.tasks.velocity.cmoe_phase2.phase2_inject import (
                inject_vae_only, inject_phase2_runner_patches,
            )
            inject_vae_only(runner.alg, cmoe_cfg, device=agent_cfg.device)
            inject_phase2_runner_patches(runner)
        else:
            print("[WARN] cmoe_cfg present but no estimator/moe/contrastive — nothing injected")

    runner.add_git_repo_to_log(__file__)
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
    elif args_cli.warm_start_ckpt is not None:
        # Warm-start from another experiment's checkpoint
        # (e.g. Phase 2a → Phase 2b: load 99-dim actor weights, augment to 134)
        ws_path = os.path.abspath(args_cli.warm_start_ckpt)
        print(f"[INFO]: Warm-starting from checkpoint: {ws_path}")
        if not os.path.isfile(ws_path):
            raise FileNotFoundError(f"warm_start_ckpt not found: {ws_path}")
        runner.load(ws_path)
        # Reset iteration count to 0 (we're starting fresh experiment, just borrowing weights)
        if hasattr(runner, "current_learning_iteration"):
            runner.current_learning_iteration = 0
            print("[INFO]: reset runner.current_learning_iteration = 0 after warm-start")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    if args_cli.train_to is not None:
        current_iter = getattr(runner, "current_learning_iteration", 0)
        iters_remaining = max(0, args_cli.train_to - current_iter)
        print(f"[INFO] --train_to {args_cli.train_to} specified; current iter = {current_iter}; "
              f"will train for {iters_remaining} more iters.")
        runner.learn(num_learning_iterations=iters_remaining, init_at_random_ep_len=True)
    else:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
