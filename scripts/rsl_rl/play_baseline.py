# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 1 baseline play/evaluation script — STANDARD PPO, NO MoE.

Runs N episodes of the trained baseline policy on flat ground at fixed cmd
speed, reports success rate and avg distance.

If success_rate > 0.9 and avg_dist > 7m at cmd=0.8 m/s for 20s,
the baseline reward design is VALIDATED and we can proceed to Phase 2.

FIX HISTORY
-----------
v2 (this version): Removed `runner.alg.eval_mode()` call.
  In rsl_rl 3.x, the standard PPO algorithm class does NOT have an
  `eval_mode()` method. That method is only added DYNAMICALLY by
  cmoe_inject._patch_mode_and_save() when CMoE is injected (line 201
  of cmoe_inject.py: `alg.eval_mode = em`).

  Since the baseline does NOT inject CMoE (correct behavior), the
  attribute does not exist → AttributeError.

  The fix: just rely on `runner.get_inference_policy()` which internally
  handles eval mode via `self.alg.policy.eval()`. This is exactly what
  the cmoe project's own play.py does (line 168 of cmoe/scripts/rsl_rl/play.py).
"""

# --- Monkey-patch: prevent NaN/negative std crashes ---
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

parser = argparse.ArgumentParser(description="Phase 1 baseline eval.")
parser.add_argument("--num_envs",   type=int,   default=64)
parser.add_argument("--task",       type=str,   default="CMoE-Baseline-Phase1-Play-v0")
parser.add_argument("--agent",      type=str,   default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--n_episodes", type=int,   default=4)
parser.add_argument("--cmd_speed",  type=float, default=0.8)
parser.add_argument("--ckpt",       type=str,   default=None)
parser.add_argument("--run_dir",    type=str,   default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectRLEnvCfg,
    DirectMARLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import cmoe.tasks  # noqa: F401


def find_latest_run(experiment_name: str) -> str:
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))
    if not os.path.isdir(log_root):
        raise FileNotFoundError(f"No experiment root: {log_root}")
    runs = sorted([
        d for d in os.listdir(log_root)
        if os.path.isdir(os.path.join(log_root, d))
    ])
    if not runs:
        raise FileNotFoundError(f"No runs in: {log_root}")
    latest = os.path.join(log_root, runs[-1])
    print(f"[INFO] Using run directory: {latest}")
    return latest


def find_latest_checkpoint(run_dir: str) -> str:
    files = [f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")]
    if not files:
        raise FileNotFoundError(f"No model_*.pt in {run_dir}")
    files.sort(key=lambda f: int(f.replace("model_", "").replace(".pt", "")))
    latest = os.path.join(run_dir, files[-1])
    print(f"[INFO] Auto-selected latest checkpoint: {latest}")
    return latest


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run Phase 1 baseline evaluation."""
    env_cfg.scene.num_envs = args_cli.num_envs

    # Lock cmd to forward at desired speed
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.cmd_speed, args_cli.cmd_speed)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading   = (0.0, 0.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)

    if args_cli.run_dir is not None:
        run_dir = os.path.abspath(args_cli.run_dir)
    else:
        run_dir = find_latest_run(agent_cfg.experiment_name)
    if args_cli.ckpt is not None:
        resume_path = os.path.abspath(args_cli.ckpt)
    else:
        resume_path = find_latest_checkpoint(run_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    log_dir = os.path.dirname(resume_path)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path)
    # ★ FIX: Do NOT call runner.alg.eval_mode() here.
    # In rsl_rl 3.x, PPO has no eval_mode method (it's added by CMoE injection).
    # runner.get_inference_policy() handles eval mode internally.
    policy = runner.get_inference_policy(device=runner.device)

    print(f"\n[INFO] Eval: {args_cli.n_episodes} episodes × {args_cli.num_envs} envs × "
          f"{env_cfg.episode_length_s}s @ cmd={args_cli.cmd_speed} m/s")

    device = env.unwrapped.device
    num_envs = args_cli.num_envs
    all_success = []
    all_dist = []

    for ep_idx in range(args_cli.n_episodes):
        env.reset()
        obs_out = env.get_observations()
        obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out

        ep_initial_pos = env.unwrapped.scene["robot"].data.root_pos_w[:, :2].clone()
        fell_buf      = torch.zeros(num_envs, dtype=torch.bool, device=device)
        timed_out_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
        max_dist           = torch.zeros(num_envs, device=device)
        last_pos_before_done = ep_initial_pos.clone()

        max_steps = int(env.unwrapped.max_episode_length)
        for t in range(max_steps):
            actions = policy(obs)
            cur_pos_before = env.unwrapped.scene["robot"].data.root_pos_w[:, :2].clone()
            d_now = torch.norm(cur_pos_before - ep_initial_pos, dim=1)

            still_active = ~(fell_buf | timed_out_buf)
            last_pos_before_done[still_active] = cur_pos_before[still_active]
            max_dist[still_active] = torch.maximum(max_dist[still_active], d_now[still_active])

            obs, _, dones, extras = env.step(actions)
            dones_flat = dones.squeeze(-1).bool() if dones.dim() > 1 else dones.bool()
            time_outs = extras.get("time_outs", None)
            if time_outs is not None:
                timeout_flat = time_outs.squeeze(-1).bool() if time_outs.dim() > 1 else time_outs.bool()
                fell_now    = dones_flat & (~timeout_flat)
                timeout_now = dones_flat &   timeout_flat
            else:
                ep_len = env.unwrapped.episode_length_buf
                ml    = env.unwrapped.max_episode_length
                near_end = ep_len >= (ml - 2)
                fell_now    = dones_flat & (~near_end)
                timeout_now = dones_flat &   near_end

            fell_buf      |= fell_now    & (~fell_buf) & (~timed_out_buf)
            timed_out_buf |= timeout_now & (~fell_buf) & (~timed_out_buf)
            if torch.all(fell_buf | timed_out_buf):
                break

        never_term = ~(fell_buf | timed_out_buf)
        if never_term.any():
            cur_pos = env.unwrapped.scene["robot"].data.root_pos_w[:, :2]
            d_now = torch.norm(cur_pos[never_term] - ep_initial_pos[never_term], dim=1)
            max_dist[never_term] = torch.maximum(max_dist[never_term], d_now)

        success_mask = ~fell_buf
        sr = success_mask.float().mean().item()
        d_avg = max_dist.mean().item()
        d_max = max_dist.max().item()
        n_fell = int(fell_buf.sum().item())
        n_to   = int(timed_out_buf.sum().item())

        all_success.append(sr)
        all_dist.append(d_avg)
        print(f"  [Ep {ep_idx+1}/{args_cli.n_episodes}] success={sr:.3f}  "
              f"avg_dist={d_avg:.2f}m  max_dist={d_max:.2f}m  "
              f"(fell={n_fell}, timed_out={n_to})")

    print()
    print("═" * 60)
    print("PHASE 1 BASELINE RESULTS — flat ground, std cmd 0.8 m/s")
    print("═" * 60)
    print(f"  mean success rate : {np.mean(all_success):.3f}")
    print(f"  mean avg_dist     : {np.mean(all_dist):.2f} m  "
          f"(theoretical max @ cmd=0.8m/s × 20s = 16m)")
    print()
    if np.mean(all_success) > 0.9 and np.mean(all_dist) > 7.0:
        print("  ✅ PHASE 1 PASSED. Reward design VALIDATED.")
        print("     Proceed to Phase 2 (add VAE/AE estimators).")
    elif np.mean(all_success) > 0.5:
        print("  ⚠️  Partial success. Robot can stand but not walk well.")
        print("     Consider: increase track_lin_vel_xy weight further, or")
        print("     check obs config / actor scale.")
    else:
        print("  ❌ PHASE 1 FAILED. Even flat-ground walking does not converge.")
        print("     This indicates a fundamental issue (obs scaling, actuator config,")
        print("     or reward bug). Investigate before adding any complexity.")
    print("═" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
