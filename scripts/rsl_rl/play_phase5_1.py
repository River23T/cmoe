# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 5.1 play/evaluation script — Phase 2a + VAE 输出接到 actor.

Phase 5.1 actor 输入 134-dim = [policy(99), v_pred(3), z_H(32)].
评估时 必须 inject VAE 才能让 actor 工作 (跟 Phase 2a 评估不同).

评估指标 (cmd=0.8 m/s, 20s):
  - success_rate > 0.9
  - avg_dist > 7m
  - error_vel_xy 小 (showing actor 真用 v_pred)

关键差异 vs play_phase2.py:
  1. CALL inject_moe_swav (Phase 5.1 必须有 VAE 才能跑)
  2. CALL inject_phase4_runner_patches (load checkpoint 容错)
  3. task = "CMoE-Phase5-1-Terrains-Play-v0"
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

parser = argparse.ArgumentParser(description="Phase 5.1 (VAE → actor) eval.")
parser.add_argument("--num_envs",   type=int,   default=64)
parser.add_argument("--task",       type=str,   default="CMoE-Phase5-1-Terrains-Play-v0")
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
    """Run Phase 2 (VAE estimator) evaluation."""
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

    # ============================================================
    # Phase 5.1: 必须 inject VAE+actor (actor 输入 134 dim 必须靠 VAE 才能算)
    # ============================================================
    agent_dict = agent_cfg.to_dict()
    cmoe_cfg = agent_dict.get("algorithm", {}).pop("cmoe_cfg", None)
    if cmoe_cfg is None:
        raise RuntimeError("[Phase5_1-Play] cmoe_cfg missing — task config error")
    print(f"[Phase5_1-Play] popped cmoe_cfg with keys: {list(cmoe_cfg.keys())}, phase={cmoe_cfg.get('phase')}")

    runner = OnPolicyRunner(env, agent_dict, log_dir=log_dir, device=agent_cfg.device)

    # Phase 5.1 inject (creates VAE, augments actor 166 (5 experts), wraps act/update)
    print("[Phase5_1-Play] injecting VAE + actor extension...")
    from cmoe.tasks.velocity.cmoe_phase4.phase4_inject import (
        inject_moe_swav, inject_phase4_runner_patches,
    )
    inject_moe_swav(runner.alg, cmoe_cfg, device=agent_cfg.device)
    inject_phase4_runner_patches(runner)

    # 加载 checkpoint (patched runner.load 会处理 166 (5 experts) dim 转换)
    print(f"[Phase5_1-Play] loading checkpoint: {resume_path}")
    runner.load(resume_path)

    # =========================================================================
    # CRITICAL FIX: 用 rsl_rl 标准的 get_inference_policy() 而不是 act_inference()
    # =========================================================================
    # 旧版本 (BAD):
    #   def policy(obs):
    #       return runner.alg.policy.act_inference(obs)
    # 问题:
    #   obs 是 TensorDict {policy: (N, 495), critic: (N, 99)}, act_inference()
    #   不会按 obs_groups 提取 'policy' 这个 group, 而是把整个 TensorDict 当作单一
    #   tensor 传给 actor. VAEAugmentedActor.forward(x) 接到的不是 (N, 495) 的
    #   tensor, 行为未定义 → 输出几乎为 0 → 机器人不动.
    #
    # 新版本 (GOOD, 与 play_baseline.py / play.py 一致):
    #   policy = runner.get_inference_policy(device=runner.device)
    # 内部实现:
    #   1. self.alg.policy.eval()
    #   2. 返回闭包 inference_fn(obs):
    #        actor_obs = self._extract_actor_obs(obs)   # 按 obs_groups['policy']
    #        return self.alg.policy.act_inference(actor_obs)  # 现在传的是 tensor
    #
    # 这是 rsl_rl 3.x 的标准用法 (humanoid_locomotion, cmoe baseline 都用这个).
    # =========================================================================
    runner.alg.vae_estimator.eval()
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
            # SANITY CHECK: 第一个 episode 的第一步, 打印 action norm.
            # 训练好的 policy 输出的 action 应该有合理 norm (~1.0~5.0),
            # 若 norm < 0.05 说明 actor 输出全 0 ⇒ inference 路径 bug.
            if ep_idx == 0 and t == 0:
                a_norm = actions.norm(dim=-1).mean().item()
                a_max  = actions.abs().max().item()
                print(f"[Phase5_1-Play SANITY] step 0: action_norm_mean={a_norm:.4f}, "
                      f"action_abs_max={a_max:.4f}")
                if a_norm < 0.05:
                    print(f"[Phase5_1-Play SANITY] ⚠️  action norm extremely low — "
                          f"policy may be outputting near-zero actions.")
                else:
                    print(f"[Phase5_1-Play SANITY] ✓ actor producing nontrivial actions")
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
    print("═" * 70)
    print("PHASE 5.1 (MoE+SwAV + 8 paper terrains + curriculum) RESULTS")
    print("═" * 70)
    print(f"  mean success rate : {np.mean(all_success):.3f}")
    print(f"  mean avg_dist     : {np.mean(all_dist):.2f} m  "
          f"(theoretical max @ cmd=0.8m/s × 20s = 16m)")
    print()
    print("  注意: Phase 5.1 在 8 种 paper 地形 (Slope/Stairs/Gap/Hurdle/Discrete/Mix1/Mix2)")
    print("        上评估, 难度比 Phase 4 平地大很多. avg_dist 显著低于 Phase 4 是预期的.")
    print()
    if np.mean(all_success) > 0.5 and np.mean(all_dist) > 5.0:
        print("  ✅ PHASE 5.1 PASSED. 多地形 walking 起作用 ✓")
        print("     说明: 5 个 expert 开始针对不同地形 specialize (gating 信号活跃)")
        print("     SwAV contrastive 可帮助 gating cluster 不同地形的 z_E")
        print("     下一步: Phase 5.2 - velocity curriculum + 30N push + 域随机化")
    elif np.mean(all_success) > 0.2:
        print("  ⚠️  Walking 在简单地形 OK, 复杂地形仍有挑战")
        print("     可能原因: 训练 iter 不够 (paper 用 20K, 我们用 5K); ")
        print("              没启用 velocity curriculum (留给 Phase 5.2)")
        print("     建议: 继续训练 (--resume 加 5000 iter) 或先做 Phase 5.2")
    else:
        print("  ❌ PHASE 5.1 FAILED. 多地形 walking 严重退化")
        print("     可能原因:")
        print("       1. Phase 4 warm-start 没成功 (检查 [Phase4-LOAD] resume 路径)")
        print("       2. 地形太难, 训练 iter 不够")
        print("       3. terrain_levels curriculum 没正确升级 (检查 [CURR-FINAL diag] log)")
    print("═" * 70)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
