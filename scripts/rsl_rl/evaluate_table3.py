# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate trained CMoE policy against paper Table III.

==================================================================
R13 — fix avg_dist=0.11m bug
==================================================================

R12 correctly fixed success-rate (78.5%, paper-comparable).
But avg_dist showed ~0.10m for all terrains, while paper reports
9-15m. Root cause:

When IsaacLab triggers `time_out=True`, the env auto-resets the
robot to its starting position BEFORE we can read the position.
In R12, we did:

    if timeout_now.any():
        cur_pos = env.unwrapped.scene["robot"].data.root_pos_w[:, :2]
        ... # this reads RESET position (~0,0), not pre-reset!
        final_pos[new_to] = cur_pos[new_to]

Fix: track per-step `max_dist_so_far` BEFORE the step that triggers
the done. Use this as the final distance when timeout/fall happens.

Also: track displacement in robot's command direction (lin_vel_x of
the world-frame velocity command), since paper §V says "movement
distance in the direction of movement". We track simple Euclidean
distance from start, which is correct for a forward-only command.

==================================================================
R13 — also adds t-SNE export for paper Fig. 2
==================================================================

When --export_tsne is set, we record (gate_weights, terrain_label)
pairs over a few episodes and save them to a .npz for offline
t-SNE plotting (matches paper Fig. 2).
"""

# --- Monkey-patch (same as train.py) ---
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

parser = argparse.ArgumentParser(description="Evaluate CMoE policy on paper Table III terrains.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="CMoE-G1-Velocity-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_episodes", type=int, default=4)
parser.add_argument("--cmd_speed", type=float, default=0.8)
parser.add_argument("--mode", type=str, default="stratified",
                    choices=["stratified", "hardest"])
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--run_dir", type=str, default=None)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--export_tsne", type=str, default=None,
                    help="If set, save (gate_weights, terrain_label) pairs to this .npz path for t-SNE plotting (paper Fig. 2).")
parser.add_argument("--tsne_max_samples", type=int, default=20000,
                    help="Max samples to record for t-SNE export.")
# R14: Eval-yaw alignment fixes (no retraining needed).
# Default values change eval semantics:
#   --fix_reset_yaw: reset_base yaw range -> (0, 0). Robot always starts facing +x.
#                    Eliminates "robot facing wrong way before heading-control aligns".
#   --heading_stiffness: override heading_control_stiffness.
#                        Higher values make yaw track command faster.
# R15 UPDATE: --heading_stiffness default 2.0 -> 0.5 (matches training).
#             --fix_reset_yaw default True -> True (kept).
#             Test A confirmed R14 stiffness change was NOT the dominant
#             cause; the real fix is wiring VAE outputs to the actor.
parser.add_argument("--fix_reset_yaw", action="store_true", default=True,
                    help="Default ON: reset_base yaw range -> (0, 0). "
                         "Pass --no_fix_reset_yaw to disable. "
                         "Paper Table III implicitly assumes the robot starts pointing +x; "
                         "training-time random yaw + eval-time heading=(0,0) command is a known mismatch.")
parser.add_argument("--no_fix_reset_yaw", dest="fix_reset_yaw", action="store_false")
parser.add_argument("--heading_stiffness", type=float, default=0.5,
                    help="Override heading_control_stiffness for eval. "
                         "R15: matches training (cmoe_env_cfg=0.5).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================
# Imports AFTER simulation_app
# ============================================================
import os
import re
import glob
import numpy as np
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    ManagerBasedRLEnvCfg,
    DirectRLEnvCfg,
    DirectMARLEnvCfg,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlBaseRunnerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import cmoe.tasks  # noqa: F401


def find_latest_checkpoint(run_dir_override=None, ckpt_override=None):
    if ckpt_override is not None:
        ckpt_path = os.path.abspath(ckpt_override)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"--ckpt path does not exist: {ckpt_path}")
        return ckpt_path

    if run_dir_override is not None:
        run_dir = os.path.abspath(run_dir_override)
    else:
        candidates = []
        for root in ["logs/rsl_rl/cmoe_g1_velocity",
                     "/root/cmoe/logs/rsl_rl/cmoe_g1_velocity",
                     os.path.abspath("logs/rsl_rl/cmoe_g1_velocity")]:
            if os.path.isdir(root):
                candidates.append(root); break
        if not candidates:
            raise FileNotFoundError("Could not find logs/rsl_rl/cmoe_g1_velocity/")
        log_root = candidates[0]
        runs = sorted(
            [d for d in glob.glob(os.path.join(log_root, "20*")) if os.path.isdir(d)],
            key=os.path.getmtime,
        )
        if not runs:
            raise FileNotFoundError(f"No run directories under {log_root}")
        run_dir = runs[-1]

    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    print(f"[INFO] Using run directory: {run_dir}")

    ckpts = glob.glob(os.path.join(run_dir, "model_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No model_*.pt in {run_dir}")

    def iter_num(path):
        m = re.search(r"model_(\d+)\.pt", os.path.basename(path))
        return int(m.group(1)) if m else -1

    ckpts.sort(key=iter_num)
    latest = ckpts[-1]
    print(f"[INFO] Auto-selected latest checkpoint: {latest}")
    return latest


PAPER_TABLE3 = {
    "slope":      {"sr": 0.991, "dist": 14.870},
    "stair_up":   {"sr": 0.886, "dist": 10.802},
    "stair_down": {"sr": 0.905, "dist": 10.824},
    "discrete":   {"sr": 0.991, "dist": 13.440},
    "gap":        {"sr": 0.974, "dist": 14.876},
    "hurdle":     {"sr": 0.987, "dist": 13.470},
    "mix1":       {"sr": 0.767, "dist": 12.055},
    "mix2":       {"sr": 0.747,  "dist":  9.750},
}

EVAL_TERRAIN_NAMES = [
    "slope_up", "slope_down", "stair_up", "stair_down",
    "gap", "hurdle", "discrete", "mix1", "mix2",
]


def make_moe_inference_fn(alg, capture_gates=False):
    """Build a deterministic MoE inference function (no noise sampling).

    R15 PAPER-FAITHFUL FIX
    ======================
    Now builds the same systematic observation that training uses:
        actor_input = [v_pred, z_H, z_E, o_c, e_t]  (259-dim)

    If capture_gates=True, the latest gate weights are exposed via
    `infer.last_gate_weights` for t-SNE export.
    """
    if not (hasattr(alg, "moe_actor") and alg.moe_actor is not None):
        raise RuntimeError("alg has no moe_actor — inject_cmoe() must run first")
    moe_actor = alg.moe_actor
    ae = getattr(alg, "ae_estimator", None)
    vae = getattr(alg, "vae_estimator", None)
    sys_dims = getattr(alg, "_cmoe_systematic_dims", None)
    if sys_dims is None:
        raise RuntimeError("R15: alg._cmoe_systematic_dims not set — re-run inject_cmoe")
    single_obs_dim = sys_dims["o_c"]

    @torch.inference_mode()
    def infer(obs):
        po = obs.get("policy", None)
        eo = obs.get("elevation", None)
        if po is None:
            raise RuntimeError("obs has no 'policy' key")
        if vae is None or ae is None:
            raise RuntimeError("R15: VAE+AE both required")

        # Build systematic obs (paper-faithful)
        z_H, v_pred = vae.get_latent_and_velocity(po)
        ze = ae.encode(eo)
        o_c = po[:, -single_obs_dim:]
        sys_obs = torch.cat([v_pred, z_H, ze, o_c, eo], dim=-1)

        am = moe_actor(sys_obs, ze)
        am = torch.nan_to_num(am, nan=0.0, posinf=10.0, neginf=-10.0)
        am = torch.clamp(am, -100.0, 100.0)

        if capture_gates:
            try:
                gw = moe_actor.get_gate_weights()
                infer.last_gate_weights = gw.detach().cpu().numpy()
            except Exception:
                infer.last_gate_weights = None

        return am

    infer.last_gate_weights = None
    return infer


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run Table-III evaluation."""
    env_cfg.scene.num_envs = args_cli.num_envs

    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.cmd_speed, args_cli.cmd_speed)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)

    # R14: Eval-yaw alignment fixes (no retraining needed).
    # Without this, robot resets to random yaw (-π, π) but command says
    # "go to yaw=0", causing 1-2 sec of off-direction travel before
    # heading_control aligns. On stair_up / hurdle / mix2 / discrete this
    # ends in hitting obstacles sideways → falls. This is the dominant cause
    # of train(70% success)→eval(33% success) gap.
    if args_cli.heading_stiffness != 0.5:
        env_cfg.commands.base_velocity.heading_control_stiffness = args_cli.heading_stiffness
        print(f"[R14] heading_control_stiffness: 0.5 -> {args_cli.heading_stiffness}")
    if args_cli.fix_reset_yaw:
        if hasattr(env_cfg.events, "reset_base"):
            old_pose = env_cfg.events.reset_base.params.get("pose_range", {})
            env_cfg.events.reset_base.params["pose_range"] = {
                "x": old_pose.get("x", (-0.1, 0.1)),
                "y": old_pose.get("y", (-0.1, 0.1)),
                "yaw": (0.0, 0.0),  # R14: was (-3.14, 3.14)
            }
            print(f"[R14] reset_base yaw range: (-π, π) -> (0, 0). Robot starts facing +x.")

    # Disable curriculum (R12)
    if hasattr(env_cfg, "curriculum"):
        for attr_name in dir(env_cfg.curriculum):
            if attr_name.startswith("_"):
                continue
            try:
                val = getattr(env_cfg.curriculum, attr_name)
                if hasattr(val, "func"):
                    setattr(env_cfg.curriculum, attr_name, None)
            except (AttributeError, TypeError):
                pass
        print("[R12] Curriculum disabled for evaluation.")

    if hasattr(env_cfg.scene.terrain, "terrain_generator") and env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0)
        }
        env_cfg.events.push_robot.interval_range_s = (1e9, 1e9)

    if args_cli.mode == "stratified":
        from cmoe.tasks.velocity.cmoe.terrains.config.cmoe_eval_stratified import (
            CMOE_EVALUATE_STRATIFIED_TERRAINS_CFG,
        )
        eval_cfg = CMOE_EVALUATE_STRATIFIED_TERRAINS_CFG
        eval_cfg.curriculum = False
        env_cfg.scene.terrain.terrain_generator = eval_cfg
        env_cfg.scene.terrain.max_init_terrain_level = None
        print(f"[INFO] EVAL MODE: 'stratified' (curriculum disabled)")
    else:
        from cmoe.tasks.velocity.cmoe.terrains.config.cmoe_eval import (
            CMOE_EVALUATE_TERRAINS_CFG,
        )
        eval_cfg = CMOE_EVALUATE_TERRAINS_CFG
        eval_cfg.curriculum = False
        env_cfg.scene.terrain.terrain_generator = eval_cfg
        env_cfg.scene.terrain.max_init_terrain_level = None
        print(f"[INFO] EVAL MODE: 'hardest' (curriculum disabled)")

    resume_path = find_latest_checkpoint(
        run_dir_override=args_cli.run_dir,
        ckpt_override=args_cli.ckpt,
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    agent_dict = agent_cfg.to_dict()
    cmoe_cfg = agent_dict.get("algorithm", {}).pop("cmoe_cfg", None)
    runner = OnPolicyRunner(env, agent_dict, log_dir=None, device=agent_cfg.device)

    if cmoe_cfg is None:
        raise RuntimeError("agent_cfg has no cmoe_cfg")

    from cmoe.custom_classes.models.cmoe_inject import inject_cmoe, inject_cmoe_runner_patches

    inject_cmoe(runner.alg, cmoe_cfg, device=agent_cfg.device)
    inject_cmoe_runner_patches(runner)
    runner.load(resume_path)

    runner.alg.eval_mode()

    capture_gates = args_cli.export_tsne is not None
    policy = make_moe_inference_fn(runner.alg, capture_gates=capture_gates)

    print("\n[SANITY] MoE weight statistics (after load):")
    with torch.no_grad():
        for name, p in runner.alg.moe_actor.named_parameters():
            if "weight" in name and p.numel() > 100:
                std_ = p.std().item()
                mean_ = p.mean().item()
                print(f"  {name}: std={std_:.4f}  mean={mean_:+.4f}  shape={tuple(p.shape)}")

    print(f"\n[INFO] Eval: {args_cli.n_episodes} episodes × {args_cli.num_envs} envs × 20s @ cmd={args_cli.cmd_speed} m/s")

    device = env.unwrapped.device
    num_envs = args_cli.num_envs

    terrain_types_t = env.unwrapped.scene.terrain.terrain_types
    terrain_levels_t = env.unwrapped.scene.terrain.terrain_levels
    num_rows_t = env.unwrapped.scene.terrain.cfg.terrain_generator.num_rows

    per_terrain_success = {n: [] for n in EVAL_TERRAIN_NAMES}
    per_terrain_dist    = {n: [] for n in EVAL_TERRAIN_NAMES}
    per_difficulty_success = {q: [] for q in ["easy(0-0.33)", "med(0.33-0.66)", "hard(0.66-1)"]}

    # t-SNE export buffer
    tsne_gates = []
    tsne_labels = []

    for ep_idx in range(args_cli.n_episodes):
        env.reset()
        obs_out = env.get_observations()
        obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out

        ep_initial_pos = env.unwrapped.scene["robot"].data.root_pos_w[:, :2].clone()
        ep_terrain_types  = terrain_types_t.clone()
        ep_terrain_levels = terrain_levels_t.clone()

        fell_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
        timed_out_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # R13 BUG FIX: track running-max distance from start, BEFORE each step.
        # When timeout fires, env auto-resets pos to (0,0); we lose the pre-reset pos.
        # So we record pos every step and keep the LATEST pos seen before being marked done.
        max_dist = torch.zeros(num_envs, device=device)
        last_pos_before_done = ep_initial_pos.clone()

        max_steps = int(env.unwrapped.max_episode_length)
        for t in range(max_steps):
            actions = policy(obs)

            # R13 FIX: Snapshot position BEFORE step (so we have pre-reset pos).
            cur_pos_before = env.unwrapped.scene["robot"].data.root_pos_w[:, :2].clone()
            d_now = torch.norm(cur_pos_before - ep_initial_pos, dim=1)

            # For envs not yet done, update last_pos and max_dist
            still_active = ~(fell_buf | timed_out_buf)
            update_mask = still_active
            last_pos_before_done[update_mask] = cur_pos_before[update_mask]
            max_dist[update_mask] = torch.maximum(max_dist[update_mask], d_now[update_mask])

            # capture gates for t-SNE (only for currently-active envs)
            if capture_gates and policy.last_gate_weights is not None:
                gw = policy.last_gate_weights  # [num_envs, num_experts]
                active_idx = still_active.cpu().numpy()
                if active_idx.any():
                    gw_active = gw[active_idx]
                    types_active = ep_terrain_types.cpu().numpy()[active_idx]
                    levels_active = ep_terrain_levels.cpu().numpy()[active_idx]
                    # encode (terrain_type, level_bucket) as a single int label
                    labels_active = types_active * 100 + levels_active
                    if len(tsne_gates) * num_envs < args_cli.tsne_max_samples:
                        tsne_gates.append(gw_active.copy())
                        tsne_labels.append(labels_active.copy())

            obs, _, dones, extras = env.step(actions)

            if dones.dim() > 1:
                dones_flat = dones.squeeze(-1).bool()
            else:
                dones_flat = dones.bool()

            time_outs = extras.get("time_outs", None)
            if time_outs is not None:
                if time_outs.dim() > 1:
                    timeout_flat = time_outs.squeeze(-1).bool()
                else:
                    timeout_flat = time_outs.bool()
                fell_now = dones_flat & (~timeout_flat)
                timeout_now = dones_flat & timeout_flat
            else:
                ep_len = env.unwrapped.episode_length_buf
                ml = env.unwrapped.max_episode_length
                near_end = ep_len >= (ml - 2)
                fell_now = dones_flat & (~near_end)
                timeout_now = dones_flat & near_end

            # R13 FIX: don't read robot.data.root_pos_w AFTER reset. We already
            # snapshotted last_pos_before_done before this step.
            new_fall = fell_now & (~fell_buf) & (~timed_out_buf)
            fell_buf |= new_fall
            new_to = timeout_now & (~fell_buf) & (~timed_out_buf)
            timed_out_buf |= new_to

            all_done = fell_buf | timed_out_buf
            if torch.all(all_done):
                break

        # For any env that NEVER triggered done, use current pos
        never_term = ~(fell_buf | timed_out_buf)
        if never_term.any():
            cur_pos = env.unwrapped.scene["robot"].data.root_pos_w[:, :2]
            last_pos_before_done[never_term] = cur_pos[never_term]
            d_now = torch.norm(cur_pos[never_term] - ep_initial_pos[never_term], dim=1)
            max_dist[never_term] = torch.maximum(max_dist[never_term], d_now)

        # R13: use max_dist (running max) as the official traveled distance
        dist_traveled = max_dist
        success_mask = ~fell_buf

        for env_id in range(num_envs):
            col = int(ep_terrain_types[env_id].item())
            row = int(ep_terrain_levels[env_id].item())
            if 0 <= col < len(EVAL_TERRAIN_NAMES):
                name = EVAL_TERRAIN_NAMES[col]
                succ = bool(success_mask[env_id].item())
                dist = float(dist_traveled[env_id].item())
                per_terrain_success[name].append(succ)
                per_terrain_dist[name].append(dist)
                difficulty = row / max(num_rows_t - 1, 1)
                if difficulty < 0.33:
                    per_difficulty_success["easy(0-0.33)"].append(succ)
                elif difficulty < 0.66:
                    per_difficulty_success["med(0.33-0.66)"].append(succ)
                else:
                    per_difficulty_success["hard(0.66-1)"].append(succ)

        sr = success_mask.float().mean().item()
        n_fell = int(fell_buf.sum().item())
        n_to = int(timed_out_buf.sum().item())
        d_avg = dist_traveled.mean().item()
        d_max = dist_traveled.max().item()
        print(f"  [Ep {ep_idx+1}/{args_cli.n_episodes}] success={sr:.3f}  "
              f"avg_dist={d_avg:.2f}m  max_dist={d_max:.2f}m  "
              f"(fell={n_fell}, timed_out={n_to}, n_envs={num_envs})")

    # final report
    print("\n" + "═" * 90)
    print(f"PAPER TABLE III COMPARISON — mode = {args_cli.mode}")
    print("═" * 90)
    print(f"{'Terrain':<14} {'OURS sr':>10} {'PAPER sr':>10} "
          f"{'OURS dist':>10} {'PAPER dist':>11}  {'sr gap':>10}")
    print("─" * 90)

    def report_row(name, ours_sr, ours_d, paper_key):
        paper = PAPER_TABLE3.get(paper_key, {"sr": float('nan'), "dist": float('nan')})
        gap_sr = ours_sr - paper["sr"]
        print(f"{name:<14} {ours_sr:>10.3f} {paper['sr']:>10.3f} "
              f"{ours_d:>10.3f} {paper['dist']:>11.3f}  {gap_sr:>+10.3f}")

    slope_succ = per_terrain_success["slope_up"] + per_terrain_success["slope_down"]
    slope_dist = per_terrain_dist["slope_up"]    + per_terrain_dist["slope_down"]
    if len(slope_succ) > 0:
        report_row("slope (avg)", float(np.mean(slope_succ)), float(np.mean(slope_dist)), "slope")

    for name in ["stair_up", "stair_down", "discrete", "gap", "hurdle", "mix1", "mix2"]:
        s = per_terrain_success[name]
        d = per_terrain_dist[name]
        if len(s) == 0:
            print(f"{name:<14} (no envs)"); continue
        report_row(name, float(np.mean(s)), float(np.mean(d)), name)

    print("═" * 90)
    all_s, all_d = [], []
    for vals in per_terrain_success.values(): all_s.extend(vals)
    for vals in per_terrain_dist.values():    all_d.extend(vals)
    if len(all_s) > 0:
        print(f"{'OVERALL':<14} {np.mean(all_s):>10.3f} {'-':>10} {np.mean(all_d):>10.3f}")
    print("═" * 90)

    print("\n┄┄┄ Difficulty breakdown ┄┄┄")
    for bucket, vals in per_difficulty_success.items():
        if len(vals) > 0:
            print(f"  {bucket:<22}: success={np.mean(vals):.3f}  (n={len(vals)})")
    print()

    # t-SNE export
    if capture_gates and len(tsne_gates) > 0:
        gates_arr = np.concatenate(tsne_gates, axis=0)
        labels_arr = np.concatenate(tsne_labels, axis=0)
        # subsample if needed
        if len(gates_arr) > args_cli.tsne_max_samples:
            idx = np.random.choice(len(gates_arr), args_cli.tsne_max_samples, replace=False)
            gates_arr = gates_arr[idx]
            labels_arr = labels_arr[idx]
        out_path = os.path.abspath(args_cli.export_tsne)
        np.savez(out_path,
                 gates=gates_arr,
                 labels=labels_arr,
                 terrain_names=np.array(EVAL_TERRAIN_NAMES))
        print(f"[t-SNE] Saved {len(gates_arr)} (gate, label) pairs → {out_path}")
        print(f"[t-SNE] To plot: run plot_tsne_fig2.py {out_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
