# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize CMoE policy on terrains and record video + expert activations.

This script does TWO things at once:
  1. Records a side-view MP4 of the robot traversing each terrain
  2. Logs expert activation weights over time → CSV → reproduces paper Fig. 5

Usage:
    isaaclab -p scripts/rsl_rl/visualize_cmoe.py \
      --task CMoE-G1-Velocity-Play-v0 \
      --num_envs 1 \
      --episode_length 25 \
      --cmd_speed 0.6 \
      --output_dir /root/cmoe/logs/visuals \
      --video

Outputs:
  - /root/cmoe/logs/visuals/run_*/video.mp4
  - /root/cmoe/logs/visuals/run_*/gate_weights.csv
  - /root/cmoe/logs/visuals/run_*/expert_activations.png

After running, generate the paper-style Fig. 5 plot:
    python plot_fig5_experts.py /root/cmoe/logs/visuals/run_*/gate_weights.csv
"""

# --- Monkey-patch ---
import torch as _torch
_OrigNormal = _torch.distributions.Normal
class _SafeNormal(_OrigNormal):
    def __init__(self, loc, scale, validate_args=None):
        scale = _torch.clamp(scale, min=1e-6)
        loc = _torch.nan_to_num(loc, nan=0.0, posinf=1e6, neginf=-1e6)
        super().__init__(loc, scale, validate_args=validate_args)
_torch.distributions.Normal = _SafeNormal
# --- end patch ---

from isaaclab.app import AppLauncher
import argparse
import sys
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Visualize CMoE policy with video+expert tracking.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="CMoE-G1-Velocity-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--episode_length", type=float, default=20.0,
                    help="Seconds to record (default 20s = paper).")
parser.add_argument("--cmd_speed", type=float, default=0.8)
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--run_dir", type=str, default=None)
parser.add_argument("--output_dir", type=str, default="logs/visuals")
parser.add_argument("--terrain", type=str, default="stratified",
                    choices=["stratified", "hardest"])
parser.add_argument("--video", action="store_true", default=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras for video
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, re, glob, time
import numpy as np
import gymnasium as gym
import torch
from datetime import datetime
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlBaseRunnerCfg
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils.hydra import hydra_task_config
import cmoe.tasks  # noqa


def find_latest_checkpoint(run_dir_override=None, ckpt_override=None):
    if ckpt_override is not None:
        return os.path.abspath(ckpt_override)
    if run_dir_override is not None:
        run_dir = os.path.abspath(run_dir_override)
    else:
        for root in ["logs/rsl_rl/cmoe_g1_velocity",
                     "/root/cmoe/logs/rsl_rl/cmoe_g1_velocity"]:
            if os.path.isdir(root):
                runs = sorted([d for d in glob.glob(os.path.join(root, "20*"))
                               if os.path.isdir(d)], key=os.path.getmtime)
                if runs:
                    run_dir = runs[-1]; break
        else:
            raise FileNotFoundError("No run dir found")
    ckpts = sorted(glob.glob(os.path.join(run_dir, "model_*.pt")),
                   key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1)))
    return ckpts[-1]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlBaseRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs

    # fixed forward command
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.cmd_speed, args_cli.cmd_speed)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)

    # disable curricula
    if hasattr(env_cfg, "curriculum"):
        for an in dir(env_cfg.curriculum):
            if an.startswith("_"): continue
            try:
                v = getattr(env_cfg.curriculum, an)
                if hasattr(v, "func"):
                    setattr(env_cfg.curriculum, an, None)
            except Exception:
                pass

    # disable push
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot.params["velocity_range"] = {"x": (0,0), "y": (0,0)}
        env_cfg.events.push_robot.interval_range_s = (1e9, 1e9)

    # extend episode length
    env_cfg.episode_length_s = args_cli.episode_length

    # eval terrains
    if args_cli.terrain == "stratified":
        from cmoe.tasks.velocity.cmoe.terrains.config.cmoe_eval_stratified import (
            CMOE_EVALUATE_STRATIFIED_TERRAINS_CFG as TCFG,
        )
    else:
        from cmoe.tasks.velocity.cmoe.terrains.config.cmoe_eval import (
            CMOE_EVALUATE_TERRAINS_CFG as TCFG,
        )
    TCFG.curriculum = False
    env_cfg.scene.terrain.terrain_generator = TCFG
    env_cfg.scene.terrain.max_init_terrain_level = None

    # output dir
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.abspath(os.path.join(args_cli.output_dir, f"run_{stamp}"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[viz] output dir: {out_dir}")

    resume_path = find_latest_checkpoint(args_cli.run_dir, args_cli.ckpt)
    print(f"[viz] checkpoint: {resume_path}")

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # video recorder
    if args_cli.video:
        video_dir = os.path.join(out_dir, "videos")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,  # only first episode
            video_length=int(args_cli.episode_length / env_cfg.sim.dt / env_cfg.decimation),
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    agent_dict = agent_cfg.to_dict()
    cmoe_cfg = agent_dict.get("algorithm", {}).pop("cmoe_cfg", None)
    runner = OnPolicyRunner(env, agent_dict, log_dir=None, device=agent_cfg.device)

    from cmoe.custom_classes.models.cmoe_inject import inject_cmoe, inject_cmoe_runner_patches
    inject_cmoe(runner.alg, cmoe_cfg, device=agent_cfg.device)
    inject_cmoe_runner_patches(runner)
    runner.load(resume_path)
    runner.alg.eval_mode()

    moe_actor = runner.alg.moe_actor
    ae = getattr(runner.alg, "ae_estimator", None)
    vae = getattr(runner.alg, "vae_estimator", None)

    @torch.inference_mode()
    def policy_with_gates(obs):
        po = obs.get("policy")
        eo = obs.get("elevation")
        if vae and po is not None: vae.get_latent_and_velocity(po)
        ze = ae.encode(eo) if (ae and eo is not None) else None
        am = moe_actor(po, ze)
        am = torch.nan_to_num(am, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-100, 100)
        gw = moe_actor.get_gate_weights().detach().cpu().numpy()
        return am, gw

    # rollout one episode
    obs_out = env.get_observations()
    obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out

    gate_log = []  # list of (t_seconds, gate_weights[num_experts])
    pos_log = []
    terrain_log = []

    dt_step = env_cfg.sim.dt * env_cfg.decimation  # seconds per env step

    n_steps = int(args_cli.episode_length / dt_step)
    print(f"[viz] running {n_steps} steps × {dt_step:.3f}s = {args_cli.episode_length}s")

    for t in range(n_steps):
        actions, gw = policy_with_gates(obs)
        # gw shape [num_envs, num_experts]; we only viz env 0
        gate_log.append(gw[0].copy())
        pos = env.unwrapped.scene["robot"].data.root_pos_w[0, :3].cpu().numpy()
        pos_log.append(pos.copy())
        ttype = int(env.unwrapped.scene.terrain.terrain_types[0].item())
        tlevel = int(env.unwrapped.scene.terrain.terrain_levels[0].item())
        terrain_log.append((ttype, tlevel))

        obs, _, dones, _ = env.step(actions)
        if dones.any():
            print(f"[viz] env terminated at step {t} ({t*dt_step:.2f}s)")
            break

    # write CSV
    import csv
    csv_path = os.path.join(out_dir, "gate_weights.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        n_e = gate_log[0].shape[0] if gate_log else 5
        header = ["t_seconds", "x", "y", "z", "terrain_type", "terrain_level"] + \
                 [f"expert_{i+1}" for i in range(n_e)]
        w.writerow(header)
        for i, (gw, pos, (tt, tl)) in enumerate(zip(gate_log, pos_log, terrain_log)):
            t_s = i * dt_step
            row = [f"{t_s:.3f}", f"{pos[0]:.3f}", f"{pos[1]:.3f}", f"{pos[2]:.3f}",
                   tt, tl] + [f"{x:.4f}" for x in gw]
            w.writerow(row)
    print(f"[viz] gate weights → {csv_path}")

    # quick plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        gates = np.array(gate_log)
        ts = np.arange(len(gates)) * dt_step
        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(gates.shape[1]):
            ax.plot(ts, gates[:, i], label=f"Expert {i+1}", linewidth=1.5)
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Activation")
        ax.set_title(f"CMoE Expert Activations (cmd={args_cli.cmd_speed} m/s)")
        ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(loc="upper right")
        plt.tight_layout()
        png_path = os.path.join(out_dir, "expert_activations.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"[viz] expert plot → {png_path}")
    except ImportError:
        print("[viz] matplotlib not installed; skip png")

    env.close()
    print(f"[viz] DONE — outputs in {out_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
