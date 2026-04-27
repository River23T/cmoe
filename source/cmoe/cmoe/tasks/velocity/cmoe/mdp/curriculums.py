# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CMoE curriculum functions (R11 — diagnostic + robust terrain curriculum).

=============================================================
R11 vs R10: CRITICAL FIX for terrain_levels stuck at 0.0000
=============================================================

iter 948-967 log (R10) shows:
  * track_lin_vel_xy reward = 0.81 (GREAT, R9 was 0.22)
  * error_vel_xy = 0.18 m/s (EXCELLENT tracking, R9 was 1.0)
  * Mean episode length = 280 steps = 5.6 s
  * base_contact = 85% (robot WALKS but falls)
  * time_out = 14%
  * **terrain_levels = 0.0000** ← still STUCK

The tracking is FINALLY WORKING (R10 weight=4.0 + std=0.5 fixed it).
Envs that timeout travel ~8m (0.4 m/s × 20s) >> promote threshold 2.0m.
Even base_contact envs walk ~1.3m on average.

Yet mean terrain_level == 0.0000 after 967 iter. R11 adds diagnostic
instrumentation to pinpoint the exact issue, and makes env_ids
handling more robust.

R11 FIXES:
  1. Coerce env_ids to an int64 tensor explicitly before any indexing.
  2. Print ONE-LINE diagnostic every 500 curriculum calls showing
     distance stats and promotion counts.
  3. Fail-safe slice(None) handling: if called with slice, skip
     (can't compute per-env distances meaningfully from all envs).
  4. Explicit dtype matching for terrain_levels update.

R10 thresholds preserved:
  promote: distance > 2.0 m
  demote:  distance < 0.3 m
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# A module-level global counter so we only print diagnostics every N calls.
_DIAG_CALL_COUNT = 0
_DIAG_EVERY = 500         # print every 500 calls to curriculum


# ---------------------------------------------------------------------------
# Linear velocity command curriculum (R11 — robustified)
# ---------------------------------------------------------------------------
def lin_vel_cmd_levels(
    env: "ManagerBasedRLEnv",
    env_ids,
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    """Per-reset command curriculum (same logic as R5-R10)."""
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)

    # Handle slice(None) from CurriculumManager init
    is_full = isinstance(env_ids, slice)

    if not is_full and hasattr(env_ids, "__len__") and len(env_ids) > 0:
        # Ensure tensor
        if not isinstance(env_ids, torch.Tensor):
            env_ids_t = torch.as_tensor(list(env_ids), dtype=torch.long, device=env.device)
        else:
            env_ids_t = env_ids.to(dtype=torch.long, device=env.device)
        reward = torch.mean(
            env.reward_manager._episode_sums[reward_term_name][env_ids_t]
        ) / env.max_episode_length_s
    else:
        reward = torch.tensor(0.0, device=env.device)

    upgrade_threshold = reward_term.weight * 0.6
    min_progress = 1.0

    if reward > upgrade_threshold and reward > min_progress:
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)
        ranges.lin_vel_x = torch.clamp(
            torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
            limit_ranges.lin_vel_x[0],
            limit_ranges.lin_vel_x[1],
        ).tolist()
        ranges.lin_vel_y = torch.clamp(
            torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
            limit_ranges.lin_vel_y[0],
            limit_ranges.lin_vel_y[1],
        ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


# ---------------------------------------------------------------------------
# Terrain-level curriculum (R11 — robust, diagnostic)
# ---------------------------------------------------------------------------
def terrain_levels_vel(
    env: "ManagerBasedRLEnv",
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terrain-difficulty curriculum (R11 robust + diagnostic)."""
    global _DIAG_CALL_COUNT

    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # ----- coerce env_ids to a proper tensor -----
    if isinstance(env_ids, slice):
        # Full slice → all envs. Can't compute per-env travel sensibly;
        # only real resets will pass a tensor.
        return torch.mean(terrain.terrain_levels.float())

    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(list(env_ids), dtype=torch.long, device=env.device)
    else:
        env_ids = env_ids.to(dtype=torch.long, device=env.device)

    if env_ids.numel() == 0:
        return torch.mean(terrain.terrain_levels.float())

    # ----- compute per-env travel distance -----
    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )

    # ----- R10 thresholds -----
    promote_distance = 2.0
    demote_distance = 0.3

    move_up = distance > promote_distance
    move_down = (distance < demote_distance) & (~move_up)

    # ----- update env origins -----
    terrain.update_env_origins(env_ids, move_up, move_down)

    # ----- max-level re-seed -----
    max_level_idx = terrain.cfg.terrain_generator.num_rows - 1
    current_levels = terrain.terrain_levels[env_ids]
    reset_condition = (current_levels == max_level_idx) & move_up
    if torch.any(reset_condition):
        reset_ids = env_ids[reset_condition]
        num_resets = int(reset_ids.numel())
        random_levels = torch.randint(
            0, max_level_idx + 1, (num_resets,), device=env.device,
            dtype=terrain.terrain_levels.dtype,
        )
        terrain.terrain_levels[reset_ids] = random_levels
        cols = terrain.terrain_types[reset_ids]
        new_origins = terrain.terrain_origins[random_levels, cols]
        terrain.env_origins[reset_ids] = new_origins

    # ----- R11 DIAGNOSTIC (every _DIAG_EVERY calls) -----
    _DIAG_CALL_COUNT += 1
    if _DIAG_CALL_COUNT % _DIAG_EVERY == 0:
        n = int(env_ids.numel())
        d_mean = float(distance.mean().item()) if n > 0 else 0.0
        d_min = float(distance.min().item()) if n > 0 else 0.0
        d_max = float(distance.max().item()) if n > 0 else 0.0
        n_up = int(move_up.sum().item())
        n_dn = int(move_down.sum().item())
        tl_mean = float(terrain.terrain_levels.float().mean().item())
        tl_max_v = int(terrain.terrain_levels.max().item())
        tl_nonzero = int((terrain.terrain_levels > 0).sum().item())
        tl_total = int(terrain.terrain_levels.numel())
        print(
            f"[R11-CURR diag #{_DIAG_CALL_COUNT}] "
            f"batch n={n}  dist mean={d_mean:.2f} min={d_min:.2f} max={d_max:.2f}  "
            f"promoted={n_up} demoted={n_dn}  "
            f"levels mean={tl_mean:.4f} max={tl_max_v} nonzero={tl_nonzero}/{tl_total}",
            flush=True,
        )

    return torch.mean(terrain.terrain_levels.float())


# ---------------------------------------------------------------------------
# Push-disturbance curriculum (same as R8)
# ---------------------------------------------------------------------------
def push_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids,
) -> torch.Tensor:
    """Ramp push_robot event: (0.2 m/s, 30 s) → (0.5 m/s, 16 s) as level rises."""
    terrain: TerrainImporter = env.scene.terrain
    mean_level = torch.mean(terrain.terrain_levels.float())

    ramp = torch.clamp(mean_level, min=0.0, max=1.0).item()

    boot_vel = 0.2
    paper_vel = 0.5
    boot_interval = 30.0
    paper_interval = 16.0

    vel_max = boot_vel + ramp * (paper_vel - boot_vel)
    interval_s = boot_interval + ramp * (paper_interval - boot_interval)

    try:
        push_term = env.event_manager.get_term_cfg("push_robot")
        push_term.params["velocity_range"] = {
            "x": (-vel_max, vel_max),
            "y": (-vel_max, vel_max),
        }
        push_term.interval_range_s = (interval_s, interval_s)
    except (AttributeError, KeyError):
        try:
            push_term = env.event_manager._term_cfgs[
                env.event_manager._term_names.index("push_robot")
            ]
            push_term.params["velocity_range"] = {
                "x": (-vel_max, vel_max),
                "y": (-vel_max, vel_max),
            }
            push_term.interval_range_s = (interval_s, interval_s)
        except Exception:
            pass

    return torch.tensor(vel_max, device=env.device)


# ---------------------------------------------------------------------------
# Tracking-weight curriculum (R10 — ramp 4.0 → 2.0 paper as level rises)
# ---------------------------------------------------------------------------
def tracking_weight_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids,
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    """Ramp track_lin_vel_xy weight from bootstrap 4.0 → paper 2.0."""
    terrain: TerrainImporter = env.scene.terrain
    mean_level = torch.mean(terrain.terrain_levels.float()).item()

    ramp = min(max((mean_level - 1.0) / 2.0, 0.0), 1.0)
    boot_weight = 4.0
    paper_weight = 2.0
    new_weight = boot_weight + ramp * (paper_weight - boot_weight)

    try:
        term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        term_cfg.weight = new_weight
        env.reward_manager.set_term_cfg(reward_term_name, term_cfg)
    except Exception:
        pass

    return torch.tensor(new_weight, device=env.device)
