# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CMoE curriculum functions — PAPER-STRICT (R-FINAL).

============================================================================
设计原则: 严格论文 §IV-A "curriculum learning mechanism".
============================================================================

论文 §IV-A 原文:
  "we employed a curriculum learning mechanism. Furthermore, since all
   terrains were trained in the same phase, to balance the difficulty
   of each terrain, we divided them into simple and complex categories.
   We then performed velocity command curriculum learning on complex
   terrains, gradually increasing the magnitude and direction of the
   velocity commands."

论文引用 [37] = Rudin et al. "Learning to walk in minutes using massively
parallel deep reinforcement learning". 该参考使用 Isaac Lab 标准 terrain
curriculum (size[0]/2 promote, 速度命令 demote).

R-FINAL changes vs R17:
  - REMOVED tracking_weight_curriculum (paper Table II 是固定 weight=2.0)
  - REMOVED push_curriculum (paper §IV-B 30 N every 16 s 是固定值)
  - terrain promote 1.5 m → size[0]/2 (Isaac Lab/Rudin [37] 标准)
  - terrain demote 改为速度命令模式 (Rudin [37] 标准)
  - 保留 lin_vel_cmd_levels (paper §IV-A 明确提及)

仅保留两个 curriculum (paper §IV-A 明确提及):
  1. terrain_levels_vel — Rudin [37] 标准
  2. lin_vel_cmd_levels — paper "speed command curriculum"

============================================================================
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


# 模块级诊断计数器
_DIAG_CALL_COUNT = 0
_DIAG_EVERY = 500


# ---------------------------------------------------------------------------
# Linear velocity command curriculum (paper §IV-A)
# ---------------------------------------------------------------------------
def lin_vel_cmd_levels(
    env: "ManagerBasedRLEnv",
    env_ids,
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    """Per-reset velocity-command curriculum.

    Paper §IV-A: "performed velocity command curriculum learning on complex
    terrains, gradually increasing the magnitude and direction of the velocity
    commands". 当 tracking reward 超过阈值时扩大命令范围。
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)

    # 处理 CurriculumManager 初始化时传入的 slice(None)
    is_full = isinstance(env_ids, slice)

    if not is_full and hasattr(env_ids, "__len__") and len(env_ids) > 0:
        if not isinstance(env_ids, torch.Tensor):
            env_ids_t = torch.as_tensor(list(env_ids), dtype=torch.long, device=env.device)
        else:
            env_ids_t = env_ids.to(dtype=torch.long, device=env.device)
        reward = torch.mean(
            env.reward_manager._episode_sums[reward_term_name][env_ids_t]
        ) / env.max_episode_length_s
    else:
        reward = torch.tensor(0.0, device=env.device)

    # 阈值: 80% of max reward (humanoid_locomotion 参考标准)
    upgrade_threshold = reward_term.weight * 0.8

    if reward > upgrade_threshold:
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
# Terrain-level curriculum (paper §IV-A 引用 [37] Rudin 标准)
# ---------------------------------------------------------------------------
def terrain_levels_vel(
    env: "ManagerBasedRLEnv",
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terrain-difficulty curriculum (Rudin [37] 标准实现).

    Promote: distance > size[0] / 2 (走过半个地形块就升级)
    Demote: distance < command_velocity * episode_length * 0.5
            (走的距离 < 应该走的一半就降级)
    """
    global _DIAG_CALL_COUNT

    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # 处理 slice(None) - 不能计算 per-env distance, 直接返回均值
    if isinstance(env_ids, slice):
        return torch.mean(terrain.terrain_levels.float())

    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(list(env_ids), dtype=torch.long, device=env.device)
    else:
        env_ids = env_ids.to(dtype=torch.long, device=env.device)

    if env_ids.numel() == 0:
        return torch.mean(terrain.terrain_levels.float())

    # 计算 robot 走的距离
    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )

    # 获取速度命令 (Rudin [37] 标准 demote 条件)
    command = env.command_manager.get_command("base_velocity")
    cmd_norm = torch.norm(command[env_ids, :2], dim=1)

    # ----- Rudin [37] 标准阈值 -----
    # Promote: 走过半个 terrain block (size[0]/2 = 4 m)
    # Demote: 走过的距离 < 命令距离的一半
    promote_distance = terrain.cfg.terrain_generator.size[0] / 2

    move_up = distance > promote_distance
    move_down = distance < cmd_norm * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    # 更新 env origins
    terrain.update_env_origins(env_ids, move_up, move_down)

    # 最高 level reset
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

    # 诊断 (每 _DIAG_EVERY 次)
    _DIAG_CALL_COUNT += 1
    if _DIAG_CALL_COUNT % _DIAG_EVERY == 0:
        n = int(env_ids.numel())
        d_mean = float(distance.mean().item()) if n > 0 else 0.0
        d_max = float(distance.max().item()) if n > 0 else 0.0
        n_up = int(move_up.sum().item())
        n_dn = int(move_down.sum().item())
        tl_mean = float(terrain.terrain_levels.float().mean().item())
        tl_max_v = int(terrain.terrain_levels.max().item())
        tl_nonzero = int((terrain.terrain_levels > 0).sum().item())
        tl_total = int(terrain.terrain_levels.numel())
        print(
            f"[CURR-FINAL diag #{_DIAG_CALL_COUNT}] "
            f"batch n={n}  dist mean={d_mean:.2f} max={d_max:.2f}  "
            f"promote_thr={promote_distance:.1f}  promoted={n_up} demoted={n_dn}  "
            f"levels mean={tl_mean:.4f} max={tl_max_v} nonzero={tl_nonzero}/{tl_total}",
            flush=True,
        )

    return torch.mean(terrain.terrain_levels.float())
