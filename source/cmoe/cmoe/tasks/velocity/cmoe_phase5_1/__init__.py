# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 5.1 task registration — Phase 4 + 8 paper terrains + terrain curriculum.

注册:
- CMoE-Phase5-1-Terrains-v0      : 训练
- CMoE-Phase5-1-Terrains-Play-v0 : 评估

Phase 5.1 = Phase 4 (architecture) + paper §IV-A 的 8 种地形 + Rudin curriculum

NEW vs Phase 4 (env-level):
  1. terrain: plane → generator (CMOE_TERRAINS_CFG, 9 sub-terrains)
  2. CurriculumCfg: terrain_levels_vel (Rudin [37] 标准升级机制)
  3. episode_length_s: 10.0 → 20.0 (paper §V-A)
  4. feet_edge reward (-1.0, gated to Hurdle/Gap, paper §IV-C)
  5. foot_height_scanner_left / right (新 sensor, 给 feet_edge 用)

NOT changed (留给 Phase 5.2 / 5.3):
  - 域随机化 (摩擦/质量/电机/推力)
  - velocity command curriculum
  - elevation noise (salt-pepper, edge chamfer)

Architecture: 与 Phase 4 完全相同 (VAE + AE + MoE 5 experts + SwAV)
  → 仍用 phase4_inject.py 的 inject_moe_swav

Warm-start: from Phase 4 checkpoint
  Phase 4 ckpt 有完整的 actor/critic/gating/swav, architecture 不变
  → Phase 4 patched_load 直接走 "Phase 4 checkpoint (resume)" 路径就行
"""

import gymnasium as gym

gym.register(
    id="CMoE-Phase5-1-Terrains-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cmoe_phase5_1_env_cfg:CmoePhase5_1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_phase5_1_cfg:CmoePhase5_1PPORunnerCfg",
    },
)

gym.register(
    id="CMoE-Phase5-1-Terrains-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cmoe_phase5_1_env_cfg:CmoePhase5_1EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_phase5_1_cfg:CmoePhase5_1PPORunnerCfg_PLAY",
    },
)
