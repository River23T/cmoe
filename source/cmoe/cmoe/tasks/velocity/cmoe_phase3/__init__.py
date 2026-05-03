# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 3 task registration — Phase 2b + AE 高度图编码器.

注册两个 gym task ID:
- CMoE-Phase3-AE-Actor-v0      : 训练任务
- CMoE-Phase3-AE-Actor-Play-v0 : 评估任务

Phase 3 = Phase 2b + AE 高度图编码器:
  - 加 elevation map sensor (paper §IV-A: 0.7m × 1.1m, resolution 0.1m → 96 dim)
  - 加 AE estimator: e_t (96) → z_E_t (32) (paper §III-C, Eq. 5)
  - actor 输入: 99 + 3 + 32 + 32 = 166 dim (新增 z_E)
  - critic 输入: 99 + 32 = 131 dim (paper Fig.3: critic 也接 z_E)
  - 仍是平地 (Phase 5 才引入复杂地形)
  - warm-start from Phase 2b checkpoint (LoRA-style: 新增列 = 0)
"""

import gymnasium as gym

gym.register(
    id="CMoE-Phase3-AE-Actor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cmoe_phase3_env_cfg:CmoePhase3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_phase3_cfg:CmoePhase3PPORunnerCfg",
    },
)

gym.register(
    id="CMoE-Phase3-AE-Actor-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cmoe_phase3_env_cfg:CmoePhase3EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_phase3_cfg:CmoePhase3PPORunnerCfg_PLAY",
    },
)
