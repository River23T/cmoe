# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 2b task registration — Phase 2a + VAE 输出接到 actor.

注册两个 gym task ID:
- CMoE-Phase2b-VAE-Actor-v0      : 训练任务
- CMoE-Phase2b-VAE-Actor-Play-v0 : 评估任务

Phase 2b = Phase 2a (VAE estimator-only) + VAE 输出 v_pred + z_H 接到 actor

关键变化:
  - actor 输入 99 → 134 dim (新增 v_pred(3) + z_H(32) = 35 dim)
  - critic 仍 99 dim (paper Fig 3 critic 不连 VAE)
  - 修复 VaeHistoryCfg.history_length = 5 (Phase 2a 用的 4 不匹配)
  - 从 Phase 2a checkpoint warm-start (walking 已验证 OK)
"""

import gymnasium as gym

# 同样不用 `from . import agents`, 用静态字符串 (避免 task discovery 时的循环 import 错误)

gym.register(
    id="CMoE-Phase2b-VAE-Actor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cmoe_phase2b_env_cfg:CmoePhase2bEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_phase2b_cfg:CmoePhase2bPPORunnerCfg",
    },
)

gym.register(
    id="CMoE-Phase2b-VAE-Actor-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cmoe_phase2b_env_cfg:CmoePhase2bEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_phase2b_cfg:CmoePhase2bPPORunnerCfg_PLAY",
    },
)
