# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 4 task registration — Phase 3 + MoE actor/critic + SwAV contrastive learning.

注册两个 gym task ID:
- CMoE-Phase4-MoE-CL-v0      : 训练任务
- CMoE-Phase4-MoE-CL-Play-v0 : 评估任务

Phase 4 = Phase 3 + MoE + Contrastive (paper §III-D §III-E):
  - 把 actor 替换为 MoE actor (5 experts, 共享 gating network)
  - 把 critic 替换为 MoE critic (5 experts, 共享 actor 的 gating network)
  - Gating input = z_E (paper Fig.3 strict, 32 dim)
  - Action = Σ softmax(g_i) * μ_i  (Eq. 6)
  - SwAV contrastive: gate logits ↔ z_E (Eq. 7-8)
  - num_prototype=32, τ=0.2 (paper §IV-A)
  - 仍是平地, Phase 5 才引入复杂地形
  - warm-start from Phase 3 checkpoint:
    * 5 个 actor experts 全部 = Phase 3 single actor (重复 5 次)
    * 5 个 critic experts 全部 = Phase 3 single critic (重复 5 次)
    * Gating: random init
    * SwAV prototypes: random init
"""

import gymnasium as gym

gym.register(
    id="CMoE-Phase4-MoE-CL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cmoe_phase4_env_cfg:CmoePhase4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_phase4_cfg:CmoePhase4PPORunnerCfg",
    },
)

gym.register(
    id="CMoE-Phase4-MoE-CL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cmoe_phase4_env_cfg:CmoePhase4EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_phase4_cfg:CmoePhase4PPORunnerCfg_PLAY",
    },
)
