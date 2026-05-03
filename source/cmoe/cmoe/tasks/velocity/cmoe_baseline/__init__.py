# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 1 baseline task registration.

Registers two gym task IDs:
- CMoE-Baseline-Phase1-v0      : training task (4096 envs, randomized cmd)
- CMoE-Baseline-Phase1-Play-v0 : eval task (32 envs)
"""

import gymnasium as gym

from . import agents

gym.register(
    id="CMoE-Baseline-Phase1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cmoe_baseline_env_cfg:CmoeBaselineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_baseline_cfg:CmoeBaselinePPORunnerCfg",
    },
)

gym.register(
    id="CMoE-Baseline-Phase1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cmoe_baseline_env_cfg:CmoeBaselineEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_baseline_cfg:CmoeBaselinePPORunnerCfg_PLAY",
    },
)
