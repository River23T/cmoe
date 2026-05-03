# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 1 baseline PPO config — STANDARD MLP, NO MoE, NO estimators.

Compatible with isaaclab_rl 0.4.7 + rsl_rl 3.3.0.

This is a pure standard PPO with a simple MLP. NO custom injection at all.
The goal is to verify that with reference-tuned rewards, a vanilla MLP can
walk on flat ground. If THIS fails, the rewards are wrong.

EXPECTED RESULT after ~3000 iterations on flat ground:
- Mean reward: 30+
- Mean episode length: ~500/500 (full 10s episodes)
- error_vel_xy: <0.3 m/s
- base_contact termination: <5%
- avg_dist on eval: 7-8m in 20s @ cmd=0.8 m/s

If we get this, the reward design is validated. Then we can layer on:
- Phase 2: VAE + AE estimators (still flat)
- Phase 3: MoE actor-critic + contrastive (still flat or simple terrain)
- Phase 4: All 9 paper terrains
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class CmoeBaselinePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 1 baseline runner config — pure standard PPO."""

    num_steps_per_env = 24
    max_iterations    = 3000          # ← Phase 1 only needs 3k for flat-ground walking
    save_interval     = 200
    experiment_name   = "cmoe_baseline_phase1"
    run_name          = ""
    logger            = "tensorboard"

    # Standard obs groups: actor sees policy, critic sees critic. NO elevation.
    obs_groups = {
        "actor":  ["policy"],
        "critic": ["critic"],
    }

    # Standard ActorCritic with MLP — no MoE, no estimators.
    policy = RslRlPpoActorCriticCfg(
        init_noise_std            = 1.0,
        actor_obs_normalization   = False,
        critic_obs_normalization  = False,
        actor_hidden_dims         = [512, 256, 128],
        critic_hidden_dims        = [512, 256, 128],
        activation                = "elu",
    )

    # Standard rsl_rl PPO. NO cmoe_cfg — no injection happens.
    algorithm = RslRlPpoAlgorithmCfg(
        class_name                       = "PPO",
        value_loss_coef                  = 1.0,
        use_clipped_value_loss           = True,
        clip_param                       = 0.2,
        entropy_coef                     = 0.01,
        num_learning_epochs              = 5,
        num_mini_batches                 = 4,
        learning_rate                    = 1.0e-3,
        schedule                         = "adaptive",
        gamma                            = 0.99,
        lam                              = 0.95,
        desired_kl                       = 0.01,
        max_grad_norm                    = 1.0,
    )


@configclass
class CmoeBaselinePPORunnerCfg_PLAY(CmoeBaselinePPORunnerCfg):
    """Play/evaluation variant."""
    resume = True
