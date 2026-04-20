# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO runner config for CMoE G1 velocity training.

Paper IV-A parameters:
  - 4096 parallel environments
  - 20,000 epochs (iterations)
  - 5 experts (handled in custom_classes/models/, not here)
  - elevation map 0.7m x 1.1m (handled in scene cfg)
  - contrastive learning: num_prototype=32, temperature=0.2
    (handled in custom_classes/algorithms/ppo.py)
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class CmoePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration matching CMoE paper IV-A."""

    # --- runner ---
    num_steps_per_env = 24           # rollout length per env per iteration
    max_iterations = 20000           # paper: "20,000 epochs"
    save_interval = 500              # save checkpoint every 500 iterations
    experiment_name = "cmoe_g1_velocity"
    run_name = ""
    logger = "tensorboard"

    # --- policy network ---
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # --- PPO algorithm ---
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class CmoePPORunnerCfg_PLAY(CmoePPORunnerCfg):
    """Play/evaluation variant - no training, just inference."""
    resume = True
