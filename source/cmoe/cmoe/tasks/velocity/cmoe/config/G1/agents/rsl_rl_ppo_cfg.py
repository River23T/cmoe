# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO runner config for CMoE G1 velocity training (R4 — paper-strict).

Compatibility: isaaclab_rl==0.4.7 (IsaacLab 2.3.2) + rsl-rl-lib==3.3.0.
Uses standard PPO class + post-construction injection for CMoE components.

=============================================================
R4 HYPERPARAMETERS (paper §IV-A compliance)
=============================================================

Iteration 418-438 log shows R3 successfully bootstrapped the policy:
  - reward climbing 11 -> 14
  - episode length 471 -> 517 (walking stably through most of 20s)
  - bad_orientation = 0% (solved)
  - BUT curricula frozen: terrain_levels=0.0002, cmd_levels=0.5

R4 changes (toward paper compliance, now that boot-strap succeeded):

  * entropy_coef 0.005 kept (matches paper §IV-A; was the R2->R3 value)
  * learning_rate 2e-4 -> 1e-3 with adaptive KL schedule.
    Paper uses 1e-3 (standard PPO for locomotion).  The KL-adaptive
    scheduler will shrink it if updates become too aggressive.
  * desired_kl 0.01 kept (paper-default tolerance).
  * contrastive_weight 0.05 -> 0.1. Paper §III-E gives the SwAV loss
    equal weighting with PPO losses (no explicit scaling); 0.1 is a
    conservative middle ground. Robot now walks so gate activations
    have real signal; contrastive can begin to differentiate experts.
  * estimator learning_rate 5e-4 -> 1e-3. Paper uses 1e-3; at iter
    400+ the VAE+AE latents should be stable enough.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class CmoePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Extended PPO algorithm config with CMoE-specific fields.

    ``cmoe_cfg`` is popped from the dict before passing to rsl_rl's PPO,
    then used by ``inject_cmoe()`` after runner construction.
    """
    cmoe_cfg: dict | None = None


@configclass
class CmoePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration matching CMoE paper §IV-A (R4 paper-strict)."""

    # --- runner ---
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "cmoe_g1_velocity"
    run_name = ""
    logger = "tensorboard"

    # --- observation group mapping ---
    obs_groups = {
        "actor": ["policy", "elevation"],
        "critic": ["critic", "elevation"],
    }

    # --- policy network (old-style API for isaaclab_rl 0.4.7) ---
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # --- PPO algorithm: use STANDARD rsl_rl PPO, CMoE injected after ---
    algorithm = CmoePpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # R4: paper §IV-A value
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        # R4: 2e-4 -> 1e-3 (paper-typical for PPO locomotion, KL-adaptive)
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        # R4: paper §IV-A value
        desired_kl=0.01,
        max_grad_norm=1.0,
        # --- CMoE config (popped before PPO, injected after) ---
        cmoe_cfg={
            # Estimators: Paper III-C, Eq. 3-5
            "estimator": {
                # R4: 5e-4 -> 1e-3 (paper-typical estimator lr)
                "learning_rate": 1e-3,
                "vae": {
                    "obs_dim": 96,
                    "history_length": 4,
                    "latent_dim": 32,
                    "velocity_dim": 3,
                    "encoder_hidden_dims": [256, 128],
                    "decoder_hidden_dims": [128, 256],
                    "beta": 0.001,
                    "activation": "elu",
                },
                "ae": {
                    "input_dim": 96,
                    "latent_dim": 32,
                    "encoder_hidden_dims": [128, 64],
                    "decoder_hidden_dims": [64, 128],
                    "activation": "elu",
                },
            },
            # MoE architecture: Paper III-D, Eq. 6
            "moe": {
                "num_experts": 5,           # paper §IV-A
                "gate_input_dim": 32,
                "gate_hidden_dims": [64, 32],
                "init_noise_std": 1.0,
                "single_obs_dim": 96,
            },
            # Contrastive learning: Paper III-E, Eq. 7-8
            "contrastive": {
                "elevation_dim": 32,
                "projection_dim": 64,
                "num_prototypes": 32,        # paper §IV-A
                "temperature": 0.2,          # paper §IV-A
                "sinkhorn_iters": 3,
                # R4: 0.05 -> 0.1 (robot walks, gate outputs now meaningful)
                "weight": 0.1,
            },
        },
    )


@configclass
class CmoePPORunnerCfg_PLAY(CmoePPORunnerCfg):
    """Play/evaluation variant."""
    resume = True
