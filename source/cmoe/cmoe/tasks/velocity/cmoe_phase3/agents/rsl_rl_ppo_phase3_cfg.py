# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 3 PPO config — actor 接 VAE+AE 输出.

Phase 3 vs Phase 2b 的关键差异:
  - actor input dim: 134 → 166 (新增 z_E(32))
  - critic input dim: 99 → 131 (新增 z_E(32) — paper Fig.3 critic 也接)

所有其他 PPO 超参 与 Phase 2b 完全相同 (LR/clip/KL/etc).
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class CmoePhase3PpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """扩展 PPO config 支持 cmoe_cfg field."""
    cmoe_cfg: dict | None = None


@configclass
class CmoePhase3PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 3 runner config — actor & critic 接 VAE + AE."""

    num_steps_per_env = 24
    max_iterations    = 3000
    save_interval     = 200
    experiment_name   = "cmoe_phase3_vae_ae_actor"
    run_name          = ""
    logger            = "tensorboard"

    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std            = 1.0,
        actor_obs_normalization   = False,
        critic_obs_normalization  = False,
        actor_hidden_dims         = [512, 256, 128],
        critic_hidden_dims        = [512, 256, 128],
        activation                = "elu",
    )

    algorithm = CmoePhase3PpoAlgorithmCfg(
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

        # Phase 3 inject config: VAE estimator + AE estimator + actor/critic extension
        # train.py 看到 phase=="3" 走 phase3 inject 路径
        cmoe_cfg={
            "phase": "3",
            # actor 在 134 dim 基础上加 z_E(32) = 166 dim
            "actor_extra_dim_z_e": 32,
            # critic 在 99 dim 基础上加 z_E(32) = 131 dim
            "critic_extra_dim_z_e": 32,
            "estimator": {
                "learning_rate": 1.0e-3,
                # VAE: 与 Phase 2b 完全相同 (用 Phase 2b checkpoint warm-start)
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
                # NEW: AE for elevation map (paper §III-C, Eq. 5)
                # input_dim = 96 (12×8 grid)
                # latent_dim = 32 (与 z_H 同维, paper §III-A)
                "ae": {
                    "input_dim": 96,
                    "latent_dim": 32,
                    "encoder_hidden_dims": [128, 64],
                    "decoder_hidden_dims": [64, 128],
                    "activation": "elu",
                },
            },
        },
    )


@configclass
class CmoePhase3PPORunnerCfg_PLAY(CmoePhase3PPORunnerCfg):
    resume = True
