# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 4 PPO config — MoE actor + MoE critic + SwAV contrastive.

Phase 4 vs Phase 3 的关键差异 (architecture-only, 由 inject 处理):
  - actor: Phase 3 single MLP → MoE (5 experts + 共享 gating)
  - critic: Phase 3 single MLP → MoE (5 experts + 共享 gating)
  - 新增 SwAV contrastive loss (gate ↔ z_E)

PPO 超参 与 Phase 3 完全相同 (LR/clip/KL/etc).

Paper §IV-A 参数:
  - num_experts = 5
  - num_prototypes = 32
  - temperature = 0.2
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class CmoePhase4PpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """扩展 PPO config 支持 cmoe_cfg field."""
    cmoe_cfg: dict | None = None


@configclass
class CmoePhase4PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 4 runner config — MoE actor + MoE critic + SwAV contrastive."""

    num_steps_per_env = 24
    max_iterations    = 3000
    save_interval     = 200
    experiment_name   = "cmoe_phase4_moe_swav"
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
        # 这些 hidden dims 仍按 single-MLP 报给 rsl_rl, 之后 inject 替换
        actor_hidden_dims         = [512, 256, 128],
        critic_hidden_dims        = [512, 256, 128],
        activation                = "elu",
    )

    algorithm = CmoePhase4PpoAlgorithmCfg(
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

        # Phase 4 inject config: VAE + AE + MoE + SwAV
        # train.py 看到 phase=="4" 走 phase4 inject 路径
        cmoe_cfg={
            "phase": "4",
            "estimator": {
                "learning_rate": 1.0e-3,
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
            # MoE config (paper §III-D)
            "moe": {
                "num_experts": 5,                          # paper §IV-A
                "expert_hidden_dims": [512, 256, 128],     # 与 Phase 3 single actor 一致
                "gate_input_dim": 32,                      # = AE latent dim
                "gate_hidden_dims": [64, 32],
                # actor obs dim = 99 (proprio) + 3 (v_pred) + 32 (z_H) + 32 (z_E) = 166
                "actor_obs_dim": 166,
                # critic obs dim = 99 (proprio) + 32 (z_E) = 131
                "critic_obs_dim": 131,
                "action_dim": 29,
            },
            # SwAV contrastive config (paper §III-E, §IV-A)
            "contrastive": {
                "num_prototypes": 32,        # paper §IV-A
                "temperature": 0.2,          # paper §IV-A
                "projection_dim": 64,
                "sinkhorn_iters": 3,
                "loss_weight": 0.1,          # weight in total loss
                "learning_rate": 1.0e-3,
            },
        },
    )


@configclass
class CmoePhase4PPORunnerCfg_PLAY(CmoePhase4PPORunnerCfg):
    resume = True
