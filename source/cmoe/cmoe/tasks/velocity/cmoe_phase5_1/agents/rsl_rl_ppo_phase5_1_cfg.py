# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 5.1 PPO config — Phase 4 architecture (MoE+SwAV) + paper terrains.

Architecture: 100% same as Phase 4 (5 MoE experts + shared gating + SwAV).
唯一变化是 experiment_name (单独 log dir) 和 phase tag = "5.1".

train.py 在 phase_tag == "5.1" 时直接 reuse Phase 4 inject:
  inject_moe_swav (与 Phase 4 完全相同的 architecture)

Phase 4 checkpoint warm-start 直接走 Phase 4 patched_load 的 "resume" 路径
(architecture 一模一样, 不需要任何 weight copy/expansion).
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class CmoePhase5_1PpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    cmoe_cfg: dict | None = None


@configclass
class CmoePhase5_1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 5.1 runner — same architecture as Phase 4, new env (8 terrains)."""

    num_steps_per_env = 24
    max_iterations    = 5000     # 比 Phase 4 多, 因为多地形需要更多训练
    save_interval     = 200
    experiment_name   = "cmoe_phase5_1_terrains"
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

    algorithm = CmoePhase5_1PpoAlgorithmCfg(
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

        # Phase 5.1: phase tag "5.1" 让 train.py 走特殊路由 (reuse Phase 4 inject)
        # cmoe_cfg 内容与 Phase 4 完全相同
        cmoe_cfg={
            "phase": "5.1",
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
            "moe": {
                "num_experts": 5,
                "expert_hidden_dims": [512, 256, 128],
                "gate_input_dim": 32,
                "gate_hidden_dims": [64, 32],
                "actor_obs_dim": 166,
                "critic_obs_dim": 131,
                "action_dim": 29,
            },
            "contrastive": {
                "num_prototypes": 32,
                "temperature": 0.2,
                "projection_dim": 64,
                "sinkhorn_iters": 3,
                "loss_weight": 0.1,
                "learning_rate": 1.0e-3,
            },
        },
    )


@configclass
class CmoePhase5_1PPORunnerCfg_PLAY(CmoePhase5_1PPORunnerCfg):
    resume = True
