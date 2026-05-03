# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 2b PPO config — actor 接 VAE 输出.

Phase 2b 的关键差异 vs Phase 2a:
  - actor input dim: 99 → 134 (新增 v_pred(3) + z_H(32))
  - critic input dim: 99 (与 Phase 2a 相同, paper critic 不接 VAE)

但是 rsl_rl 的 ActorCritic 是从 obs_groups 拼接维度自动推断 input_dim 的.
  obs_groups = {"actor": ["policy"], "critic": ["critic"]}
  → actor 自动用 policy.shape[-1] = 99 创建第一层

我们的方案: 在 inject 中**手动改造 actor 第一层** in_features 99 → 134.
  - rsl_rl 创建 actor MLP 后, 替换 actor.0 (Linear) 为 Linear(134, 512)
  - 旧权重 [99, 512] 复制到新权重 [134, 512] 前 99 列, 后 35 列随机初始化
  - 在 act forward 时, 我们 wrap, concat [v_pred, z_H] 到 obs.policy 后调 actor
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class CmoePhase2bPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """扩展 PPO config 支持 cmoe_cfg field."""
    cmoe_cfg: dict | None = None


@configclass
class CmoePhase2bPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 2b runner config — actor 接 VAE."""

    num_steps_per_env = 24
    max_iterations    = 3000
    save_interval     = 200
    experiment_name   = "cmoe_phase2b_vae_actor"
    run_name          = ""
    logger            = "tensorboard"

    # actor 看 policy (5 帧 × 99 dim = 495 dim, 含 history)
    # critic 看 critic (99 dim, 单帧 privileged)
    # 用最小化 obs_groups 配置避免 rsl_rl 的 resolve_obs_groups quirk
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

    algorithm = CmoePhase2bPpoAlgorithmCfg(
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

        # Phase 2b: VAE estimator + actor injection
        # train.py 看到 "actor_inject" key 就走 phase2b 路径
        cmoe_cfg={
            "phase": "2b",                     # 标记 Phase 2b (区别于 Phase 2a)
            "actor_extra_dim": 35,             # v_pred(3) + z_H(32) = 35
            "estimator": {
                "learning_rate": 1.0e-3,
                "vae": {
                    "obs_dim": 96,
                    "history_length": 4,       # VAE 内部 encoder_input_dim = 96*(4+1) = 480
                    "latent_dim": 32,
                    "velocity_dim": 3,
                    "encoder_hidden_dims": [256, 128],
                    "decoder_hidden_dims": [128, 256],
                    "beta": 0.001,
                    "activation": "elu",
                },
            },
        },
    )


@configclass
class CmoePhase2bPPORunnerCfg_PLAY(CmoePhase2bPPORunnerCfg):
    resume = True
