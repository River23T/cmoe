# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 2b env cfg — Phase 2a + VAE 输出接到 actor.

============================================================================
Phase 2b = Phase 2a (VAE estimator-only) + VAE 输出 v_pred + z_H 接到 actor
============================================================================

继承自 Phase 2a, 关键差异:
  - PolicyCfg history_length = 5 → actor 输入 99 × 5 = 495-dim (含 5 帧 history)
  - 但 inject 中 VAEAugmentedActor 内部:
    1. 取最近一帧 99 dim (= phase 1.5 obs)
    2. 取整个 5 帧 移除 base_lin_vel = 96 × 5 = 480 dim (paper Eq 2 history)
    3. VAE encode → v_pred(3), z_H(32)
    4. cat([99, 3, 32]) = 134 dim → actor_mlp
  - critic 仍是 99 dim (paper Fig.3 critic 不连 VAE)

为什么 PolicyCfg.history_length=5 而不是单独的 VaeHistoryCfg?
  - 旧版 rsl_rl 的 ActorCritic 不正确处理 obs_groups 字典 (实测 log 显示
    actor MLP 创建为 99 dim 而不是 99+480=579 dim, 即使 obs_groups['actor'] 配了
    ['policy', 'vae_history']). 必须把 vae history 合进 obs.policy 才生效.

Paper Eq. 2: o_t = [ω_t (3), g_t (3), c_v (3), θ_t (29), θ̇_t (29), a_{t-1} (29)]
  = 96 dim per timestep (NO base_lin_vel — VAE 要预测的)

IsaacLab 实测: history_length=N 输出 N 帧 concat (不是 N+1).
  → history_length=5 输出 99×5 = 495 dim ✓

============================================================================
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from source.cmoe.cmoe.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from cmoe.tasks.velocity.cmoe import mdp


# ============================================================
# Scene (与 Phase 1.5 baseline 完全相同 — 平地)
# ============================================================
@configclass
class Phase2bSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                     f"TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.5, size=[0.1, 0.1]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.0,
        history_length=0,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
                         f"kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ============================================================
# Commands (与 Phase 1.5 完全相同)
# ============================================================
@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
    )


# ============================================================
# Actions (与 Phase 1.5 完全相同)
# ============================================================
@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


# ============================================================
# Observations
# ============================================================
@configclass
class ObservationsCfg:
    """Phase 2 obs groups:
      - policy:       99-dim, no history       (与 Phase 1.5 完全相同, 给 actor 用)
      - critic:       99-dim, no history       (与 Phase 1.5 完全相同, 给 critic 用)
      - vae_history:  96-dim × 5 = 480-dim     (NEW for Phase 2, 给 VAE 训练用)
      - vae_target:   96-dim, current obs only (NEW for Phase 2, VAE 重建目标)
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Phase 2b — actor 输入 99×5 = 495 dim (5 帧 history, 含 base_lin_vel).
        
        最近一帧 99 dim = Phase 1.5 obs (warm-start 兼容).
        前 4 帧 = history 给 VAE 用.
        
        VAEAugmentedActor 内部:
          - 取 x[:, -99:] 作为 actor MLP 直接输入 (warm-start 兼容)
          - 取 x[:, :] 全部 5 帧, 移除 base_lin_vel → 96×5=480 给 VAE
        """
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,      noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel,      scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel     = ObsTerm(func=mdp.joint_pos_rel,     noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel     = ObsTerm(func=mdp.joint_vel_rel,     scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action       = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            # IsaacLab history_length=5 输出 5 帧 concat = 99×5 = 495 dim
            # (实测自 Phase 2a log: history_length=5 给 99×5=495 dim)
            self.history_length    = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """与 Phase 1.5 完全相同 — critic 输入 99-dim, no noise.
        
        critic obs 前 3 dim = base_lin_vel ground truth, 用于 VAE supervision.
        """
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel     = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel     = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action       = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length    = 0
            self.concatenate_terms = True

    # 注: Phase 2b 不需要 VaeHistoryCfg / VaeTargetCfg
    # vae_history 从 obs.policy 内部提取 (5 帧, 移除 base_lin_vel)
    # vae_target 用 obs.policy 最新一帧的非 base_lin_vel 部分 (96 dim)

    policy: PolicyCfg            = PolicyCfg()
    critic: CriticCfg            = CriticCfg()


# ============================================================
# Events (与 Phase 1.5 完全相同 — 平地无 push, 无关节随机化)
# ============================================================
@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


# ============================================================
# Rewards (与 Phase 1.5 v7 完全相同 — 19/20 paper + action_rate sim fix + 3 EXTRA)
# ============================================================
@configclass
class PaperRewardsCfg:
    """Phase 1.5 v7 已验证的 reward 配置 — 不动."""

    # ===== 20 PAPER TABLE II (19 paper-exact + 1 sim-to-sim) =====
    velocity_tracking = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    yaw_tracking = RewTerm(
        func=mdp.track_ang_vel_z_exp_l1, weight=2.0,
        params={"command_name": "base_velocity"},
    )
    z_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    rp_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-15.0,
        params={"target_height": 0.78, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link")},
    )
    collision = RewTerm(
        func=mdp.undesired_contacts, weight=-15.0,
        params={
            "threshold": 0.1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )
    feet_distance = RewTerm(
        func=mdp.feet_lateral_distance, weight=0.8,
        params={
            "d_min": 0.20, "d_max": 0.50,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time, weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_ground_parallel = RewTerm(
        func=mdp.feet_ground_parallel, weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
            "foot_half_length": 0.10, "foot_half_width": 0.04,
        },
    )
    hip_dof_error = RewTerm(
        func=mdp.joint_deviation_l2, weight=-0.5,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"],
        )},
    )
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-5.0e-4)
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    # action_rate: paper -0.3 → -5e-3 (Phase 1.5 v6 验证的 sim-to-sim fix)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5.0e-3)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits, weight=-1.0, params={"soft_ratio": 1.0},
    )
    torque_limits = RewTerm(func=mdp.applied_torque_limits_paper, weight=-1.0)
    # feet_edge: 平地不需要 (paper §IV-C: only Hurdle/Gap)

    # ===== 3 EXTRA — Phase 1.5 v7 已验证 =====
    # E1: arms 关节偏离惩罚
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l2, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
        )},
    )
    # E2: waist 关节偏离惩罚
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l2, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        )},
    )
    # E3: no_fly anti-bunny-hop
    no_fly = RewTerm(
        func=mdp.desired_contacts, weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "threshold": 1.0,
        },
    )


# ============================================================
# Terminations (与 Phase 1.5 完全相同)
# ============================================================
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0},
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_link", "pelvis"]),
            "threshold": 1.0,
        },
    )


# ============================================================
# Env cfg
# ============================================================
@configclass
class CmoePhase2bEnvCfg(ManagerBasedRLEnvCfg):
    """Phase 2 = Phase 1.5 v7 + VAE estimator (estimator-only).

    ALL env settings (scene/commands/actions/rewards/terminations/events)
    are IDENTICAL to Phase 1.5 v7. The ONLY new thing is the VAE estimator
    that runs in the background, trained on `vae_history` obs group.
    """

    scene: Phase2bSceneCfg          = Phase2bSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg  = ObservationsCfg()
    actions: ActionsCfg            = ActionsCfg()
    commands: CommandsCfg          = CommandsCfg()
    rewards: PaperRewardsCfg       = PaperRewardsCfg()
    terminations: TerminationsCfg  = TerminationsCfg()
    events: EventCfg               = EventCfg()

    def __post_init__(self):
        self.decimation       = 4
        self.episode_length_s = 10.0   # Phase 1.5 一样
        self.sim.dt                              = 0.005
        self.sim.render_interval                 = self.decimation
        self.sim.physics_material                = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 15
        self.scene.contact_forces.update_period  = self.sim.dt


@configclass
class CmoePhase2bEnvCfg_PLAY(CmoePhase2bEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs   = 32
        self.episode_length_s = 20.0
        self.commands.base_velocity.rel_standing_envs = 0.0
