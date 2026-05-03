# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 4 env cfg — Phase 3 + MoE actor/critic + SwAV contrastive learning.

============================================================================
Phase 4 = Phase 3 + MoE actor-critic + SwAV (paper §III-D §III-E)
============================================================================

Env-level changes vs Phase 3: NONE.
  - Scene、Observations、Rewards、Terminations 100% 与 Phase 3 相同
  - 唯一变化在 actor/critic 架构, 由 phase4_inject.py 处理:
    * actor 由 single MLP → MoE actor (5 experts + gating)
    * critic 由 single MLP → MoE critic (5 experts + 共享 gating)
    * 加 SwAV contrastive loss (gate ↔ z_E, paper Eq. 7-8)

仍是平地训练 (Phase 5 才引入复杂地形).
警告: 平地下 z_E 几乎是常数, gating 网络无信号可学
  → 5 experts 在平地下学不到 specialization
  → 但 Phase 4 主要是把 MoE 架构搭好并验证 walking 不被破坏
  → Phase 5 引入地形后, MoE 才真正发挥作用

obs.policy = 975 dim (per-term flatten, 8 terms × 5 frames history)
obs.critic = 195 dim (8 terms, no history)
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
# Scene — Phase 2b 完整继承 + 新增 elevation_scanner
# ============================================================
@configclass
class Phase4SceneCfg(InteractiveSceneCfg):
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

    # ---- height_scanner: Phase 2b 的旧 sensor, 用于 base_height reward (1 ray) ----
    # 不动: Phase 2b reward 完全保留
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

    # ---- NEW: elevation_scanner — paper §IV-A: 0.7m × 1.1m grid ----
    # GridPatternCfg(resolution=0.1, size=[1.1, 0.7]):
    #   ray count = (1.1/0.1 + 1) × (0.7/0.1 + 1) = 12 × 8 = 96 ray
    # offset z=20m: 从 torso 上方 20m 向下 raycast (穿过 torso 击中地面)
    # ray_alignment="yaw": 跟随机器人 yaw 旋转, 但不跟随 pitch/roll (paper § IV-B)
    elevation_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.1, 0.7]),
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
# Commands (与 Phase 2b 完全相同)
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
# Actions (与 Phase 2b 完全相同)
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
    """Phase 4 obs groups:
      - policy: Phase 2b 7 terms + elevation = 8 terms
                with history_length=5, 总 dim = 99×5 + 96×5 = 975
      - critic: Phase 2b 7 terms + elevation = 8 terms (no history)
                总 dim = 99 + 96 = 195
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Phase 4 — actor 输入 (99+96)×5 = 975 dim.
        
        layout (per-term flatten, IsaacLab convention):
          [base_lin_vel(15), base_ang_vel(15), gravity(15), cmd(15),
           jpos(145), jvel(145), last_act(145), elevation(480)]
          total = 60 + 435 + 480 = 975 dim
        
        AE 用最新一帧 elevation (96 dim) 编码为 z_E (32 dim).
        VAE 用 5 帧 [base_lin_vel..., last_act] (no elevation) → 480 dim.
        """
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,      noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel,      scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel     = ObsTerm(func=mdp.joint_pos_rel,     noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel     = ObsTerm(func=mdp.joint_vel_rel,     scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action       = ObsTerm(func=mdp.last_action)
        # NEW: elevation map (paper §III-A)
        # Use IsaacLab's built-in height_scan func: 返回 raycaster 命中点 z
        # clip = (-1.0, 1.0): 限制深度范围 [-1m, 1m] (paper-style preprocessing)
        elevation         = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("elevation_scanner")},
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),  # 模拟 lidar 噪声 (paper § IV-B)
        )

        def __post_init__(self):
            self.history_length    = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """critic 输入 99 + 96 = 195 dim (no history, no noise).
        
        critic 也接 elevation: paper Fig.3 critic 接的是 ground-truth e_t
        (而非估计的 z_E_t), AE 编码后 critic 自己学习消化.
        
        critic obs 前 3 dim = base_lin_vel ground truth (用于 VAE supervision).
        critic obs 后 96 dim = elevation ground truth (用于 AE supervision).
        """
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel     = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel     = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action       = ObsTerm(func=mdp.last_action)
        # NEW: elevation map ground truth (no noise — privileged)
        elevation         = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("elevation_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.history_length    = 0
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ============================================================
# Events (与 Phase 2b 完全相同)
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
# Rewards (与 Phase 2b 完全相同 — 19/20 paper + action_rate sim fix + 3 EXTRA)
# ============================================================
@configclass
class PaperRewardsCfg:
    """Phase 1.5 v7 / Phase 2b 已验证的 reward 配置 — 不动."""

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
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5.0e-3)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits, weight=-1.0, params={"soft_ratio": 1.0},
    )
    torque_limits = RewTerm(func=mdp.applied_torque_limits_paper, weight=-1.0)

    # 3 EXTRA — Phase 1.5 v7 已验证 (用户保留)
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
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l2, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        )},
    )
    no_fly = RewTerm(
        func=mdp.desired_contacts, weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "threshold": 1.0,
        },
    )


# ============================================================
# Terminations (与 Phase 2b 完全相同)
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
class CmoePhase4EnvCfg(ManagerBasedRLEnvCfg):
    """Phase 4 = Phase 2b + AE 高度图编码器.

    All env settings except scene (added elevation_scanner) and observations
    (added elevation term) are IDENTICAL to Phase 2b. Reward and termination
    are 100% unchanged so Phase 2b warm-start preserves walking.
    """

    scene: Phase4SceneCfg          = Phase4SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg  = ObservationsCfg()
    actions: ActionsCfg            = ActionsCfg()
    commands: CommandsCfg          = CommandsCfg()
    rewards: PaperRewardsCfg       = PaperRewardsCfg()
    terminations: TerminationsCfg  = TerminationsCfg()
    events: EventCfg               = EventCfg()

    def __post_init__(self):
        self.decimation       = 4
        self.episode_length_s = 10.0
        self.sim.dt                              = 0.005
        self.sim.render_interval                 = self.decimation
        self.sim.physics_material                = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 15
        self.scene.contact_forces.update_period  = self.sim.dt


@configclass
class CmoePhase4EnvCfg_PLAY(CmoePhase4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs   = 32
        self.episode_length_s = 20.0
        self.commands.base_velocity.rel_standing_envs = 0.0
