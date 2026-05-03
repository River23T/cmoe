# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 5.1 env cfg — Phase 4 + 8 paper terrains + terrain curriculum.

============================================================================
Phase 5.1 = Phase 4 + 8 paper terrains (paper Table I) + terrain curriculum
============================================================================

Architecture: 100% same as Phase 4 (5 MoE experts + shared gating + SwAV).
  - Reuses phase4_inject.inject_moe_swav for actor/critic injection
  - 仅 env 层面变化, inject 路径完全不变

Env changes vs Phase 4:
  1. Scene.terrain: plane → generator with CMOE_TERRAINS_CFG
     (paper §IV-A 8 sub-terrains: SlopeUp/Down, StairUp/Down, Gap, Hurdle,
      Discrete, Mix1, Mix2)
  2. Scene: + foot_height_scanner_left, foot_height_scanner_right (给 feet_edge)
  3. Rewards: + feet_edge (-1.0, gated to Gap/Hurdle, paper §IV-C)
  4. Curriculum: + terrain_levels_vel (Rudin [37], paper §IV-A)
  5. Episode length: 10s → 20s (paper §V-A)

NOT in this phase (后续 Phase 5.2 / 5.3):
  - velocity command curriculum
  - 域随机化 (摩擦/质量/电机/推力)
  - elevation noise (salt-pepper, edge chamfer)

Warm-start: Phase 4 checkpoint
  Phase 4 inject 创建的 actor/critic 架构与 Phase 5.1 完全一致 (obs.policy=975
  per-term layout, obs.critic=195 也保持不变, terrain 变化不影响 obs shape).
  → Phase 4 patched_load 走 "Phase 4 checkpoint (resume)" 路径
============================================================================
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from source.cmoe.cmoe.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from cmoe.tasks.velocity.cmoe import mdp
from cmoe.tasks.velocity.cmoe.terrains import CMOE_TERRAINS_CFG


# ============================================================
# Scene — Phase 4 scene + terrain generator + foot scanners
# ============================================================
@configclass
class Phase5_1SceneCfg(InteractiveSceneCfg):
    """Scene with 8 paper terrains + foot scanners for feet_edge reward."""

    # Paper §IV-A: 8 sub-terrains with curriculum
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=CMOE_TERRAINS_CFG,
        max_init_terrain_level=0,                      # 全部 robot 起始 level 0
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

    # ---- height_scanner: 用于 base_height reward (1×1 ray, Phase 2b 起就保留) ----
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

    # ---- elevation_scanner: paper §IV-A: 0.7m × 1.1m grid → 12×8=96 rays ----
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

    # ---- NEW: foot scanners for feet_edge reward (paper §IV-C) ----
    foot_height_scanner_left = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.2, 0.1]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.0,
    )
    foot_height_scanner_right = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.2, 0.1]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.0,
    )


# ============================================================
# Commands  (Phase 4 standard, 不引入 velocity curriculum)
# ============================================================
@configclass
class CommandsCfg:
    """Phase 4 同样的命令分布. 不引入 limit_ranges (留给 Phase 5.2)."""
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
# Actions — Phase 4 完全相同
# ============================================================
@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True,
    )


# ============================================================
# Observations — Phase 4 完全相同 (per-term layout 不变)
# ============================================================
@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            history_length=5, flatten_history_dim=True,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=5, flatten_history_dim=True,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=5, flatten_history_dim=True,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            history_length=5, flatten_history_dim=True,
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=5, flatten_history_dim=True,
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=5, flatten_history_dim=True,
        )
        last_action = ObsTerm(
            func=mdp.last_action,
            history_length=5, flatten_history_dim=True,
        )
        elevation = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("elevation_scanner"), "offset": 0.5},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            history_length=5, flatten_history_dim=True,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        last_action = ObsTerm(func=mdp.last_action)
        elevation = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("elevation_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ============================================================
# Events — Phase 4 完全相同 (Phase 5.2 才加 push / domain rand)
# ============================================================
@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )


# ============================================================
# Rewards — Phase 4 reward + feet_edge (paper §IV-C)
# ============================================================
@configclass
class PaperRewardsCfg:
    """Paper Table II + 3 EXTRA + feet_edge (Phase 5.1 NEW)."""

    # ---- Paper Table II rewards (与 Phase 4 完全相同) ----
    velocity_tracking = RewTerm(
        func=mdp.track_lin_vel_xy_exp_paper, weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    yaw_tracking = RewTerm(
        func=mdp.track_yaw_exp_paper, weight=2.0,
        params={"command_name": "base_velocity"},
    )
    z_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    rp_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    base_height = RewTerm(
        func=mdp.base_height_l2_paper, weight=-15.0,
        params={"target_height": 0.78, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble_paper, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link")},
    )
    collision = RewTerm(
        func=mdp.undesired_contacts, weight=-15.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*shoulder.*", ".*elbow.*", "torso_link", "pelvis", ".*hip.*", ".*knee.*"],
            ),
            "threshold": 0.1,
        },
    )
    feet_distance = RewTerm(
        func=mdp.feet_distance_paper, weight=0.8,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
            "d_min": 0.20, "d_max": 0.50,
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_paper, weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "command_name": "base_velocity",
            "target_air_time": 0.4, "min_command_norm": 0.1,
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

    # ---- 3 EXTRA rewards (用户保留) ----
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l2, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=[
                ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint", ".*_wrist_pitch_joint", ".*_wrist_yaw_joint",
            ],
        )},
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l2, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg(
            "robot", joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        )},
    )
    no_fly = RewTerm(
        func=mdp.desired_contacts, weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "threshold": 1.0,
        },
    )

    # ---- NEW (Phase 5.1, paper §IV-C): feet_edge gated to Hurdle/Gap ----
    feet_edge = RewTerm(
        func=mdp.feet_edge_gated, weight=-1.0,
        params={
            "sensor_cfg_left":  SceneEntityCfg("foot_height_scanner_left"),
            "sensor_cfg_right": SceneEntityCfg("foot_height_scanner_right"),
            "edge_height_threshold": 0.02,
            "gap_hurdle_names": ("Gap", "Hurdle"),
        },
    )


# ============================================================
# Terminations — Phase 4 同 (训练 1.0 rad, paper §V-A 也是 1°)
# ============================================================
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.2},   # 训练放宽到 1.2 rad (复杂地形不可学习)
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_link", "pelvis"]),
            "threshold": 1.0,
        },
    )
    base_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -3.0},
    )


# ============================================================
# Curriculum — paper §IV-A 的 Rudin [37] 标准
# ============================================================
@configclass
class CurriculumCfg:
    """Paper §IV-A 提到的 curriculum learning.
    
    Phase 5.1 仅启用 terrain_levels_vel (Rudin [37] 标准).
    velocity command curriculum 留给 Phase 5.2.
    """
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


# ============================================================
# Env cfg
# ============================================================
@configclass
class CmoePhase5_1EnvCfg(ManagerBasedRLEnvCfg):
    """Phase 5.1 = Phase 4 architecture + 8 paper terrains + curriculum."""

    scene: Phase5_1SceneCfg          = Phase5_1SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg    = ObservationsCfg()
    actions: ActionsCfg              = ActionsCfg()
    commands: CommandsCfg            = CommandsCfg()
    rewards: PaperRewardsCfg         = PaperRewardsCfg()
    terminations: TerminationsCfg    = TerminationsCfg()
    events: EventCfg                 = EventCfg()
    curriculum: CurriculumCfg        = CurriculumCfg()

    def __post_init__(self):
        self.decimation       = 4
        self.episode_length_s = 20.0   # paper §V-A "for 20 seconds"
        self.sim.dt                              = 0.005
        self.sim.render_interval                 = self.decimation
        self.sim.physics_material                = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 15
        self.scene.contact_forces.update_period  = self.sim.dt
        self.scene.foot_height_scanner_left.update_period  = self.decimation * self.sim.dt
        self.scene.foot_height_scanner_right.update_period = self.decimation * self.sim.dt


@configclass
class CmoePhase5_1EnvCfg_PLAY(CmoePhase5_1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs   = 32
        self.episode_length_s = 20.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        # Play 模式下不要 curriculum (使用全部地形 levels)
        self.scene.terrain.max_init_terrain_level = None
