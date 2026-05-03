# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CMoE velocity-locomotion env cfg — PAPER-STRICT (R-FINAL).

============================================================================
设计原则: 严格按 paper Table II + §III + §IV 实现.
============================================================================

R-FINAL 移除的所有非论文偏离:
  - REMOVED no_fly reward (paper Table II 没有此项)
  - REMOVED joint_deviation_arms (paper Table II 没有此项)
  - REMOVED joint_deviation_waist (paper Table II 没有此项)
  - REMOVED tracking_weight_curriculum (paper Table II weight=2.0 固定)
  - REMOVED push_curriculum (paper §IV-B 30 N every 16 s 固定)
  - REMOVED feet_edge_gated 用论文原始 feet_edge (但保留 hurdle/gap 限制)
  - REVERT track_lin_vel_xy weight 4.0 → 2.0 (paper Table II EXACT)
  - REVERT initial command range (-0.8, 0.8) → 论文未规定, 用标准 (-1.0, 1.0)
  - REVERT rel_standing_envs 0 → 0.02 (Isaac Lab 标准)
  - REVERT push_robot 30 → 16 s, 0.2 → 0.5 m/s (paper §IV-B EXACT)

保留 (paper Fig 3 + §III-D 要求):
  - R15 actor obs systematic obs [v_pred, z_H, z_E, o_c, e_t]
  - 5 experts MoE
  - VAE β=0.001, AE 32-dim latent
  - SwAV K=32, τ=0.2

工程必要修复 (不影响算法):
  - inject_cmoe_runner_patches (rsl_rl 不会自动保存 MoE 权重)
  - bad_orientation 1.2 rad 训练 / 1.0° 评估 (paper §V-A)
  - episode_length_s = 20.0 (paper §V-A "for 20 seconds")
============================================================================
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from source.cmoe.cmoe.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG

from .terrains import CMOE_TERRAINS_CFG

from . import mdp


# ============================================================
# Scene
# ============================================================
@configclass
class CmoeSceneCfg(InteractiveSceneCfg):
    """Scene with CMoE's 8 paper terrains + G1-29DoF + sensors."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=CMOE_TERRAINS_CFG,
        max_init_terrain_level=0,
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

    # elevation map sensor — paper §IV-A: 0.7 m × 1.1 m around the robot
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.7, 1.1]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    foot_height_scanner_left = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=[0.20, 0.10]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    foot_height_scanner_right = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=[0.20, 0.10]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
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
# Commands  (paper §V-A: 0.8 m/s evaluation)
# ============================================================
@configclass
class CommandsCfg:
    """Velocity commands.

    Paper §V-A evaluates at 0.8 m/s. Initial range = limit range (Isaac Lab
    标准 — 论文没有规定 bootstrap 阶段缩小命令).
    """
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,  # Isaac Lab 标准
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            # 初始训练范围 (curriculum 会逐步扩大到 limit)
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.5, 0.5),
            heading=(-3.14, 3.14),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            # paper §V-A 评估在 0.8 m/s, 留 25% 余量
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
    )


# ============================================================
# Actions
# ============================================================
@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


# ============================================================
# Observations  (paper Eq. 2 proprioception + elevation map)
# ============================================================
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioception. Paper Eq. 2: o_t = [ω_t, g_t, c_v, θ_t, θ̇_t, a_{t-1}]."""
        base_ang_vel     = ObsTerm(func=mdp.base_ang_vel,       scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity,           noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel    = ObsTerm(func=mdp.joint_pos_rel,                noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel    = ObsTerm(func=mdp.joint_vel_rel,      scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action      = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length   = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic inputs (paper Fig. 3: o_t^P)."""
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel     = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel     = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action       = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length   = 0
            self.concatenate_terms = True

    @configclass
    class ElevationCfg(ObsGroup):
        """Elevation map e_t, paper §IV-B salt-pepper + chamfer DR."""
        height_scan = ObsTerm(
            func=mdp.height_scan_sp,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "sp_prob": 0.02,
                "chamfer_enable": True,
            },
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg    = PolicyCfg()
    critic: CriticCfg    = CriticCfg()
    elevation: ElevationCfg = ElevationCfg()


# ============================================================
# Events  (paper §IV-B Domain Randomization)
# ============================================================
@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.25),
            "dynamic_friction_range": (0.3, 1.25),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params":   (0.8, 1.2),
            "operation": "scale",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        },
    )
    # 关节重置无随机化 (paper §IV-B 没有规定)
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # paper §IV-B EXACT: "applied a perturbation of up to 30 N to the robot every 16 seconds"
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(16.0, 16.0),  # paper EXACT
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},  # ~30 N for G1
    )


# ============================================================
# Rewards  (Paper Table II — 严格 20 项)
# ============================================================
@configclass
class CmoeRewardsCfg:
    """Paper Table II - 严格 20 个 reward terms.

    所有 weight 严格按 paper Table II:
      velocity tracking: 2.0
      yaw tracking: 2.0
      z velocity: -1.0
      roll-pitch velocity: -0.05
      orientation: -2.0
      base height: -15.0  (注: paper 笔误? 实际 IsaacLab 用 -10.0; paper 文本是 -15.0)
      feet stumble: -1.0
      collision: -15.0
      feet distance: 0.8
      feet air time: 1.0
      feet ground parallel: -0.02
      hip dof error: -0.5
      dof acc: -2.5e-7
      dof vel: -5.0e-4
      torques: -1.0e-5
      action rate: -0.3
      dof pos limits: -2.0
      dof vel limits: -1.0
      torque limits: -1.0
      feet edge: -1.0  (paper §IV-C: gated to Hurdle/Gap only)
    """
    # [#1] velocity tracking — Paper Table II EXACT (weight=2.0)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # [#2] yaw tracking (PAPER EXACT 2.0)
    yaw_tracking = RewTerm(
        func=mdp.track_ang_vel_z_exp_l1, weight=2.0,
        params={"command_name": "base_velocity"},
    )
    # [#3] z velocity (PAPER -1.0)
    z_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    # [#4] roll-pitch velocity (PAPER -0.05)
    rp_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # [#5] orientation (PAPER -2.0)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    # [#6] base height (PAPER -15.0)
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-15.0,
        params={
            "target_height": 0.78,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    # [#7] feet stumble (PAPER -1.0)
    feet_stumble = RewTerm(
        func=mdp.feet_stumble, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link")},
    )
    # [#8] collision (PAPER -15.0)
    collision = RewTerm(
        func=mdp.undesired_contacts, weight=-15.0,
        params={
            "threshold": 0.1,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["(?!.*ankle.*).*"],
            ),
        },
    )
    # [#9] feet lateral distance (PAPER 0.8)
    feet_distance = RewTerm(
        func=mdp.feet_lateral_distance, weight=0.8,
        params={
            "d_min": 0.20, "d_max": 0.50,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
        },
    )
    # [#10] feet air time (PAPER 1.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time, weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    # [#11] feet ground parallel (PAPER -0.02)
    feet_ground_parallel = RewTerm(
        func=mdp.feet_ground_parallel, weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
            "foot_half_length": 0.10,
            "foot_half_width":  0.04,
        },
    )
    # [#12] hip dof error (PAPER -0.5)
    hip_dof_error = RewTerm(
        func=mdp.joint_deviation_l2, weight=-0.5,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"],
        )},
    )
    # [#13] dof acc (PAPER -2.5e-7)
    dof_acc   = RewTerm(func=mdp.joint_acc_l2,     weight=-2.5e-7)
    # [#14] dof vel (PAPER -5e-4)
    dof_vel   = RewTerm(func=mdp.joint_vel_l2,     weight=-5.0e-4)
    # [#15] torques (PAPER -1e-5)
    torques   = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    # [#16] action rate (PAPER -0.3)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.3)
    # [#17] dof pos limits (PAPER -2.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    # [#18] dof vel limits (PAPER -1.0)
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits, weight=-1.0, params={"soft_ratio": 1.0}
    )
    # [#19] torque limits (PAPER -1.0)
    torque_limits = RewTerm(func=mdp.applied_torque_limits_paper, weight=-1.0)
    # [#20] feet edge — PAPER §IV-C: gated to Hurdle/Gap envs only
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
# Terminations  (paper §V-A)
# ============================================================
@configclass
class TerminationsCfg:
    """Episode termination.

    Paper §V-A: (1) collision with parts other than feet,
              (2) torso roll/pitch exceeds 1°.

    训练用 1.2 rad 宽松 termination (1° 训练阶段太严苛, 不可学习);
    评估时用 paper §V-A 严格 1° (在 eval_env_cfg 中).
    """
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -3.0},
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.2},
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["torso_link", "pelvis"],
            ),
            "threshold": 1.0,
        },
    )


# ============================================================
# Curriculum (paper §IV-A)
# ============================================================
@configclass
class CurriculumCfg:
    """Paper §IV-A 提到的 curriculum:
       (1) terrain-difficulty curriculum (Rudin [37] 标准)
       (2) velocity command curriculum 'on complex terrains'
    """
    terrain_levels     = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)


# ============================================================
# Env cfg
# ============================================================
@configclass
class CmoeEnvCfg(ManagerBasedRLEnvCfg):
    """CMoE locomotion-velocity training env cfg (PAPER-STRICT R-FINAL)."""

    scene: CmoeSceneCfg            = CmoeSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg  = ObservationsCfg()
    actions: ActionsCfg            = ActionsCfg()
    commands: CommandsCfg          = CommandsCfg()
    rewards: CmoeRewardsCfg        = CmoeRewardsCfg()
    terminations: TerminationsCfg  = TerminationsCfg()
    events: EventCfg               = EventCfg()
    curriculum: CurriculumCfg      = CurriculumCfg()

    def __post_init__(self):
        self.decimation       = 4
        self.episode_length_s = 20.0   # paper §V-A "for 20 seconds"
        self.sim.dt                                = 0.005
        self.sim.render_interval                   = self.decimation
        self.sim.physics_material                  = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count   = 10 * 2 ** 15
        self.scene.contact_forces.update_period      = self.sim.dt
        self.scene.height_scanner.update_period      = self.decimation * self.sim.dt
        self.scene.foot_height_scanner_left.update_period  = self.decimation * self.sim.dt
        self.scene.foot_height_scanner_right.update_period = self.decimation * self.sim.dt
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True


@configclass
class CmoeEnvCfg_PLAY(CmoeEnvCfg):
    """Play/evaluation variant: fewer envs, hardest difficulty.

    评估时强制 yaw=0, 防止 robot 旋转后绕过障碍 (mix2 单木桥).
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs                           = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.terrain_generator.difficulty_range = (1.0, 1.0)
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

        # 评估强制前向 yaw, 避免 mix2 绕路
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "yaw": (0.0, 0.0),
        }
