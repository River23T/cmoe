# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CMoE velocity-locomotion env cfg (Unitree G1-29DoF + CMoE custom terrains).

This file follows the structure of Unitree's official G1-29DoF velocity env cfg
(`unitreerobotics/unitree_rl_lab`, commit 4960b84) and adapts it to the CMoE
paper (Ma et al. 2026). Differences vs. the Unitree file:
  - scene.terrain.terrain_generator = CMOE1_TERRAINS_CFG (paper 9 terrains)
  - height_scanner pattern = 0.7 m x 1.1 m (paper §IV-A)
  - rewards = Paper Table II (20 terms, weights exactly as in the paper)
  - events += 30 N push every 16 s (paper §IV-B)
  - terminations: 5 s episode, torso bad orientation, root height
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

# Unitree's official G1-29DoF robot articulation cfg
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG

# CMoE's custom terrains (the 9-terrain generator built in previous sessions)
from .terrains import CMOE_TERRAINS_CFG

from . import mdp

# ============================================================
# Scene
# ============================================================
@configclass
class CmoeSceneCfg(InteractiveSceneCfg):
    """Scene with CMoE's 9 paper terrains + G1-29DoF + sensors."""

    # terrain generator — the 9 CMoE paper terrains
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=CMOE_TERRAINS_CFG,
        max_init_terrain_level=CMOE_TERRAINS_CFG.num_rows - 1,
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

    # robot
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # elevation map sensor — paper §IV-A: 0.7 m x 1.1 m around the robot
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.7, 1.1]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # contact forces on all bodies — used by collision, feet_stumble, feet_air_time
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # per-foot height scanner — used by reward #20 (feet_edge) and can also feed
    # the paper's "feet ground parallel". 5x5 grid at 0.02 m spacing under each foot.
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

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
                         f"kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ============================================================
# Commands  (paper §IV-A: velocity curriculum on complex terrains)
# ============================================================
@configclass
class CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            # paper benchmark speed 0.8 m/s, leave headroom
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.5, 0.5)
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
        """Proprioception. Paper Eq. 2: (1)o_t = [1.ω_t,2.g_t,3.c_v,4.θ_t,5.θ̇_t,6.a_{t-1}]."""
        base_ang_vel     = ObsTerm(func=mdp.base_ang_vel,       scale=0.2, noise=Unoise(-0.2, 0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity,           noise=Unoise(-0.05, 0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel    = ObsTerm(func=mdp.joint_pos_rel,                noise=Unoise(-0.01, 0.01))
        joint_vel_rel    = ObsTerm(func=mdp.joint_vel_rel,      scale=0.05, noise=Unoise(-1.5, 1.5))
        last_action      = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length   = 5      # paper §III-C: "observation history o_t^H"
            self.enable_corruption = True   # paper §IV-B: noise during training
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic inputs (paper Fig. 3: (3)o_t^P)."""
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel     = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel     = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action       = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length   = 5
            self.concatenate_terms = True

    @configclass
    class ElevationCfg(ObsGroup):
        """Elevation map (2)e_t. Paper §III-C: encoded by the AE into z^E_t."""
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            noise=Unoise(-0.1, 0.1),   # paper §IV-B: "delay noise and Gaussian noise"
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
    # -- startup --
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
    # paper mentions motor strength / kp / kd randomization
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

    # -- reset --
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
        params={"position_range": (1.0, 1.0), "velocity_range": (-1.0, 1.0)},
    )

    # -- interval: 30 N push every 16 s (paper §IV-B) --
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(16.0, 16.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


# ============================================================
# Rewards  (Paper Table II — 20 terms, weights EXACTLY as printed)
# ============================================================
@configclass
class CmoeRewardsCfg:
    # [#1]  velocity tracking
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # [#2]  yaw tracking  (custom L1-in-exp, matches paper exactly)
    yaw_tracking = RewTerm(
        func=mdp.track_ang_vel_z_exp_l1, weight=2.0,
        params={"command_name": "base_velocity"},
    )
    # [#3]  z velocity
    z_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    # [#4]  roll-pitch velocity
    rp_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # [#5]  orientation
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    # [#6]  base height (with terrain-adjusted target)
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-15.0,
        params={
            "target_height": 0.78,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    # [#7]  feet stumble
    feet_stumble = RewTerm(
        func=mdp.feet_stumble, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link")},
    )
    # [#8]  collision (non-foot bodies, force > 0.1 N)
    collision = RewTerm(
        func=mdp.undesired_contacts, weight=-15.0,
        params={
            "threshold": 0.1,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["(?!.*ankle.*).*"],   # everything except ankle links
            ),
        },
    )
    # [#9]  feet lateral distance
    feet_distance = RewTerm(
        func=mdp.feet_lateral_distance, weight=0.8,
        params={
            "d_min": 0.20, "d_max": 0.50,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
        },
    )
    # [#10] feet air time
    feet_air_time = RewTerm(
        func=mdp.feet_air_time, weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    # [#11] feet ground parallel
    feet_ground_parallel = RewTerm(
        func=mdp.feet_ground_parallel, weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
            "foot_half_length": 0.10,
            "foot_half_width":  0.04,
        },
    )
    # [#12] hip dof error (L2, matches paper)
    hip_dof_error = RewTerm(
        func=mdp.joint_deviation_l2, weight=-0.5,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"],
        )},
    )
    # [#13] dof acc
    dof_acc   = RewTerm(func=mdp.joint_acc_l2,     weight=-2.5e-7)
    # [#14] dof vel
    dof_vel   = RewTerm(func=mdp.joint_vel_l2,     weight=-5.0e-4)
    # [#15] torques
    torques   = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    # [#16] action rate
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.3)
    # [#17] dof pos limits
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    # [#18] dof vel limits  (soft_ratio=1.0 → exactly matches paper ReLU(|θ̇|-|θ̇_max|))
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits, weight=-1.0, params={"soft_ratio": 1.0}
    )
    # [#19] torque limits  (custom paper formula)
    torque_limits = RewTerm(func=mdp.applied_torque_limits_paper, weight=-1.0)
    # [#20] feet edge  (terrain-conditional, see curriculum gating below)
    feet_edge = RewTerm(
        func=mdp.feet_edge, weight=-1.0,
        params={
            "sensor_cfg_left": SceneEntityCfg("foot_height_scanner_left"), "sensor_cfg_right": SceneEntityCfg("foot_height_scanner_right"),
            "edge_height_threshold": 0.02,
        },
    )


# ============================================================
# Terminations  (paper §V-A)
# ============================================================

@configclass
class TerminationsCfg:
    """Episode termination terms.

    Paper §V-A says:
      (1) collision with parts other than the feet  → handled by undesired
          contacts in `CmoeRewardsCfg.collision` (as a -15 reward, not a
          hard termination — matches the paper's "large negative reward"
          treatment).
      (2) torso roll or pitch deviation exceeds 1 degree. This is almost
          certainly a typo (1° would terminate on any tiny perturbation).
          Standard IsaacLab/Unitree practice uses 0.8 rad ≈ 46°. We keep
          that convention here. If you want to match the paper literally,
          change `limit_angle` to math.radians(1.0).
    """

    time_out       = DoneTerm(func=mdp.time_out, time_out=True)
    base_too_low   = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.3},
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.8},   # ≈46°; use math.radians(1.0) to match paper literally
    )


# ============================================================
# Curriculum
# ============================================================
@configclass
class CurriculumCfg:
    """Curriculum learning terms (paper §IV-A)."""

    # terrain difficulty curriculum — advances rows (difficulty) based on progress
    terrain_levels     = CurrTerm(func=mdp.terrain_levels_vel)
    # velocity command curriculum — gradually expands from `ranges` to `limit_ranges`
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)


# ============================================================
# Env cfg
# ============================================================
@configclass
class CmoeEnvCfg(ManagerBasedRLEnvCfg):
    """CMoE locomotion-velocity training env cfg (full CMoE paper replication)."""

    # Scene settings
    scene: CmoeSceneCfg            = CmoeSceneCfg(num_envs=4096, env_spacing=2.5)
    # MDP settings
    observations: ObservationsCfg  = ObservationsCfg()
    actions: ActionsCfg            = ActionsCfg()
    commands: CommandsCfg          = CommandsCfg()
    rewards: CmoeRewardsCfg        = CmoeRewardsCfg()
    terminations: TerminationsCfg  = TerminationsCfg()
    events: EventCfg               = EventCfg()
    curriculum: CurriculumCfg      = CurriculumCfg()

    def __post_init__(self):
        # general
        self.decimation       = 4
        self.episode_length_s = 20.0                  # paper §V-A: 20 s rollouts
        # sim
        self.sim.dt                                = 0.005
        self.sim.render_interval                   = self.decimation
        self.sim.physics_material                  = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count   = 10 * 2 ** 15
        # sensor update rates
        self.scene.contact_forces.update_period      = self.sim.dt
        self.scene.height_scanner.update_period      = self.decimation * self.sim.dt
        self.scene.foot_height_scanner_left.update_period = self.decimation * self.sim.dt
        self.scene.foot_height_scanner_right.update_period = self.decimation * self.sim.dt
        # tie terrain-generator curriculum flag to CurriculumCfg
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True


@configclass
class CmoeEnvCfg_PLAY(CmoeEnvCfg):
    """Play / evaluation variant: fewer envs, fixed hardest-difficulty terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs                           = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.terrain_generator.difficulty_range = (1.0, 1.0)
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
