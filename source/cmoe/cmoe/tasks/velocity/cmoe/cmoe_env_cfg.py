# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CMoE velocity-locomotion env cfg (R5 — tracking-quality focus).

=============================================================
R5 vs R4: focus on closing the gap to paper's Table III
=============================================================

iter-12600 log shows R4 succeeded in the bootstrap phase:
  * Mean reward: ~20           (positive, stable)
  * Mean episode length: ~780  (78% of 20s)
  * terrain_levels = 4.67       (on scale 0..9, close to paper Fig. 4)
  * lin_vel_cmd_levels = 1.5    (at R4 limit_ranges maximum!)
  * bad_orientation = 0%       (solved)

But tracking and stability are NOT paper-grade yet:
  * error_vel_xy = 1.0 m/s     (terrible — paper expects <0.3)
  * track_lin_vel_xy reward = 0.65 of max 2.0  (30% of potential)
  * base_contact = 44%          (44% of eps end with falls, vs paper's
                                 ~10-25% implied by Table III)

ROOT CAUSES found in R4 log:

  (1) `limit_ranges.lin_vel_x = (-1.0, 1.5)` is WRONG. Paper §V-A
      evaluates at 0.8 m/s. Our 1.5 m/s ceiling means the curriculum
      pushed commands past what the robot can track on level-4
      stairs/gaps. This creates a self-inflicted tracking deficit.

  (2) `feet_edge` activates on ALL terrains. Paper §IV-C:
      "the foot edge reward is activated ONLY when the robot is
      trained in a hurdle or gap environment. This is because
      touching the edge of a step or irregular terrain should
      NOT be penalized."

      Current log shows feet_edge ≈ -0.08 per episode constantly —
      i.e. robots on stair/slope are being punished for standing
      naturally on step surfaces.

  (3) `terrain_levels_vel` promote distance 1.5 m is slightly low.
      Now that command curriculum is 1.5 m/s × 20 s = 30 m of
      commanded motion, promoting on 1.5 m real distance means
      envs that only achieved 5% of commanded motion still go up.
      Paper's [37] Rudin default is size/2 = 4 m. R5 uses 2.5 m
      (balanced — above the noise floor but not so strict that
      learning stalls).

R5 changes (strictly toward paper compliance):

  A. `limit_ranges.lin_vel_x = (-1.0, 1.5) → (-1.0, 1.0)`
     Matches the paper §V-A evaluation speed (0.8 m/s) with a small
     safety margin above. lin_vel_cmd_levels will stop growing at 1.0
     instead of 1.5, and the existing trained policy is compatible
     with a REDUCED limit (just caps growth).

  B. `feet_edge` → `feet_edge_gated` (Paper §IV-C EXACT).
     Swap to the new ``mdp.feet_edge_gated`` function that masks
     the penalty to Gap + Hurdle envs only.

  C. `terrain_levels_vel` promote threshold 1.5 m → 2.5 m
     (curriculums.py R5). Demote 0.2 → 0.3 (envs stuck should
     demote faster so they don't stay at too-hard levels).

Everything else matches paper §III-IV exactly.
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
    """Scene with CMoE's paper terrains (+FlatSlope bootstrap) + G1-29DoF + sensors."""

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
# Commands  (R5: paper §V-A compatible limit 0.8 m/s target)
# ============================================================
@configclass
class CommandsCfg:
    """Velocity commands.

    R5: `limit_ranges.lin_vel_x` lowered from (-1.0, 1.5) to (-1.0, 1.0).
    Paper §V-A evaluates at 0.8 m/s. Our R5 max = 1.0 m/s gives 25%
    headroom above eval speed. 1.5 was forcing the curriculum past the
    robot's real tracking capability on level-4+ terrains.
    """
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        # R9: rel_standing_envs 0.02 → 0.0. No envs are forced to stand.
        # Otherwise 2% of envs get zero command and the policy learns that
        # "standing" is a valid behavior.
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            # R9: initial range widened (-0.5, 0.5) → (-0.8, 0.8).
            # With (-0.5, 0.5), a stationary robot has err ≈ 0.25 m/s
            # → exp(-0.0625/0.09) = 0.50 per-step tracking reward, giving
            # the policy a strong free-standing local optimum.
            # With (-0.8, 0.8), stationary err ≈ 0.4 m/s → exp(-0.16/0.09)
            # = 0.17, much worse.  Combined with `std: 0.5 → 0.3` below and
            # `rel_standing_envs=0.0`, "stand still" stops being optimal.
            lin_vel_x=(-0.8, 0.8),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.5, 0.5),
            heading=(-3.14, 3.14),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            # R5: 1.5 → 1.0 (paper §V-A evaluates at 0.8 m/s)
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
        """Elevation map e_t, with paper §IV-B salt-pepper + chamfer."""
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
    # R8: joint reset randomization disabled (was (0.5, 1.5)).
    #
    # The previous R7 value `position_range=(0.5, 1.5)` multiplies each
    # joint's default position by a random factor in [0.5, 1.5].  For
    # joints with nonzero defaults (notably the G1's knees at ~0.3 rad),
    # this produces reset configurations where the robot is already in a
    # half-crouched, nearly unstable posture — leading to an 88 % base_contact
    # termination rate in iter 600 of the previous run.
    #
    # Paper §IV-B does NOT specify any joint reset randomization, and the
    # reference `humanoid_locomotion/ame_1/stage1` project omits joint reset
    # randomization entirely.  R8 returns to default-pose reset (factor 1.0),
    # which matches both paper and reference.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),   # R8: no randomization
            "velocity_range": (0.0, 0.0),
        },
    )

    # R8: push_robot interval initially longer, narrower velocity range,
    # expanded by the `push_curriculum` term in CurriculumCfg as terrain
    # level rises.  Paper §IV-B specifies 30 N every 16 s; we ramp to
    # those values by terrain_level >= 1.0 (see curriculums.push_curriculum).
    #
    # Reference `humanoid_locomotion/ame_1/stage1` sets push_robot = None
    # entirely during bootstrap.  R8 uses a gentler "always-on but ramping"
    # approach so the policy sees disturbance training throughout but is
    # never overwhelmed in bootstrap.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(30.0, 30.0),      # R8: 16 → 30 during bootstrap
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},  # R8: ±0.5 → ±0.2 during bootstrap
    )


# ============================================================
# Rewards  (Paper Table II — 20 terms, R5 PAPER STRICT)
# ============================================================
@configclass
class CmoeRewardsCfg:
    # [#1] velocity tracking — PAPER TABLE II EXACT (weight=2.0)
    #
    # FRAME FIX: rewards.py track_lin_vel_xy_exp now projects world velocity
    # into the YAW frame (matching IsaacLab official G1/H1 implementations).
    # This was the root cause of the "stand-still local optimum" — body-frame
    # velocity tracking gave misleading gradient signals on slopes/stairs
    # because body-frame v_x is corrupted by roll/pitch tilt.
    #
    # Paper: exp(-||v_xy - v^c_xy||² / σ),  weight=2.0, σ unspecified.
    # IsaacLab convention: σ=0.5 (i.e. σ²=0.25) — confirmed in G1/H1/Digit.
    velocity_tracking = RewTerm(
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
    # [#6] base height (PAPER EXACT -15.0)
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
    # [#20] feet edge — R5: use TERRAIN-GATED version (Paper §IV-C EXACT).
    #                    Only active on Gap and Hurdle sub-terrains.
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

    Paper §V-A gives two conditions: (1) collision with parts other than
    feet, (2) torso roll/pitch exceeds 1°. We keep bad_orientation=1.2
    rad for TRAINING (strict 1° is unlearnable); the paper's 1° is the
    EVALUATION condition.
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
# Curriculum
# ============================================================
@configclass
class CurriculumCfg:
    """Curriculum learning (paper §IV-A, R5 threshold tuning)."""
    terrain_levels     = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
    # R8: ramp push disturbance from (0.2 m/s, 30 s) back to paper-spec
    # (0.5 m/s, 16 s) as terrain_level rises above 1.0. See
    # curriculums.push_curriculum for the ramp schedule.
    push_curriculum    = CurrTerm(func=mdp.push_curriculum)
    # [DISABLED] tracking_weight_curriculum was a hack to ramp from 4.0→2.0.
    # Now that the body-frame bug is fixed (yaw frame), weight stays at the
    # paper-exact value of 2.0 the whole training. Curriculum no longer needed.
    # tracking_weight_curriculum = CurrTerm(func=mdp.tracking_weight_curriculum)


# ============================================================
# Env cfg
# ============================================================
@configclass
class CmoeEnvCfg(ManagerBasedRLEnvCfg):
    """CMoE locomotion-velocity training env cfg (R5 tracking-quality)."""

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
        self.episode_length_s = 20.0
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
    """Play/evaluation variant: fewer envs, hardest difficulty."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs                           = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.terrain_generator.difficulty_range = (1.0, 1.0)
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
