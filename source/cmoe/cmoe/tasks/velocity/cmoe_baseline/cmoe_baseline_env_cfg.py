# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 1.5 v7 — v6 + anti-bunny-hop reward (humanoid_locomotion's no_fly).

================================================================
WHY v6 → v7
================================================================
v6 successfully fixed the v5 stand-still trap:
    success=1.000  avg_dist=15.24 m  (95% of theoretical max!)
    Episode length = 500/500, robot walks fast and far.

But the gait QUALITY is wrong: v6 evaluation screenshots show
the robot **bunny-hopping** (both feet leaving and landing the
ground SIMULTANEOUSLY) instead of human-like alternating walking.

ROOT CAUSE — Paper Table II has NO anti-airborne term
================================================================
Reward analysis at cmd=0.8 m/s for bunny-hop vs proper walking:

  velocity_tracking (paper +2.0):     same +1.92/s (both reach cmd)
  feet_air_time (paper +1.0, t=0.4s): only -0.16/s favors walking
  z_velocity (paper -1.0):            -1.13/s for hop, -0.04/s for
                                       walk → favors walk by 1.09/s
  All other terms:                    ~equal in both gaits

  Net per-step balance:
    bunny-hop: +1.92 - 1.13 - 0.33 ≈ +0.46
    walk:      +1.92 - 0.04 - 0.17 ≈ +1.71

  Walking IS preferred by paper Table II by ~1.25/s, BUT bunny-hop
  is still positive (+0.46/s), making it a viable LOCAL OPTIMUM
  that PPO can converge to from random init.

  This is sim-dependent: in IsaacGym (paper) the joint dynamics
  apparently make synchronized leg push impossible/unrewarding;
  in IsaacLab the physics let PPO discover the bunny-hop attractor.

================================================================
THE FIX: Add `no_fly` reward (NEW EXTRA-3 term)
================================================================
The humanoid_locomotion reference project (validated walking on
IsaacLab + Unitree H1) has THIS EXACT mechanism to prevent
bunny-hopping:

    no_fly = RewTerm(
        func=mdp.desired_contacts,                # IsaacLab built-in
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces",
                                          body_names=".*ankle_link.*"),
            "threshold": 1.0,
        },
    )

What it does:
  `desired_contacts` returns -1 ONLY when ALL listed contact bodies
  have force < threshold simultaneously, i.e., when BOTH feet are
  airborne. With weight -5.0:

    Both feet on ground:        reward = 0  (normal stance)
    One foot airborne (swing):  reward = 0  (normal walking!)
    Both feet airborne (hop):   reward = -5 (PUNISH)

  This is a SURGICAL fix for bunny-hop without disturbing any other
  gait pattern. Walking has alternating contacts → never triggers.
  Bunny-hopping has both-airborne phases → strongly punished.

  Expected per-second penalty: bunny-hop has both-airborne ~50% of
  cycle ≈ -2.5/s. This dominates the +0.46/s bunny-hop bonus and
  swings the optimum decisively to walking.

================================================================
v7 CHANGES vs v6 — ONE NEW TERM
================================================================
ADDED:
  EXTRA-3: no_fly  weight=-5.0  (anti-bunny-hop, humanoid_locomotion validated)

UNCHANGED from v6:
  • All 19/20 paper Table II weights stay paper-exact.
  • action_rate sim-to-sim fix (-5e-3) stays from v6.
  • EXTRA-1 joint_deviation_arms  -1.0 stays.
  • EXTRA-2 joint_deviation_waist -1.0 stays.
  • feet_edge omitted on flat (paper §IV-C compliant).

================================================================
PAPER REPRODUCTION NOTE
================================================================
This file is the Phase 1 baseline (port-internal sanity check).
The paper has no Phase 1 stage; paper figures come from the FULL
CMoE training task (`CMoE-G1-Velocity-v0` / cmoe/cmoe_env_cfg.py).
The full training will get the SAME no_fly addition for
consistency, since bunny-hop is sim-induced and applies regardless
of terrain.

================================================================
FINAL TABLE: 21 paper terms (20 Table II + feet_edge omitted on flat)
            + 3 EXTRA IsaacLab port additions
================================================================
  #   Term                Weight       Status
  --  ------------------  -----------  ------------------------
  1   velocity tracking   +2.0         paper-exact
  2   yaw tracking        +2.0         paper-exact
  3   z velocity          -1.0         paper-exact
  4   roll-pitch velocity -0.05        paper-exact
  5   orientation         -2.0         paper-exact
  6   base height         -15.0        paper-exact
  7   feet stumble        -1.0         paper-exact
  8   collision           -15.0        paper-exact
  9   feet distance       +0.8         paper-exact
  10  feet air time       +1.0         paper-exact
  11  feet ground //      -0.02        paper-exact
  12  hip dof error       -0.5         paper-exact
  13  dof acc             -2.5e-7      paper-exact
  14  dof vel             -5e-4        paper-exact
  15  torques             -1e-5        paper-exact
  16  action rate         -5e-3        v6 sim-to-sim (paper -0.3)
  17  dof pos limits      -2.0         paper-exact
  18  dof vel limits      -1.0         paper-exact
  19  torque limits       -1.0         paper-exact
  20  feet edge           N/A          paper §IV-C: only Hurdle/Gap

  E1  joint_dev_arms      -1.0         v5 EXTRA: arm pose stabilizer
  E2  joint_dev_waist     -1.0         v5 EXTRA: waist pose stabilizer
  E3  no_fly              -5.0         v7 EXTRA: anti-bunny-hop (NEW)

  Final tally: 19/20 paper-exact + 1 sim-to-sim (action_rate)
               + 3 EXTRA labeled clearly
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
# Scene
# ============================================================
@configclass
class BaselineSceneCfg(InteractiveSceneCfg):
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
# Commands
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
# Actions
# ============================================================
@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


# ============================================================
# Observations (proprio only, 99-dim)
# ============================================================
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,      noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel,      scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel     = ObsTerm(func=mdp.joint_pos_rel,     noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel     = ObsTerm(func=mdp.joint_vel_rel,     scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action       = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length    = 0
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
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

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ============================================================
# Events
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
# Rewards: 20 paper Table II terms + 2 EXTRA pose stabilizers
# ============================================================
@configclass
class PaperRewardsCfg:
    # =================================================================
    # 20 PAPER TABLE II TERMS — PAPER-EXACT WEIGHTS
    # =================================================================

    # ===== [#1] velocity tracking — PAPER EXACT +2.0 =====
    velocity_tracking = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # ===== [#2] yaw tracking — PAPER EXACT +2.0 =====
    yaw_tracking = RewTerm(
        func=mdp.track_ang_vel_z_exp_l1, weight=2.0,
        params={"command_name": "base_velocity"},
    )
    # ===== [#3] z velocity — PAPER EXACT -1.0 =====
    z_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    # ===== [#4] roll-pitch velocity — PAPER EXACT -0.05 =====
    rp_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # ===== [#5] orientation — PAPER EXACT -2.0 =====
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    # ===== [#6] base height — PAPER EXACT -15.0 =====
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-15.0,
        params={
            "target_height": 0.78,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    # ===== [#7] feet stumble — PAPER EXACT -1.0 =====
    feet_stumble = RewTerm(
        func=mdp.feet_stumble, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link")},
    )
    # ===== [#8] collision — PAPER EXACT -15.0 PER-STEP =====
    # Restored to paper's per-step penalty (was -200 termination in v4).
    # Excludes ankle/foot links (those *should* contact).
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
    # ===== [#9] feet distance — PAPER EXACT +0.8 =====
    feet_distance = RewTerm(
        func=mdp.feet_lateral_distance, weight=0.8,
        params={
            "d_min": 0.20, "d_max": 0.50,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
        },
    )
    # ===== [#10] feet air time — PAPER EXACT +1.0 =====
    feet_air_time = RewTerm(
        func=mdp.feet_air_time, weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    # ===== [#11] feet ground parallel — PAPER EXACT -0.02 =====
    feet_ground_parallel = RewTerm(
        func=mdp.feet_ground_parallel, weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
            "foot_half_length": 0.10,
            "foot_half_width":  0.04,
        },
    )
    # ===== [#12] hip dof error — PAPER EXACT -0.5 =====
    # Per paper: "hip joints" = hip yaw + hip roll (NOT pitch — pitch
    # is the leg-swing DoF; locking it prevents stepping).
    hip_dof_error = RewTerm(
        func=mdp.joint_deviation_l2, weight=-0.5,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"],
        )},
    )
    # ===== [#13] dof acc — PAPER EXACT -2.5e-7 =====
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # ===== [#14] dof vel — PAPER EXACT -5e-4 =====
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-5.0e-4)
    # ===== [#15] torques — PAPER EXACT -1e-5 =====
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    # ===== [#16] action rate — PAPER -0.3 → v6 SIM-TO-SIM FIX -5e-3 =====
    # v5 PROVED IN PRACTICE that paper-exact -0.3 collapses PPO to
    # stand-still on flat-ground baseline (success=1.000, avg_dist=0.05m).
    # See module docstring "WHY v5 → v6" for the full math.
    #
    # In the FULL CMoE training task (cmoe/cmoe_env_cfg.py), -0.3 stays
    # paper-exact because tracking_weight_curriculum + multi-terrain
    # terminations protect against stand-still. But this baseline has
    # NEITHER protection, so it needs the IsaacLab-validated -5e-3.
    #
    # This is the ONLY of 20 paper Table II weights changed in v6.
    # All other 19 weights remain paper-exact.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5.0e-3)
    # ===== [#17] dof pos limits — PAPER EXACT -2.0 =====
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    # ===== [#18] dof vel limits — PAPER EXACT -1.0 =====
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits, weight=-1.0,
        params={"soft_ratio": 1.0},
    )
    # ===== [#19] torque limits — PAPER EXACT -1.0 =====
    torque_limits = RewTerm(func=mdp.applied_torque_limits_paper, weight=-1.0)
    # ===== [#20] feet edge — disabled on flat ground per paper §IV-C =====
    # Paper §IV-C verbatim: "the foot edge reward is activated only when
    # the robot is trained in a hurdle or gap environment." So on Phase 1
    # flat ground this term is correctly OMITTED (paper-compliant).

    # =================================================================
    # 2 EXTRA TERMS (BEYOND PAPER) — IsaacLab POSE STABILIZERS
    # =================================================================
    #
    # These compensate for IsaacLab's permissive upper-body physics
    # vs. IsaacGym. They use the SAME functional form as paper's
    # hip_dof_error (joint_deviation_l2) — just on different joint
    # groups. They are 0 at default joint pose, so they do NOT
    # incentivize stand-still vs. walking.
    #
    # These two terms specifically address the weird poses observed
    # in v4 evaluation screenshots.

    # [EXTRA-1] Arm joints deviation -1.0
    # ROLE: Constrains shoulder/elbow/wrist to stay near default.
    # FIXES IN IMAGES:
    #   - Awkward arm positions (one arm forward, one back at random)
    #   - Arms locked at extreme angles (full-elbow-bent / hyper-extended)
    #   - "T-pose" or "Y-pose" arms during walking
    # JOINTS COVERED (14 of 29):
    #   .*_shoulder_pitch_joint   (defaults: 0.3 rad)
    #   .*_shoulder_roll_joint    (defaults: ±0.25 rad)
    #   .*_shoulder_yaw_joint     (default: 0)
    #   .*_elbow_joint            (default: 0.97 rad — natural bent)
    #   .*_wrist_roll_joint       (defaults: ±0.15 rad)
    #   .*_wrist_pitch_joint      (default: 0)
    #   .*_wrist_yaw_joint        (default: 0)
    # WEIGHT RATIONALE: -1.0 matches the magnitude of feet_stumble (-1.0).
    # Slightly stronger than hip_dof_error (-0.5) because there are 7×2=14
    # arm joints vs. only 2×2=4 hip yaw/roll joints — so per-joint
    # penalty stays comparable.
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

    # [EXTRA-2] Waist joints deviation -1.0
    # ROLE: Keeps torso upright and aligned (no twist, no lean).
    # FIXES IN IMAGES:
    #   - Torso leaning forward / backward beyond what the orientation
    #     penalty (#5) constrains (orientation only checks gravity in
    #     the BASE frame; waist twist relative to pelvis is invisible
    #     to it)
    #   - Torso twisting left/right during walking (waist_yaw drift)
    # JOINTS COVERED (3 of 29):
    #   waist_yaw_joint    (default: 0)
    #   waist_roll_joint   (default: 0)
    #   waist_pitch_joint  (default: 0)
    # WEIGHT RATIONALE: -1.0 — same scale as joint_deviation_arms.
    # Only 3 joints, so per-joint pressure is higher (which is
    # appropriate — waist drift is more visually disruptive).
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l2, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        )},
    )

    # [EXTRA-3] no_fly -5.0 — anti-bunny-hop (NEW in v7)
    # ROLE: Punishes any timestep where BOTH feet are simultaneously
    #       airborne (force < 1.0 N on both ankle_roll_link's).
    # FIXES IN IMAGES:
    #   - Bunny-hopping: both feet leave/land ground at same moment
    #   - Synchronized leg push (looks like jumping forward)
    # FUNCTION: mdp.desired_contacts (IsaacLab built-in, imported via
    #           `from isaaclab.envs.mdp import *` in cmoe.mdp.__init__).
    #   Returns -1 ONLY when ALL listed contact bodies have force
    #   below threshold simultaneously (i.e., all feet in air).
    #   Returns 0 when at least one foot has contact > threshold.
    # WALKING IS UNAFFECTED:
    #   In normal walking, exactly one foot is in swing phase at a
    #   time, so the OTHER foot has contact → no penalty.
    #   Only bunny-hopping triggers the penalty.
    # WEIGHT RATIONALE: -5.0 matches humanoid_locomotion reference
    #   (proven on IsaacLab + Unitree H1). At ~50% both-airborne
    #   fraction during bunny-hop, this gives ~-2.5/s penalty,
    #   decisively dominating bunny-hop's +0.46/s bonus.
    no_fly = RewTerm(
        func=mdp.desired_contacts, weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*ankle_roll_link",
            ),
            "threshold": 1.0,
        },
    )


# ============================================================
# Terminations  — paper §V-A
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
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["torso_link", "pelvis"],
            ),
            "threshold": 1.0,
        },
    )


# ============================================================
# Env cfg
# ============================================================
@configclass
class CmoeBaselineEnvCfg(ManagerBasedRLEnvCfg):
    """Phase 1.5 v7. Flat ground + simple MLP +
       19/20 paper Table II rewards paper-exact +
       1 sim-to-sim weight fix (action_rate -0.3 → -5e-3) +
       3 IsaacLab pose/gait stabilizer additions
       (joint_deviation_arms/waist + no_fly anti-bunny-hop)."""

    scene: BaselineSceneCfg        = BaselineSceneCfg(num_envs=4096, env_spacing=2.5)
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
class CmoeBaselineEnvCfg_PLAY(CmoeBaselineEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs   = 32
        self.episode_length_s = 20.0
        self.commands.base_velocity.rel_standing_envs = 0.0
