# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CMoE-specific MDP functions.

Import order matters — *later imports override earlier ones* with the same
name. We therefore import in this order:

    1. ``isaaclab.envs.mdp``                      — baseline IsaacLab MDP funcs
       (track_lin_vel_xy_exp, lin_vel_z_l2, ..., joint_pos_limits, ...)

    2. ``unitree_rl_lab.tasks.locomotion.mdp``    — Unitree's extensions
       (UniformLevelVelocityCommandCfg, terrain_levels_vel, lin_vel_cmd_levels,
        randomize_actuator_gains, feet_gait, feet_slide, foot_clearance_reward,
        energy, ...)

    3. ``.rewards``                               — CMoE paper-specific rewards
       (track_ang_vel_z_exp_l1, feet_stumble, feet_lateral_distance,
        feet_air_time, feet_ground_parallel, joint_deviation_l2,
        applied_torque_limits_paper, feet_edge, ...)
       Any CMoE reward with the same name as an IsaacLab one will
       override the IsaacLab version here — intended.
"""

from isaaclab.envs.mdp import *                          # noqa: F401, F403
from unitree_rl_lab.tasks.locomotion.mdp import *        # noqa: F401, F403
from .rewards import *                                    # noqa: F401, F403
