# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.utils.math import quat_apply, quat_apply_inverse

# 1.速度跟踪（velocity tracking）
def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)

# 2.航向跟踪（yaw tracking）（自定义）
# ----- [#2] Yaw tracking (L1 inside exp, no σ) -----
def track_ang_vel_z_exp_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Paper: R_yaw = exp(-|ψ_cmd - ψ|), weight +2.0.

    Same as ``track_ang_vel_z_exp`` but uses L1 (|·|) inside the exponential
    instead of L2 with σ², matching the paper exactly.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    err = torch.abs(
        env.command_manager.get_command(command_name)[:, 2]
        - asset.data.root_ang_vel_b[:, 2]
    )
    return torch.exp(-err)

# 3.z轴速度（z velocity）
def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])

# 4.横滚-俯仰角速度（roll-pitch velocity）
def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)

# 5.姿态（orientation）
def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

# 6.基座高度（base height）
def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)

# 7.足部绊脚（feet stumble）（自定义）
# ----- [#7] Feet stumble -----
def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Paper: ⋁_{i∈feet} {‖F_{i,xy}‖₂ > 3·|F_{i,z}|}, weight -1.0.

    Returns 1.0 if ANY foot's lateral force exceeds 3× its vertical force in
    the recent contact-history window, otherwise 0.0.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w_history shape: [N, history_len, num_bodies, 3]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    f_xy = torch.norm(forces[..., :2], dim=-1)   # [N, hist, F]
    f_z = torch.abs(forces[..., 2])              # [N, hist, F]
    stumble = f_xy > 3.0 * f_z                   # [N, hist, F]
    return stumble.any(dim=-1).any(dim=-1).float()

# 8.碰撞（collision）
def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)

# 9.足部间距（feet distance）（自定义）
# ----- [#9] Feet lateral distance -----
def feet_lateral_distance(
    env: ManagerBasedRLEnv,
    d_min: float,
    d_max: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Paper: min(|p_{y,0} - p_{y,1}| - d_min, d_max - d_min), weight +0.8.

    Computed in the **body frame** (y is the lateral axis). The reward
    saturates at (d_max - d_min) when the feet are at or beyond d_max apart,
    and becomes negative when the feet are closer than d_min — penalizing
    foot collisions/over-narrow stances.

    ``asset_cfg.body_ids`` must be exactly the two feet, in any order.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]  # [N, 2, 3]
    foot_vec_w = foot_pos_w[:, 0] - foot_pos_w[:, 1]           # [N, 3]
    foot_vec_b = quat_apply_inverse(asset.data.root_quat_w, foot_vec_w)
    feet_y_dist = torch.abs(foot_vec_b[:, 1])                  # [N]
    return torch.clamp(feet_y_dist - d_min, max=d_max - d_min)

# 10.足部腾空时间（feet air time）（自定义）
# ----- [#10] Feet air time -----
def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Paper: Σ_{i=1}^{2}(t_air,i - t_target_air) · F_i, weight +1.0.

    F_i is the binary "first contact" indicator (1 only at the timestep where
    foot i transitions from air to ground). ``last_air_time`` then holds the
    air-time the foot just finished. Reward only fires while a non-trivial
    locomotion command is active, otherwise standing still would be penalized.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    cmd = env.command_manager.get_command(command_name)
    cmd_active = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) > 0.1
    return reward * cmd_active.float()

# 11.足部地面平行度（feet ground parallel）（自定义）
# ----- [#11] Feet ground parallel -----
def feet_ground_parallel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_half_length: float = 0.10,
    foot_half_width: float = 0.04,
) -> torch.Tensor:
    """Paper: Σ_{i=1}^{2} Var(p_{z,i}), weight -0.02.

    Sample 4 corners on each foot (in the foot's local frame), transform to
    world frame, and take the variance of their z-heights. Sum across feet.
    Encourages the foot sole to remain parallel to the ground.

    ⚠ Adjust ``foot_half_length`` / ``foot_half_width`` to match the actual
    geometry of the Unitree G1 ankle/foot link.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]    # [N, F, 3]
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]  # [N, F, 4]
    N, F = foot_pos_w.shape[:2]

    corners_b = torch.tensor(
        [
            [ foot_half_length,  foot_half_width, 0.0],
            [ foot_half_length, -foot_half_width, 0.0],
            [-foot_half_length,  foot_half_width, 0.0],
            [-foot_half_length, -foot_half_width, 0.0],
        ],
        device=foot_pos_w.device,
        dtype=foot_pos_w.dtype,
    )  # [4, 3]

    quat_exp = foot_quat_w.unsqueeze(2).expand(N, F, 4, 4).reshape(-1, 4)
    corners_exp = corners_b.expand(N, F, 4, 3).reshape(-1, 3)
    corners_rot = quat_apply(quat_exp, corners_exp).reshape(N, F, 4, 3)
    corners_w = foot_pos_w.unsqueeze(2) + corners_rot           # [N, F, 4, 3]
    z_var = corners_w[..., 2].var(dim=2)                        # [N, F]
    return torch.sum(z_var, dim=1)

# 12.髋关节自由度误差（hip dof error）（自定义）
# ----- [#12] Hip joint deviation, L2 -----
def joint_deviation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Paper: Σ_{i∈hip}(θ_i - θ_default,i)², weight -0.5.

    L2 version of the official ``joint_deviation_l1``. Specify the hip joints
    via ``asset_cfg.joint_names`` in the RewTerm config.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    angle = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.square(angle), dim=1)

# 13.关节角加速度（dof acc）
def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    .. note::
        Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations
        contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)

# 14.关节角速度（dof vel）
def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    .. note::
        Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities
        contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)

# 15.关节扭矩（torques）
def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    .. note::
        Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques
        contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)

# 16.动作变化率（action rate）
def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)

# 17.关节位置限制（dof pos limits）
def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)

# 18.关节速度限制（dof vel limits）
def joint_vel_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint velocity and the soft limits.

    Args:
        soft_ratio: The ratio of the soft limits to be used.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = (
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
        - asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)

# 19.扭矩限制（torque limits）（自定义）
# ----- [#19] Torque-limit penalty (paper formula) -----
def applied_torque_limits_paper(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Paper: Σ ReLU(|τ_i| - |τ_max,i|), weight -1.0.

    Note: the official ``applied_torque_limits`` computes
    Σ |τ_applied - τ_computed|, which is the actuator-tracking error, not the
    paper's per-joint over-limit penalty. This version directly compares the
    applied torque magnitude against the joint's effort limit.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    applied_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    # joint effort limits as exposed by Articulation data
    torque_max = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
    out_of_limits = torch.abs(applied_torque) - torque_max
    return torch.sum(out_of_limits.clip(min=0.0), dim=1)

# 20.足部边缘触地（feet edge）（自定义）
# ----- [#20] Feet edge -----
def feet_edge(
    env: ManagerBasedRLEnv,
    sensor_cfg_left: SceneEntityCfg,
    sensor_cfg_right: SceneEntityCfg,
    edge_height_threshold: float = 0.02,
) -> torch.Tensor:
    """Paper: 1_{foot at edge of terrain}, weight -1.0 (terrain-conditional).

    Implementation: a small grid-pattern ``RayCaster`` is attached to each
    foot. If the std of ground heights sampled under any foot exceeds
    ``edge_height_threshold``, that foot is flagged as standing on an edge.
    Returns the count of feet on edges (0/1/2).

    ⚠ The paper enables this reward ONLY for the Hurdle and Gap terrains.
    Use a curriculum/rewards-cfg gating per terrain type — e.g. set the term
    weight to 0 for non-hurdle/gap envs in your env's curriculum manager.
    """
    sensor_left: RayCaster = env.scene.sensors[sensor_cfg_left.name]
    sensor_right: RayCaster = env.scene.sensors[sensor_cfg_right.name]
    # ray_hits_w: [N, num_rays, 3]; compute height std under each foot
    ray_z_left = sensor_left.data.ray_hits_w[..., 2]
    ray_z_right = sensor_right.data.ray_hits_w[..., 2]
    edge_left = (ray_z_left.std(dim=1) > edge_height_threshold).float()
    edge_right = (ray_z_right.std(dim=1) > edge_height_threshold).float()
    return edge_left + edge_right
