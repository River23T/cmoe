# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to specify the symmetry in the observation and action space for ANYmal."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    four symmetrical transformations: original, left-right, front-back, and diagonal. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """

    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we have 4 different symmetries, we need to augment the batch size by 4
        obs_aug = obs.repeat(2)

        # actor group
        # -- original
        obs_aug["actor"][:batch_size] = obs["actor"][:]
        # -- left-right
        obs_aug["actor"][batch_size : 2 * batch_size] = _transform_proprioception_left_right(obs["actor"])

        # critic group
        # -- original
        obs_aug["critic"][:batch_size] = obs["critic"][:]
        # -- left-right
        obs_aug["critic"][batch_size: 2 * batch_size] = _transform_proprioception_left_right(obs["critic"])

        # actor map group
        # -- original
        obs_aug["actor_map"][:batch_size] = obs["actor_map"][:]
        # -- left-right
        obs_aug["actor_map"][batch_size: 2 * batch_size] = _transform_map_scans_left_right(obs["actor_map"])

        # critic map group
        # -- original
        obs_aug["critic_map"][:batch_size] = obs["critic_map"][:]
        # -- left-right
        obs_aug["critic_map"][batch_size: 2 * batch_size] = _transform_map_scans_left_right(obs["critic_map"])
    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 4 different symmetries, we need to augment the batch size by 4
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _transform_proprioception_left_right(obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the proprioception tensor.

    This function modifies the given observation tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    negating certain components of the linear and angular velocities, projected gravity,
    velocity commands, and flipping the joint positions, joint velocities, and last actions
    for the H1 robot.

    Args:
        obs: The observation tensor to be transformed.

    Returns:
        The transformed proprioception tensor with left-right symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # base_lin_vel
    obs[:, :3] = obs[:, :3] * torch.tensor([1, -1, 1], device=device)
    # base_ang_vel
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([-1, 1, -1], device=device)
    # projected_gravity
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1, -1, 1], device=device)
    # velocity_commands
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([1, -1, -1], device=device)
    # joint_pos_rel
    obs[:, 12:31] = _switch_h1_joints_left_right(obs[:, 12:31])
    # joint_vel_rel
    obs[:, 31:50] = _switch_h1_joints_left_right(obs[:, 31:50])
    # last_action
    obs[:, 50:69] = _switch_h1_joints_left_right(obs[:, 50:69])
    # contact_state
    if obs.shape[1] >= 80:
        obs[:, 69:88] = _switch_h1_joints_left_right(obs[:, 69:88])

    return obs


def _transform_map_scans_left_right(obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the map scans tensor.

    This function modifies the given map scans tensor by applying transformations
    that represent a symmetry with respect to the left-right axis.
    for the H1 robot.

    Args:
        obs: The map scans tensor to be transformed.

    Returns:
        The transformed observation tensor with left-right symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    obs[:, 2:3, :, :] = torch.flip(obs[:, 2:3, :, :], dims=[2])
    obs[:, 1:2, :, :] = -obs[:, 1:2, :, :]

    return obs


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    ANYmal robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    actions = actions.clone()
    actions[:] = _switch_h1_joints_left_right(actions[:])
    return actions



def _switch_h1_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    joint_data_switched = torch.zeros_like(joint_data)
    # left <-- right
    joint_data_switched[..., [0, 2, 3, 5, 7, 9, 11, 13, 15, 17]] = joint_data[..., [1, 2, 4, 6, 8, 10, 12, 14, 16, 18]]
    # right <-- left
    joint_data_switched[..., [1, 4, 6, 8, 10, 12, 14, 16, 18]] = joint_data[..., [0, 3, 5, 7, 9, 11, 13, 15, 17]]

    # Flip the sign of the HAA joints
    joint_data_switched[..., [0, 1, 2, 3, 4, 9, 10, 13, 14, ]] *= -1.0

    return joint_data_switched
