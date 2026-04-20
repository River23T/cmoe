"""CMoE curriculum functions.

Provides ``lin_vel_cmd_levels`` which gradually expands the velocity command
ranges toward the configured ``limit_ranges`` as the agent improves.

This replaces the unitree_rl_lab.tasks.locomotion.mdp.curriculums module.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    """Curriculum that widens velocity-command ranges as tracking improves.

    At the end of each episode cycle, if the mean tracking reward exceeds 80%
    of its weight, the linear velocity command ranges are expanded by +/-0.1
    (clamped to ``limit_ranges``).

    Args:
        env: The environment instance.
        env_ids: The environment indices that have been reset.
        reward_term_name: Name of the reward term to monitor for progress.

    Returns:
        Current maximum lin_vel_x as a scalar tensor (for logging).
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(
        env.reward_manager._episode_sums[reward_term_name][env_ids]
    ) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)
