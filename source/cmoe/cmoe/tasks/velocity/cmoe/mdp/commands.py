"""Custom velocity command config with curriculum limit_ranges.

This replaces the unitree_rl_lab.tasks.locomotion.mdp.commands module.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """Uniform velocity command config extended with ``limit_ranges`` for curriculum.

    ``ranges`` starts narrow and is gradually expanded toward ``limit_ranges``
    by the ``lin_vel_cmd_levels`` curriculum function.
    """

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """The maximum command ranges that the curriculum can expand ``ranges`` to."""
