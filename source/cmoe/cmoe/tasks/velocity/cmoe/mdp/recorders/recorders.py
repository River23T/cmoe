# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.utils.math import quat_apply_inverse
import numpy as np


class InitialStateRecorder(RecorderTerm):
    """Recorder term that records the initial state of the environment after reset."""

    def record_post_reset(self, env_ids: Sequence[int] | None):
        def extract_env_ids_values(value):
            nonlocal env_ids
            if isinstance(value, dict):
                return {k: extract_env_ids_values(v) for k, v in value.items()}
            return value[env_ids]

        return "initial_state", extract_env_ids_values(self._env.scene.get_state(is_relative=True))


class LinearTrackingErrorRecorder(RecorderTerm):
    """Recorder term that records the state of the environment at the end of each step."""

    def record_post_step(self):
        world_linear = self._env.scene.get_state(is_relative=True)["articulation"]["robot"]["root_velocity"][:, :3]
        robot_pose = self._env.scene.get_state(is_relative=True)["articulation"]["robot"]["root_pose"][:, 3:7]
        robot_linear = quat_apply_inverse(robot_pose, world_linear)
        command = self._env.command_manager.get_command("base_velocity")
        linear_error = torch.abs(robot_linear[:,0:1] - command[:,0:1])
        return "linear_tracking_error", linear_error


class AngularTrackingErrorRecorder(RecorderTerm):
    """Recorder term that records the state of the environment at the end of each step."""

    def record_post_step(self):
        world_angular = self._env.scene.get_state(is_relative=True)["articulation"]["robot"]["root_velocity"][:,3:]
        robot_pose = self._env.scene.get_state(is_relative=True)["articulation"]["robot"]["root_pose"][:, 3:7]
        robot_angular = quat_apply_inverse(robot_pose, world_angular)
        command = self._env.command_manager.get_command("base_velocity")
        angular_error = torch.abs(robot_angular[:,2:] - command[:,2:])
        return "angular_tracking_error", angular_error


class SubTerrainIndexRecorder(RecorderTerm):
    """Recorder term that records the state of the environment at the end of each step."""

    def record_post_reset(self, env_ids: Sequence[int] | None):
        sub_terrain_index = self._env.scene.terrain.terrain_types[env_ids]
        return "sub_terrain", sub_terrain_index


class CountRecorder(RecorderTerm):
    """Recorder term that records the state of the environment at the end of each step."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.episode_count = torch.zeros(self._env.scene.num_envs, dtype=torch.long, device=self._env.device)

    def record_post_reset(self, env_ids: Sequence[int] | None):
        self.episode_count[env_ids] += 1
        return "episode_count", self.episode_count[env_ids]


class TerminationRecorder(RecorderTerm):
    """Recorder term that records the state of the environment at the end of each step."""

    def record_pre_reset(self, env_ids: Sequence[int] | None):
        if not hasattr(self, "_startup_done"):
            self._startup_done = True
            return None, None
        reasons = torch.zeros(len(env_ids), dtype=torch.long, device=self._env.device)
        for idx, term_name in enumerate(self._env.termination_manager.active_terms):
            triggered = self._env.termination_manager.get_term(term_name)[env_ids] > 0.5
            reasons[triggered] = idx
        return "Termination", reasons


class PreStepActionsRecorder(RecorderTerm):
    """Recorder term that records the actions in the beginning of each step."""

    def record_pre_step(self):
        return "actions", self._env.action_manager.action


class PreStepFlatPolicyObservationsRecorder(RecorderTerm):
    """Recorder term that records the policy group observations in each step."""

    def record_pre_step(self):
        return "obs", self._env.obs_buf["policy"]


class PostStepProcessedActionsRecorder(RecorderTerm):
    """Recorder term that records processed actions at the end of each step."""

    def record_post_step(self):
        processed_actions = None

        # Loop through active terms and concatenate their processed actions
        for term_name in self._env.action_manager.active_terms:
            term_actions = self._env.action_manager.get_term(term_name).processed_actions.clone()
            if processed_actions is None:
                processed_actions = term_actions
            else:
                processed_actions = torch.cat([processed_actions, term_actions], dim=-1)

        return "processed_actions", processed_actions
