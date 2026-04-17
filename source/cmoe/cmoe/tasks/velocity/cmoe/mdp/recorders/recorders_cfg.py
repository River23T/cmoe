# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from . import recorders

##
# State recorders.
##


@configclass
class InitialStateRecorderCfg(RecorderTermCfg):
    """Configuration for the initial state recorder term."""

    class_type: type[RecorderTerm] = recorders.InitialStateRecorder


@configclass
class LinearTrackingErrorRecorderRecorderCfg(RecorderTermCfg):
    """Configuration for the step state recorder term."""

    class_type: type[RecorderTerm] = recorders.LinearTrackingErrorRecorder


@configclass
class AngularTrackingErrorRecorderRecorderCfg(RecorderTermCfg):
    """Configuration for the step state recorder term."""

    class_type: type[RecorderTerm] = recorders.AngularTrackingErrorRecorder


@configclass
class SubTerrainIndexRecorderCfg(RecorderTermCfg):
    """Configuration for the step state recorder term."""

    class_type: type[RecorderTerm] = recorders.SubTerrainIndexRecorder


@configclass
class CountRecorderCfg(RecorderTermCfg):
    """Configuration for the step state recorder term."""

    class_type: type[RecorderTerm] = recorders.CountRecorder


@configclass
class TerminationRecorderCfg(RecorderTermCfg):
    """Configuration for the step state recorder term."""

    class_type: type[RecorderTerm] = recorders.TerminationRecorder


@configclass
class PreStepActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the step action recorder term."""

    class_type: type[RecorderTerm] = recorders.PreStepActionsRecorder


@configclass
class PreStepFlatPolicyObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation recorder term."""

    class_type: type[RecorderTerm] = recorders.PreStepFlatPolicyObservationsRecorder


@configclass
class PostStepProcessedActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the post step processed actions recorder term."""

    class_type: type[RecorderTerm] = recorders.PostStepProcessedActionsRecorder


##
# Recorder manager configurations.
##


@configclass
class TrackingErrorRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configurations for recording actions and states."""

    # record_initial_state = InitialStateRecorderCfg()
    record_linear_tracking_error = LinearTrackingErrorRecorderRecorderCfg()
    record_angular_tracking_error = AngularTrackingErrorRecorderRecorderCfg()
    sub_terrain_index = SubTerrainIndexRecorderCfg()
    count = CountRecorderCfg()
    termination = TerminationRecorderCfg()
    # record_pre_step_actions = PreStepActionsRecorderCfg()
    # record_pre_step_flat_policy_observations = PreStepFlatPolicyObservationsRecorderCfg()
    # record_post_step_processed_actions = PostStepProcessedActionsRecorderCfg()
