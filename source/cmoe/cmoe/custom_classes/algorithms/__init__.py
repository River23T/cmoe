# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learning algorithms."""


def __getattr__(name):
    if name == "PPO":
        from .ppo import PPO
        return PPO
    if name == "Distillation":
        from .distillation import Distillation
        return Distillation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PPO", "Distillation"]
