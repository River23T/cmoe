# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CMoE observations: elevation map with paper §IV-B domain randomization.

Paper §IV-B details TWO elevation-map randomizations that earlier versions
of this codebase missed:

  1. **Nonlinear salt-and-pepper noise** (Paper Eq. 9). For each height
     sample h(i):
       h(i) =  U(M, 2M - m)     with prob p  ("salt")
       h(i) =  U(2m - M, m)     with prob p  ("pepper")
       h(i) =  h(i)             with prob 1 - 2p

     Where M, m are the max / min values of the elevation map at the
     corresponding point (per-env window, per-step).

  2. **Edge chamfering** (Paper §IV-B). Real-world elevation sensors
     produce smooth curves rather than sharp corners at terrain edges.
     We apply a small depth-wise smoothing pass to mimic this.

This module exposes one new observation function:

    ``height_scan_sp(env, sensor_cfg, sp_prob, chamfer_enable)``

which wraps IsaacLab's default ``height_scan`` and applies both effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

# IsaacLab's built-in height_scan — we call it first, then augment.
try:
    from isaaclab.envs.mdp.observations import height_scan as _isaaclab_height_scan
except Exception:  # pragma: no cover - fallback only
    _isaaclab_height_scan = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _fallback_height_scan(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Identical to IsaacLab's height_scan: sensor_z - ray_hit_z - 0.5 (offset bias)."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    sensor_z = sensor.data.pos_w[:, 2].unsqueeze(1)
    ray_z = sensor.data.ray_hits_w[..., 2]
    scan = sensor_z - ray_z - 0.5
    return scan


def height_scan_sp(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    sp_prob: float = 0.02,
    chamfer_enable: bool = True,
) -> torch.Tensor:
    """Elevation map with salt-and-pepper noise (Paper Eq. 9) and edge chamfering.

    Args:
        env: Isaac Lab RL env.
        sensor_cfg: RayCaster entity config (the torso-mounted height scanner).
        sp_prob: per-point probability p of salt OR pepper. Paper §IV-B.
        chamfer_enable: apply edge chamfering (paper §IV-B).

    Returns:
        [N, num_rays] elevation scan with domain-randomized values.
    """
    if _isaaclab_height_scan is not None:
        scan = _isaaclab_height_scan(env, sensor_cfg)
    else:
        scan = _fallback_height_scan(env, sensor_cfg)

    # Safety clip
    scan = torch.nan_to_num(scan, nan=0.0, posinf=1.5, neginf=-1.5)
    scan = torch.clamp(scan, min=-1.5, max=1.5)

    N, R = scan.shape
    device = scan.device

    # ---- Edge chamfering (paper §IV-B) ----
    if chamfer_enable and R >= 3:
        kernel = torch.tensor(
            [0.25, 0.5, 0.25], device=device, dtype=scan.dtype
        ).view(1, 1, 3)
        padded = torch.nn.functional.pad(scan.unsqueeze(1), (1, 1), mode="replicate")
        scan = torch.nn.functional.conv1d(padded, kernel).squeeze(1)

    # ---- Salt-and-pepper noise (Paper Eq. 9) ----
    h_max = scan.max(dim=1, keepdim=True).values  # [N, 1]
    h_min = scan.min(dim=1, keepdim=True).values  # [N, 1]

    u = torch.rand(N, R, device=device)
    salt_mask   = u < sp_prob
    pepper_mask = (u >= sp_prob) & (u < 2.0 * sp_prob)

    u01 = torch.rand(N, R, device=device)
    delta = (h_max - h_min)
    salt_val   = h_max + delta * u01
    pepper_val = h_min - delta * u01

    scan = torch.where(salt_mask,   salt_val,   scan)
    scan = torch.where(pepper_mask, pepper_val, scan)

    return scan
