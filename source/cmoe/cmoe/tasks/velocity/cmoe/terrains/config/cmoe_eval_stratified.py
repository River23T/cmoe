# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stratified evaluation terrain — full difficulty range (0..1)."""

from .. import cmoe_terrains as cmoe_terrains_gen
from ..cmoe_terrains_cfg import (
    HfPyramidSlopedTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshGapTerrainCfg,
    MeshRepeatedBoxesTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
    MeshMix1TerrainCfg,
    MeshMix2TerrainCfg,
)

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


CMOE_EVALUATE_STRATIFIED_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=10,
    num_cols=9,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "SlopeUp": HfPyramidSlopedTerrainCfg(
            function=cmoe_terrains_gen.pyramid_sloped_terrain,
            slope_range=(0.0, 0.364),
            platform_width=0.5,
            inverted=False,
            border_width=0.5,
        ),
        "SlopeDown": HfInvertedPyramidSlopedTerrainCfg(
            function=cmoe_terrains_gen.pyramid_sloped_terrain,
            slope_range=(0.0, 0.364),
            platform_width=0.5,
            inverted=True,
            border_width=0.5,
        ),
        "StairUp": MeshPyramidStairsTerrainCfg(
            function=cmoe_terrains_gen.pyramid_stairs_terrain,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=1.0,
            holes=False,
            border_width=0.0,
        ),
        "StairDown": MeshInvertedPyramidStairsTerrainCfg(
            function=cmoe_terrains_gen.inverted_pyramid_stairs_terrain,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=1.0,
            holes=False,
            border_width=0.0,
        ),
        "Gap": MeshGapTerrainCfg(
            function=cmoe_terrains_gen.gap_terrain,
            gap_width_range=(0.1, 0.8),
            platform_width=1.0,
        ),
        "Hurdle": MeshRepeatedBoxesTerrainCfg(
            function=cmoe_terrains_gen.repeated_objects_terrain,
            object_type=cmoe_terrains_gen.make_box,
            object_params_start=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=3, height=0.2, size=(0.1, 8.0),
                max_yx_angle=0.0, degrees=True,
            ),
            object_params_end=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=5, height=0.4, size=(0.3, 8.0),
                max_yx_angle=0.0, degrees=True,
            ),
            platform_width=1.0,
            platform_height=-1.0,
        ),
        "Discrete": HfDiscreteObstaclesTerrainCfg(
            function=cmoe_terrains_gen.discrete_obstacles_terrain,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.25, 0.75),
            obstacle_height_range=(0.1, 0.2),
            num_obstacles=50,
            platform_width=1.0,
            border_width=0.5,
        ),
        "Mix1": MeshMix1TerrainCfg(
            function=cmoe_terrains_gen.mix1_gap_stairs_terrain,
            gap_width_range=(0.1, 0.8),
            step_height_range=(0.1, 0.15),
            step_width=0.3,
            start_platform_length=1.0,
            num_steps_per_section=3,
            num_sections=5,
        ),
        "Mix2": MeshMix2TerrainCfg(
            function=cmoe_terrains_gen.mix2_bridge_stairs_terrain,
            bridge_width_range=(0.5, 1.0),
            step_height_range=(0.1, 0.25),
            step_width=0.3,
            platform_width=1.5,
            num_steps=4,
        ),
    },
)
