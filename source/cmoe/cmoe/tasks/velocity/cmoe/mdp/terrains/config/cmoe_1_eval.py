# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains - Evaluation."""

import source.cmoe.cmoe.tasks.velocity.cmoe.mdp.terrains as cmoe_1_terrains_gen
from source.cmoe.cmoe.tasks.velocity.cmoe.mdp.terrains.cmoe_1_terrains_cfg import (
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

CMOE1_EVALUATE_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=10,
    num_cols=9,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(1.0, 1.0),  # 评估时固定最高难度
    use_cache=False,
    sub_terrains={
        # 1. 上坡 Slope Up: 0-20°
        "SlopeUp": HfPyramidSlopedTerrainCfg(
            function=cmoe_1_terrains_gen.pyramid_sloped_terrain,
            slope_range=(0.0, 0.364),
            platform_width=0.5,
            inverted=False,
            border_width=0.5,
        ),
        # 2. 下坡 Slope Down: 0-20°
        "SlopeDown": HfInvertedPyramidSlopedTerrainCfg(
            function=cmoe_1_terrains_gen.pyramid_sloped_terrain,
            slope_range=(0.0, 0.364),
            platform_width=0.5,
            inverted=True,
            border_width=0.5,
        ),
        # 3. 上台阶 Stair Up: step height 0.05-0.23m
        "StairUp": MeshPyramidStairsTerrainCfg(
            function=cmoe_1_terrains_gen.pyramid_stairs_terrain,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=1.0,
            holes=False,
            border_width=0.0,
        ),
        # 4. 下台阶 Stair Down: step height 0.05-0.23m
        "StairDown": MeshInvertedPyramidStairsTerrainCfg(
            function=cmoe_1_terrains_gen.inverted_pyramid_stairs_terrain,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=1.0,
            holes=False,
            border_width=0.0,
        ),
        # 5. 沟壑 Gap: width 0.1-0.8m
        "Gap": MeshGapTerrainCfg(
            function=cmoe_1_terrains_gen.gap_terrain,
            gap_width_range=(0.1, 0.8),
            platform_width=1.0,
        ),
        # 6. 障碍 Hurdle: height 0.2-0.4m, width 0.1-0.3m
        "Hurdle": MeshRepeatedBoxesTerrainCfg(
            function=cmoe_1_terrains_gen.repeated_objects_terrain,
            object_type=cmoe_1_terrains_gen.make_box,
            object_params_start=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=3,
                height=0.2,
                size=(0.1, 8.0),
                max_yx_angle=0.0,
                degrees=True,
            ),
            object_params_end=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=5,
                height=0.4,
                size=(0.3, 8.0),
                max_yx_angle=0.0,
                degrees=True,
            ),
            platform_width=1.0,
            platform_height=-1.0,
        ),
        # 7. 离散凸起 Discrete: height 0.1-0.2m
        "Discrete": HfDiscreteObstaclesTerrainCfg(
            function=cmoe_1_terrains_gen.discrete_obstacles_terrain,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.25, 0.75),
            obstacle_height_range=(0.1, 0.2),
            num_obstacles=50,
            platform_width=1.0,
            border_width=0.5,
        ),
        # 8. 混合地形1 Mix1: gap 0.1-0.8m + stairs 0.1-0.15m
        "Mix1": MeshMix1TerrainCfg(
            function=cmoe_1_terrains_gen.mix1_gap_stairs_terrain,
            gap_width_range=(0.1, 0.8),
            step_height_range=(0.1, 0.15),
            step_width=0.3,
            start_platform_length=1.0,
            num_steps_per_section=3,
            num_sections=5,
        ),
        # 9. 混合地形2 Mix2: bridge 0.5-1.0m + stairs 0.1-0.25m
        "Mix2": MeshMix2TerrainCfg(
            function=cmoe_1_terrains_gen.mix2_bridge_stairs_terrain,
            bridge_width_range=(0.5, 1.0),
            step_height_range=(0.1, 0.25),
            step_width=0.3,
            platform_width=1.5,
            num_steps=4,
        ),
    },
)
"""CMoE paper terrains configuration for evaluation (max difficulty)."""
