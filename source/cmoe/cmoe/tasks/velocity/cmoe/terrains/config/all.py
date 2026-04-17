# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

ALL_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=2.5,
    num_rows=10,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "Random": terrain_gen.HfRandomUniformTerrainCfg(
            noise_range=(0.0, 0.3),
            noise_step=(0.005),
            downsampled_scale=0.2,
            border_width=0.5,
        ),
        "Sloped": terrain_gen.HfPyramidSlopedTerrainCfg(
            slope_range=(0.1, 0.4),
            platform_width=2.,
            inverted=False,
            border_width=0.5,
        ),
        "SlopedInverted": terrain_gen.HfPyramidSlopedTerrainCfg(
            slope_range=(0.1, 0.4),
            platform_width=2.,_
            inverted=True,
            border_width=0.5,
        ),
        "Stairs": terrain_gen.HfPyramidStairsTerrainCfg(
            step_height_range=(0.1, 0.5),
            step_width=0.3,
            inverted=False,
            platform_width=2.,
            border_width=0.5,
        ),
        "StairsInverted": terrain_gen.HfPyramidStairsTerrainCfg(
            step_height_range=(0.1, 0.5),
            step_width=0.3,
            inverted=True,
            platform_width=2.,
            border_width=0.5,
        ),
        "Obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            obstacle_height_range=(0.0, 0.2),
            num_obstacles=300,
            obstacle_width_range=(0.2, 0.6),
            obstacle_height_mode="fixed",
            platform_width=2.,
            border_width=0.5,
        ),
        "Wave": terrain_gen.HfWaveTerrainCfg(
            amplitude_range=(0,0.5),
            num_waves=5,
            border_width=0.5,
        ),
        "Stepping": terrain_gen.HfSteppingStonesTerrainCfg(
            stone_height_max=0,
            stone_width_range=(0.25,0.8),
            stone_distance_range=(0.01,0.5),
            holes_depth=-10,
            platform_width=2.,
            border_width=0.5,
        ),
        "Flat": terrain_gen.trimesh.MeshPlaneTerrainCfg(),
        "MeshStairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            step_height_range=(0.1, 0.5),
            step_width=0.3,
            holes=True,
            platform_width=1.5,
            border_width=0.5,
        ),
        "MeshStairsInverted": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            step_height_range=(0.1, 0.5),
            step_width=0.3,
            holes=True,
            platform_width=1.5,
            border_width=0.5,
        ),
        "MeshRandom": terrain_gen.MeshRandomGridTerrainCfg(
            grid_width=0.3,
            grid_height_range=(0.05, 0.5),
            holes=True,
            platform_width=2.,
        ),
        "Rails": terrain_gen.MeshRailsTerrainCfg(
            rail_thickness_range=(0.5,0.1),
            rail_height_range=(0.5,0.1),
            platform_width=2.,
        ),
        "Pit": terrain_gen.MeshPitTerrainCfg(
            pit_depth_range=(0.1,0.5),
            double_pit=True,
            platform_width=2.,
        ),
        "Box": terrain_gen.MeshBoxTerrainCfg(
            box_height_range=(0.1,0.5),
            double_box=True,
            platform_width=2.,
        ),
        "Gap": terrain_gen.MeshGapTerrainCfg(
            gap_width_range=(0.1,1.),
            platform_width=2.,
        ),
        "Ring": terrain_gen.MeshFloatingRingTerrainCfg(
            ring_width_range=(0.1,1.),
            ring_height_range=(0.1,1.),
            ring_thickness=0.1,
            platform_width=2.,
        ),
        "Star": terrain_gen.MeshStarTerrainCfg(
            num_bars=10,
            bar_width_range=(0.1,0.7),
            bar_height_range=(0.1,1.),
            platform_width=2.,
        ),
        "RepeatedBoxes": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            rel_height_noise=(0.8,1.2),
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=30,
                height=0.1,
                size=(0.6,0.6),
                max_yx_angle=0.,
                degrees=True
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=300,
                height=0.3,
                size=(0.2,0.2),
                max_yx_angle=30.,
                degrees=True,
            ),
            platform_width=2.0,
            platform_height=0.0,
        ),
        "RepeatedCylinder": terrain_gen.MeshRepeatedCylindersTerrainCfg(
            rel_height_noise=(0.8,1.2),
            object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=30,
                height=0.1,
                radius=0.6,
                max_yx_angle=0.,
                degrees=True
            ),
            object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=300,
                height=0.3,
                radius=0.2,
                max_yx_angle=30.,
                degrees=True,
            ),
            platform_width=2.0,
            platform_height=0.0,
        ),
        "RepeatedPyramids": terrain_gen.MeshRepeatedPyramidsTerrainCfg(
            rel_height_noise=(0.8,1.2),
            object_params_start=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=30,
                height=0.1,
                radius=0.6,
                max_yx_angle=0.,
                degrees=True
            ),
            object_params_end=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=300,
                height=0.3,
                radius=0.2,
                max_yx_angle=30.,
                degrees=True,
            ),
            platform_width=2.0,
            platform_height=0.0,
        ),
    },
)
"""All terrains configuration."""
