# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CMoE paper terrains — PAPER-STRICT (R-FINAL).

============================================================================
设计原则: 严格按 paper Table I 实现 8 种地形.
============================================================================

Paper Table I:
  Slope:    angle 0-20°
  Stairs:   step height 0.05-0.23 m
  Gap:      ditch width 0.1-0.8 m
  Hurdle:   height 0.2-0.4 m, width 0.1-0.3 m
  Discrete: irregular protrusion height 0.1-0.2 m
  Mix1:     gaps 0.1-0.8 m + steps 0.1-0.15 m
  Mix2:     bridge 0.5-1.0 m + step on bridge 0.1-0.25 m

R-FINAL 移除了所有非论文的修改:
  - 移除 FlatSlope bootstrap (paper 没有此地形)
  - 移除 step_height min 软化 (paper 是 0.05 不是 0.02)
  - 移除 gap min 软化 (paper 是 0.1 不是 0.03)
  - 移除 hurdle level-0 单 0.05 m bump (paper 是 0.2 m)
  - 移除 mix1/mix2 step min 软化

8 种地形等比例分配 (paper §IV-A "8 types of terrain"):
  SlopeUp, SlopeDown, StairUp, StairDown, Gap, Hurdle, Discrete, Mix1, Mix2
  实际是 9 种 sub-terrain (paper 把 SlopeUp/Down 算作 1 种 "Slope")

Curriculum:
  num_rows=10 (10 difficulty levels)
  num_cols=9 (每种地形 9 列, 但只用 8 类比例)
  difficulty_range=(0.0, 1.0)
============================================================================
"""

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


CMOE_TERRAINS_CFG = TerrainGeneratorCfg(
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
        # ============================================================
        # 1. Slope Up: paper Table I "Slope angle 0-20°"
        # ============================================================
        "SlopeUp": HfPyramidSlopedTerrainCfg(
            function=cmoe_terrains_gen.pyramid_sloped_terrain,
            proportion=1.0 / 9,
            slope_range=(0.0, 0.364),  # 0° to 20.85° (paper Table I 0-20°)
            platform_width=0.5,
            inverted=False,
            border_width=0.5,
        ),
        # ============================================================
        # 2. Slope Down: paper Table I "Slope angle 0-20°"
        # ============================================================
        "SlopeDown": HfInvertedPyramidSlopedTerrainCfg(
            function=cmoe_terrains_gen.pyramid_sloped_terrain,
            proportion=1.0 / 9,
            slope_range=(0.0, 0.364),
            platform_width=0.5,
            inverted=True,
            border_width=0.5,
        ),
        # ============================================================
        # 3. Stair Up: paper Table I "Step height 0.05-0.23 m"
        # ============================================================
        "StairUp": MeshPyramidStairsTerrainCfg(
            function=cmoe_terrains_gen.pyramid_stairs_terrain,
            proportion=1.0 / 9,
            step_height_range=(0.05, 0.23),  # paper Table I EXACT
            step_width=0.3,
            platform_width=1.0,
            holes=False,
            border_width=0.0,
        ),
        # ============================================================
        # 4. Stair Down: paper Table I "Step height 0.05-0.23 m"
        # ============================================================
        "StairDown": MeshInvertedPyramidStairsTerrainCfg(
            function=cmoe_terrains_gen.inverted_pyramid_stairs_terrain,
            proportion=1.0 / 9,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=1.0,
            holes=False,
            border_width=0.0,
        ),
        # ============================================================
        # 5. Gap: paper Table I "Ditch width 0.1-0.8 m"
        # ============================================================
        "Gap": MeshGapTerrainCfg(
            function=cmoe_terrains_gen.gap_terrain,
            proportion=1.0 / 9,
            gap_width_range=(0.1, 0.8),  # paper Table I EXACT
            platform_width=1.0,
        ),
        # ============================================================
        # 6. Hurdle: paper Table I "height 0.2-0.4 m, width 0.1-0.3 m"
        # ============================================================
        "Hurdle": MeshRepeatedBoxesTerrainCfg(
            function=cmoe_terrains_gen.repeated_objects_terrain,
            proportion=1.0 / 9,
            object_type=cmoe_terrains_gen.make_box,
            object_params_start=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=1,
                height=0.2,    # paper Table I 下界
                size=(0.1, 8.0),  # paper width 下界 0.1
                max_yx_angle=0.0,
                degrees=True,
            ),
            object_params_end=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=5,
                height=0.4,    # paper Table I 上界
                size=(0.3, 8.0),  # paper width 上界 0.3
                max_yx_angle=0.0,
                degrees=True,
            ),
            platform_width=1.0,
            platform_height=-1.0,
        ),
        # ============================================================
        # 7. Discrete: paper Table I "Irregular protrusion height 0.1-0.2 m"
        # ============================================================
        "Discrete": HfDiscreteObstaclesTerrainCfg(
            function=cmoe_terrains_gen.discrete_obstacles_terrain,
            proportion=1.0 / 9,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.25, 0.75),
            obstacle_height_range=(0.1, 0.2),  # paper Table I EXACT
            num_obstacles=50,
            platform_width=1.0,
            border_width=0.5,
        ),
        # ============================================================
        # 8. Mix1: paper Table I "gaps 0.1-0.8 + steps 0.1-0.15"
        # ============================================================
        "Mix1": MeshMix1TerrainCfg(
            function=cmoe_terrains_gen.mix1_gap_stairs_terrain,
            proportion=1.0 / 9,
            gap_width_range=(0.1, 0.8),       # paper Table I EXACT
            step_height_range=(0.1, 0.15),    # paper Table I EXACT
            step_width=0.3,
            start_platform_length=1.0,
            num_steps_per_section=3,
            num_sections=5,
        ),
        # ============================================================
        # 9. Mix2: paper Table I "bridge 0.5-1.0 + step 0.1-0.25"
        # ============================================================
        "Mix2": MeshMix2TerrainCfg(
            function=cmoe_terrains_gen.mix2_bridge_stairs_terrain,
            proportion=1.0 / 9,
            bridge_width_range=(0.5, 1.0),    # paper Table I EXACT
            step_height_range=(0.1, 0.25),    # paper Table I EXACT
            step_width=0.3,
            platform_width=1.5,
            num_steps=4,
        ),
    },
)
"""CMoE paper terrains - 严格论文 Table I 实现.

总比例: 9 × (1/9) = 1.0  (paper §IV-A "8 types of terrain", SlopeUp/Down
分两个所以实际 9 个 sub-terrain).

Difficulty curriculum:
  - num_rows=10 (10 difficulty levels)
  - difficulty_range=(0.0, 1.0)
  - level k 的难度 ≈ k/9, 自动插值在 [min, max] 内
  - level 0 = 最简单 (0° slope, 0.05 m step, 0.1 m gap, 0.2 m hurdle...)
  - level 9 = 最难 (paper Table I max values)
"""
