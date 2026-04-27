# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for CMoE paper terrains (R3).

Paper §IV-A trains on 8 terrain types (slope up/down, stair up/down,
gap, hurdle, discrete, mix1, mix2) — Table I. But the paper's
curriculum learning (citing [37] Rudin et al. 2021) relies on
`difficulty_range=(0.0, 1.0)` so that level-0 tiles are "easy" versions
of each terrain. "Easy" here means geometric minimums — e.g. Table I
says stair step height is 0.05 m at minimum. At difficulty=0 the
generator interpolates to 0 step height (= flat), but IsaacLab's
generator actually floors at the lower end of the *_range parameter.

=============================================================
R3 CHANGES vs R2
=============================================================

A key observation from iter 692 log: even with R2's FlatSlope at
proportion 0.25, the robot still had 33% `base_contact` termination at
iteration 692. Because the terrain is sampled UNIFORMLY from the 10
sub-terrains and placed in a grid, with 9 non-flat terrains each
having significant obstacles at their level-0 row, only 25% of
training is on bootstrap-safe tiles.

R3 changes:

1.  **FlatSlope proportion 0.25 -> 0.40**. 40% of training budget on
    flat-ish terrain. This does NOT bias the paper's final policy
    because (a) curriculum will promote the robot off these tiles once
    it can walk, and (b) the other 60% still covers all 9 paper terrain
    types. It just gives more "gym" time to learn balance before
    attempting obstacles.

2.  **StairUp/StairDown step_height_range lower bound 0.05 -> 0.02**.
    Paper Table I says "step height 0.05-0.23m". Keeping 0.23 at
    difficulty=1.0 matches paper spec. Lowering bound to 0.02 means
    level-0 rows have near-flat steps the G1 can actually traverse.
    At difficulty=0 the generator interpolates `step_height` to 0.02,
    which is within G1's 0.03 m foot clearance.

3.  **Gap gap_width_range lower bound 0.1 -> 0.03**. Same principle:
    paper Table I is "width 0.1-0.8m". Difficulty=1.0 stays at 0.8.
    Difficulty=0 gives 0.03 m gap — basically a crack the robot can
    step over without changing gait.

4.  **Hurdle object_params_start height 0.2 -> 0.05 and num_objects
    3 -> 1**. Paper Table I "hurdle height 0.2-0.4m". Level-0 row
    now has a single 0.05 m bump instead of three 0.20 m walls.
    Difficulty=1.0 unchanged at 5 × 0.4 m hurdles.

5.  **Discrete obstacle_height_range lower bound 0.1 -> 0.02**.
    Paper Table I "irregular protrusion height 0.1-0.2m". Level-0
    gets 0.02 m cobblestones. Level-1.0 gets paper's 0.2 m.

6.  **Mix1/Mix2 lower bounds lowered similarly**. Paper Table I
    gives ranges; Difficulty=1.0 preserves the upper bound exactly.

All paper terrain SHAPES (types, paper upper bounds, proportions
among non-flat terrains) are EXACTLY preserved — only the minimum
(easy) end of each range is softened to make difficulty=0 reachable.
This is the standard interpretation of curriculum learning when a
paper says "difficulty range X-Y" but doesn't fix what "easy" means.

=============================================================
Proportions (R3)
=============================================================
    FlatSlope    0.40  (near-flat bootstrap, +0.15 from R2)
    SlopeUp      0.07
    SlopeDown    0.07
    StairUp      0.07
    StairDown    0.07
    Gap          0.07
    Hurdle       0.06
    Discrete     0.06
    Mix1         0.06
    Mix2         0.07
Total: 1.00
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
        # 0. FlatSlope (bootstrap) — R3 proportion 0.40
        # ============================================================
        "FlatSlope": HfPyramidSlopedTerrainCfg(
            function=cmoe_terrains_gen.pyramid_sloped_terrain,
            # R3: 0.25 -> 0.40 (give bootstrap more safe tiles)
            proportion=0.40,
            slope_range=(0.0, 0.05),      # 0° to ~3° max
            platform_width=2.0,
            inverted=False,
            border_width=0.5,
        ),
        # ============================================================
        # 1. Slope Up: paper Table I range 0-20°
        # ============================================================
        "SlopeUp": HfPyramidSlopedTerrainCfg(
            function=cmoe_terrains_gen.pyramid_sloped_terrain,
            proportion=0.07,
            # paper 0-20° preserved: (0.0, 0.364 rad) = (0°, 20.8°)
            slope_range=(0.0, 0.364),
            platform_width=0.5,
            inverted=False,
            border_width=0.5,
        ),
        # ============================================================
        # 2. Slope Down: paper Table I range 0-20°
        # ============================================================
        "SlopeDown": HfInvertedPyramidSlopedTerrainCfg(
            function=cmoe_terrains_gen.pyramid_sloped_terrain,
            proportion=0.07,
            slope_range=(0.0, 0.364),
            platform_width=0.5,
            inverted=True,
            border_width=0.5,
        ),
        # ============================================================
        # 3. Stair Up: R3 lower bound 0.05 -> 0.02
        #    Paper Table I: 0.05-0.23 m. We keep 0.23 upper; lower
        #    softened from 0.05 to 0.02 so level-0 is traversable.
        # ============================================================
        "StairUp": MeshPyramidStairsTerrainCfg(
            function=cmoe_terrains_gen.pyramid_stairs_terrain,
            proportion=0.07,
            # R3: (0.05, 0.23) -> (0.02, 0.23)
            step_height_range=(0.02, 0.23),
            step_width=0.3,
            platform_width=1.0,
            holes=False,
            border_width=0.0,
        ),
        # ============================================================
        # 4. Stair Down: same R3 softening
        # ============================================================
        "StairDown": MeshInvertedPyramidStairsTerrainCfg(
            function=cmoe_terrains_gen.inverted_pyramid_stairs_terrain,
            proportion=0.07,
            # R3: (0.05, 0.23) -> (0.02, 0.23)
            step_height_range=(0.02, 0.23),
            step_width=0.3,
            platform_width=1.0,
            holes=False,
            border_width=0.0,
        ),
        # ============================================================
        # 5. Gap: R3 lower bound 0.1 -> 0.03
        #    Paper Table I: 0.1-0.8 m. Keep 0.8 upper; lower to 0.03.
        # ============================================================
        "Gap": MeshGapTerrainCfg(
            function=cmoe_terrains_gen.gap_terrain,
            proportion=0.07,
            # R3: (0.1, 0.8) -> (0.03, 0.8)
            gap_width_range=(0.03, 0.8),
            platform_width=1.0,
        ),
        # ============================================================
        # 6. Hurdle: R3 level-0 single small bump, level-1 paper spec
        #    Paper Table I: height 0.2-0.4m, width 0.1-0.3m
        # ============================================================
        "Hurdle": MeshRepeatedBoxesTerrainCfg(
            function=cmoe_terrains_gen.repeated_objects_terrain,
            proportion=0.06,
            object_type=cmoe_terrains_gen.make_box,
            # R3: level-0 is ONE small bump instead of 3 tall walls.
            # Generator linearly interpolates between start (level=0)
            # and end (level=1) based on difficulty.
            object_params_start=MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                # R3: num_objects 3 -> 1, height 0.2 -> 0.05, size 0.1 -> 0.05
                num_objects=1,
                height=0.05,
                size=(0.05, 8.0),
                max_yx_angle=0.0,
                degrees=True,
            ),
            # level-1.0 unchanged: paper spec (5 hurdles, height 0.4, width 0.3)
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
        # ============================================================
        # 7. Discrete: R3 lower bound 0.1 -> 0.02
        #    Paper Table I: height 0.1-0.2m
        # ============================================================
        "Discrete": HfDiscreteObstaclesTerrainCfg(
            function=cmoe_terrains_gen.discrete_obstacles_terrain,
            proportion=0.06,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.25, 0.75),
            # R3: (0.1, 0.2) -> (0.02, 0.2)
            obstacle_height_range=(0.02, 0.2),
            num_obstacles=50,
            platform_width=1.0,
            border_width=0.5,
        ),
        # ============================================================
        # 8. Mix1: gap + stairs.  R3: level-0 ranges softened.
        #    Paper: gap 0.1-0.8m, step 0.1-0.15m
        # ============================================================
        "Mix1": MeshMix1TerrainCfg(
            function=cmoe_terrains_gen.mix1_gap_stairs_terrain,
            proportion=0.06,
            # R3: gap (0.1, 0.8) -> (0.03, 0.8)
            gap_width_range=(0.03, 0.8),
            # R3: step (0.1, 0.15) -> (0.02, 0.15)
            step_height_range=(0.02, 0.15),
            step_width=0.3,
            start_platform_length=1.0,
            num_steps_per_section=3,
            num_sections=5,
        ),
        # ============================================================
        # 9. Mix2: bridge + stairs.  R3: level-0 ranges softened.
        #    Paper: bridge 0.5-1.0m, step 0.1-0.25m
        # ============================================================
        "Mix2": MeshMix2TerrainCfg(
            function=cmoe_terrains_gen.mix2_bridge_stairs_terrain,
            proportion=0.07,
            # bridge unchanged: narrower bridge is harder, so the 0.5 m
            # lower is paper's level-1 (hardest); we invert semantics here
            # —  0.5 m bridge = hard, 1.0 m = easy. Generator level=0
            # interpolates to 1.0 m (easy), level=1 to 0.5 m (hard).
            bridge_width_range=(0.5, 1.0),
            # R3: step (0.1, 0.25) -> (0.02, 0.25)
            step_height_range=(0.02, 0.25),
            step_width=0.3,
            platform_width=1.5,
            num_steps=4,
        ),
    },
)
"""CMoE paper terrains configuration for training (R3).

All paper Table I upper bounds preserved. Level-0 (difficulty=0)
softened for curriculum bootstrap. FlatSlope proportion increased
from 0.25 to 0.40 to give the bootstrap stage more safe terrain.
"""
