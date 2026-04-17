"""Terrain Viewer for CMoE terrains.

Usage (run from your IsaacLab root directory):
    python source/cmoe/cmoe/tasks/velocity/cmoe_1/terrains/view_terrains.py

This script generates all 9 terrain sub-types from the CMoE paper and displays
them in Isaac Sim for visual inspection. No robot is spawned.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View CMoE terrains in Isaac Sim")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- imports after AppLauncher ----

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.terrain_importer import TerrainImporter
from isaaclab.utils import configclass

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


@configclass
class CMoETerrainViewerCfg(TerrainGeneratorCfg):
    """Generate all 9 CMoE terrain types in a grid for visual inspection."""

    size: tuple[float, float] = (8.0, 8.0)
    num_rows: int = 10
    num_cols: int = 9
    curriculum: bool = True
    difficulty_range: tuple[float, float] = (0.0, 1.0)
    border_width: float = 0.0
    border_height: float = -1.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float = 0.75
    color_scheme: str = "height"
    use_cache: bool = False
    seed: int = 42

    sub_terrains: dict = {
        "slope_up": HfPyramidSlopedTerrainCfg(proportion=1.0 / 9.0),
        "slope_down": HfInvertedPyramidSlopedTerrainCfg(proportion=1.0 / 9.0),
        "stair_up": MeshPyramidStairsTerrainCfg(proportion=1.0 / 9.0),
        "stair_down": MeshInvertedPyramidStairsTerrainCfg(proportion=1.0 / 9.0),
        "gap": MeshGapTerrainCfg(proportion=1.0 / 9.0),
        "hurdle": MeshRepeatedBoxesTerrainCfg(proportion=1.0 / 9.0),
        "discrete": HfDiscreteObstaclesTerrainCfg(proportion=1.0 / 9.0),
        "mix1": MeshMix1TerrainCfg(proportion=1.0 / 9.0),
        "mix2": MeshMix2TerrainCfg(proportion=1.0 / 9.0),
    }


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[40.0, 40.0, 30.0], target=[0.0, 0.0, 0.0])

    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
    cfg.func("/World/Light", cfg)

    terrain_importer_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=CMoETerrainViewerCfg(),
        debug_vis=False,
    )
    terrain_importer = TerrainImporter(terrain_importer_cfg)

    sim.reset()

    print("\n" + "=" * 60)
    print("CMoE Terrain Viewer")
    print("=" * 60)
    print("\nColumn layout (left to right, y direction):")
    print("  0: Slope Up    1: Slope Down  2: Stair Up")
    print("  3: Stair Down  4: Gap         5: Hurdle")
    print("  6: Discrete    7: Mix1        8: Mix2")
    print("\nRow 0 (front) = easiest, Row 9 (back) = hardest")
    print("=" * 60 + "\n")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
