from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
import math
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
import source.cmoe.cmoe.tasks.velocity.cmoe.terrains as cmoe_1_terrains
from typing import Literal
import warnings

# 斜坡（Slope）：描述为坡度角，参数范围0-20°。
@configclass
class HfPyramidSlopedTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a pyramid sloped height field terrain."""

    function = cmoe_1_terrains.pyramid_sloped_terrain

    slope_range: tuple[float, float] = (0.0, math.tan(math.radians(20)))# (0.0, 0.364)
    """The slope of the terrain (in radians)."""

    platform_width: float = 0.5
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

    inverted: bool = False
    """Whether the pyramid is inverted. Defaults to False.

    If True, the terrain is inverted such that the platform is at the bottom and the slopes are upwards.
    """


@configclass
class HfInvertedPyramidSlopedTerrainCfg(HfPyramidSlopedTerrainCfg):
    """Configuration for an inverted pyramid sloped height field terrain.

    Note:
        This is a subclass of :class:`HfPyramidSlopedTerrainCfg` with :obj:`inverted` set to True.
        We make it as a separate class to make it easier to distinguish between the two and match
        the naming convention of the other terrains.
    """

    inverted: bool = True

# 上台阶（Stair up）：描述为台阶高度，参数范围0.05m-0.23m。
@configclass
class MeshPyramidStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a pyramid stair mesh terrain."""

    function = cmoe_1_terrains.pyramid_stairs_terrain

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is a flat terrain with the same height as the terrain.
    """

    step_height_range: tuple[float, float] = (0.05,0.23)
    """The minimum and maximum height of the steps (in m)."""

    step_width: float = 0.3
    """The width of the steps (in m)."""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

    holes: bool = False
    """If True, the terrain will have holes in the steps. Defaults to False.

    If :obj:`holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.
    """

# 下台阶（Stair down）：描述为台阶高度，参数范围0.05m-0.23m。
@configclass
class MeshInvertedPyramidStairsTerrainCfg(MeshPyramidStairsTerrainCfg):
    """Configuration for an inverted pyramid stair mesh terrain.

    Note:
        This is the same as :class:`MeshPyramidStairsTerrainCfg` except that the steps are inverted.
    """

    function = cmoe_1_terrains.inverted_pyramid_stairs_terrain

# 沟壑（Gap）：描述为沟体宽度，参数范围0.1m-0.8m。(isaaclab环形沟，论文横向沟渠)
@configclass
class MeshGapTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a gap around the platform."""

    function = cmoe_1_terrains.gap_terrain

    gap_width_range: tuple[float, float] = (0.1,0.8)
    """The minimum and maximum width of the gap (in m)."""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

# 障碍（Hurdle）：包含两项参数，分别是障碍高度，范围0.2m-0.4m；以及障碍宽度，范围0.1m-0.3m。
@configclass
class MeshRepeatedObjectsTerrainCfg(SubTerrainBaseCfg):
    """Base configuration for a terrain with repeated objects."""

    @configclass
    class ObjectCfg:
        """Configuration of repeated objects."""

        num_objects: int = MISSING
        """The number of objects to add to the terrain."""
        height: float = MISSING
        """The height (along z) of the object (in m)."""

    function = cmoe_1_terrains.repeated_objects_terrain

    object_type: Literal["cylinder", "box", "cone"] | callable = "box"
    """The type of object to generate.

    The type can be a string or a callable. If it is a string, the function will look for a function called
    ``make_{object_type}`` in the current module scope. If it is a callable, the function will
    use the callable to generate the object.
    """

    object_params_start: ObjectCfg = MISSING
    """The object curriculum parameters at the start of the curriculum."""

    object_params_end: ObjectCfg = MISSING
    """The object curriculum parameters at the end of the curriculum."""

    max_height_noise: float | None = None
    """"This parameter is deprecated, but stated here to support backward compatibility"""

    abs_height_noise: tuple[float, float] = (0.0, 0.0)
    """The minimum and maximum amount of additive noise for the height of the objects. Default is set to 0.0,
    which is no noise.
    """

    rel_height_noise: tuple[float, float] = (1.0, 1.0)
    """The minimum and maximum amount of multiplicative noise for the height of the objects. Default is set to 1.0,
    which is no noise.
    """

    platform_width: float = 1.0
    """The width of the cylindrical platform at the center of the terrain. Defaults to 1.0."""

    platform_height: float = -1.0
    """The height of the platform. Defaults to -1.0.

    If the value is negative, the height is the same as the object height.
    """

    def __post_init__(self):
        if self.max_height_noise is not None:
            warnings.warn(
                "MeshRepeatedObjectsTerrainCfg: max_height_noise:float is deprecated and support will be removed in the"
                " future. Use abs_height_noise:list[float] instead."
            )
            self.abs_height_noise = (-self.max_height_noise, self.max_height_noise)
@configclass
class MeshRepeatedBoxesTerrainCfg(MeshRepeatedObjectsTerrainCfg):
    """Configuration for a terrain with repeated boxes."""

    @configclass
    class ObjectCfg(MeshRepeatedObjectsTerrainCfg.ObjectCfg):
        """Configuration for repeated boxes."""

        size: tuple[float, float] = MISSING
        """The width (along x) and length (along y) of the box (in m)."""
        max_yx_angle: float = 0.0
        """The maximum angle along the y and x axis. Defaults to 0.0."""
        degrees: bool = True
        """Whether the angle is in degrees. Defaults to True."""

    object_type = cmoe_1_terrains.make_box

    object_params_start: ObjectCfg = ObjectCfg(
        num_objects=3,
        height=0.2,
        size=(0.1, 8.0),  # 横跨整个8m子地形宽度
        max_yx_angle=0.0,
        degrees=True,
    )

    object_params_end: ObjectCfg = ObjectCfg(
        num_objects=5,
        height=0.4,
        size=(0.3, 8.0),  # 横跨整个8m子地形宽度
        max_yx_angle=0.0,
        degrees=True,
    )
    """The box curriculum parameters at the end of the curriculum."""

# 离散凸起地形（Discrete）：描述为不规则凸起高度，参数范围0.1m-0.2m。
@configclass
class HfDiscreteObstaclesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a discrete obstacles height field terrain."""

    function = cmoe_1_terrains.discrete_obstacles_terrain

    obstacle_height_mode: str = "fixed"
    """The mode to use for the obstacle height. Defaults to "choice".

    The following modes are supported: "choice", "fixed".
    """

    obstacle_width_range: tuple[float, float] = (0.25, 0.75)
    """The minimum and maximum width of the obstacles (in m)."""

    obstacle_height_range: tuple[float, float] = (0.1,0.2)
    """The minimum and maximum height of the obstacles (in m)."""

    num_obstacles: int = 50
    """The number of obstacles to generate."""

    platform_width: float = 1.0
    """The width of the square platform at the center of the terrain. Defaults to 1.0."""

# 混合地形1（Mix1）：为沟壑与台阶的混合地形，包含两项参数，分别是沟壑宽度，范围0.1m-0.8m；以及台阶高度，范围0.1m-0.15m。
@configclass
class MeshMix1TerrainCfg(SubTerrainBaseCfg):
    """Configuration for Mix1: mixed terrain of gaps and stairs."""

    function = cmoe_1_terrains.mix1_gap_stairs_terrain

    gap_width_range: tuple[float, float] = (0.1, 0.8)
    """The minimum and maximum width of the gaps (in m). Paper Table I: 0.1-0.8m"""

    step_height_range: tuple[float, float] = (0.1, 0.15)
    """The minimum and maximum height of the steps (in m). Paper Table I: 0.1-0.15m"""

    step_width: float = 0.3
    """The depth of each step along the walking direction (in m)."""

    start_platform_length: float = 1.0
    """The length of the starting safe zone along the walking direction (in m)."""

    num_steps_per_section: int = 3
    """The number of steps in each stair section (up then down)."""

    num_sections: int = 5
    """The total number of alternating sections (odd=stairs, even=gaps)."""

# 混合地形2（Mix2）：为独木桥与台阶的混合地形，包含两项参数，分别是桥体宽度，范围0.5m-1.0m；以及桥上的台阶高度，范围0.1m-0.25m。
@configclass
class MeshMix2TerrainCfg(SubTerrainBaseCfg):
    """Configuration for Mix2: mixed terrain of single-log bridge and stairs."""

    function = cmoe_1_terrains.mix2_bridge_stairs_terrain

    bridge_width_range: tuple[float, float] = (0.5, 1.0)
    """The minimum and maximum width of the bridge (in m). Paper Table I: 0.5-1.0m.
    Note: narrower bridge = harder, so at difficulty=1 width=0.5m, difficulty=0 width=1.0m."""

    step_height_range: tuple[float, float] = (0.1, 0.25)
    """The minimum and maximum height of the steps on the bridge (in m). Paper Table I: 0.1-0.25m"""

    step_width: float = 0.3
    """The depth of each step along the walking direction (in m)."""

    platform_width: float = 1.5
    """The length of the starting/ending safe zone along the walking direction (in m)."""

    num_steps: int = 4
    """The number of steps going up (same number going down)."""