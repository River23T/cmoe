from __future__ import annotations
import numpy as np
import trimesh
from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403
from isaaclab.terrains.trimesh.utils import make_border, make_plane
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import cmoe_1_terrains_cfg

# 斜坡（Slope）：描述为坡度角，参数范围0-20°。
@height_field_to_mesh
def pyramid_sloped_terrain(difficulty: float, cfg: cmoe_1_terrains_cfg.HfPyramidSlopedTerrainCfg) -> np.ndarray:
    """Generate a terrain with a truncated pyramid structure.

    The terrain is a pyramid-shaped sloped surface with a slope of :obj:`slope` that trims into a flat platform
    at the center. The slope is defined as the ratio of the height change along the x axis to the width along the
    x axis. For example, a slope of 1.0 means that the height changes by 1 unit for every 1 unit of width.

    If the :obj:`cfg.inverted` flag is set to :obj:`True`, the terrain is inverted such that
    the platform is at the bottom.

    .. image:: ../../_static/terrains/height_field/pyramid_sloped_terrain.jpg
       :width: 40%

    .. image:: ../../_static/terrains/height_field/inverted_pyramid_sloped_terrain.jpg
       :width: 40%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    if cfg.inverted:
        slope = -cfg.slope_range[0] - difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    else:
        slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- height
    # we want the height to be 1/2 of the width since the terrain is a pyramid
    height_max = int(slope * cfg.size[0] / 2 / cfg.vertical_scale)
    # -- center of the terrain
    center_x = int(width_pixels / 2)
    center_y = int(length_pixels / 2)

    # create a meshgrid of the terrain
    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    # offset the meshgrid to the center of the terrain
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    # reshape the meshgrid to be 2D
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)
    # create a sloped surface
    hf_raw = np.zeros((width_pixels, length_pixels))
    hf_raw = height_max * xx * yy

    # create a flat platform at the center of the terrain
    platform_width = int(cfg.platform_width / cfg.horizontal_scale / 2)
    # get the height of the platform at the corner of the platform
    x_pf = width_pixels // 2 - platform_width
    y_pf = length_pixels // 2 - platform_width
    z_pf = hf_raw[x_pf, y_pf]
    hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)

# 上台阶（Stair up）：描述为台阶高度，参数范围0.05m-0.23m。
def pyramid_stairs_terrain(
    difficulty: float, cfg: cmoe_1_terrains_cfg.MeshPyramidStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a pyramid stair pattern.

    The terrain is a pyramid stair pattern which trims to a flat platform at the center of the terrain.

    If :obj:`cfg.holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`cfg.platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.

    .. image:: ../../_static/terrains/trimesh/pyramid_stairs_terrain.jpg
       :width: 45%

    .. image:: ../../_static/terrains/trimesh/pyramid_stairs_terrain_with_holes.jpg
       :width: 45%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])

    # compute number of steps in x and y direction
    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    # we take the minimum number of steps in x and y direction
    num_steps = int(min(num_steps_x, num_steps_y))

    # initialize list of meshes
    meshes_list = list()

    # generate the border if needed
    if cfg.border_width > 0.0 and not cfg.holes:
        # obtain a list of meshes for the border
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_height / 2]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders

    # generate the terrain
    # -- compute the position of the center of the terrain
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    # -- generate the stair pattern
    for k in range(num_steps):
        # check if we need to add holes around the steps
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * cfg.step_width, terrain_size[1] - 2 * k * cfg.step_width)
        # compute the quantities of the box
        # -- location
        box_z = terrain_center[2] + k * step_height / 2.0
        box_offset = (k + 0.5) * cfg.step_width
        # -- dimensions
        box_height = (k + 2) * step_height
        # generate the boxes
        # top/bottom
        box_dims = (box_size[0], cfg.step_width, box_height)
        # -- top
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- bottom
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # right/left
        if cfg.holes:
            box_dims = (cfg.step_width, box_size[1], box_height)
        else:
            box_dims = (cfg.step_width, box_size[1] - 2 * cfg.step_width, box_height)
        # -- right
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- left
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # add the boxes to the list of meshes
        meshes_list += [box_top, box_bottom, box_right, box_left]

    # generate final box for the middle of the terrain
    box_dims = (
        terrain_size[0] - 2 * num_steps * cfg.step_width,
        terrain_size[1] - 2 * num_steps * cfg.step_width,
        (num_steps + 2) * step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] + num_steps * step_height / 2)
    box_middle = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    meshes_list.append(box_middle)
    # origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], (num_steps + 1) * step_height])

    return meshes_list, origin

# 下台阶（Stair down）：描述为台阶高度，参数范围0.05m-0.23m。
def inverted_pyramid_stairs_terrain(
    difficulty: float, cfg: cmoe_1_terrains_cfg.MeshInvertedPyramidStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a inverted pyramid stair pattern.

    The terrain is an inverted pyramid stair pattern which trims to a flat platform at the center of the terrain.

    If :obj:`cfg.holes` is True, the terrain will have pyramid stairs of length or width
    :obj:`cfg.platform_width` (depending on the direction) with no steps in the remaining area. Additionally,
    no border will be added.

    .. image:: ../../_static/terrains/trimesh/inverted_pyramid_stairs_terrain.jpg
       :width: 45%

    .. image:: ../../_static/terrains/trimesh/inverted_pyramid_stairs_terrain_with_holes.jpg
       :width: 45%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])

    # compute number of steps in x and y direction
    num_steps_x = (cfg.size[0] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    num_steps_y = (cfg.size[1] - 2 * cfg.border_width - cfg.platform_width) // (2 * cfg.step_width) + 1
    # we take the minimum number of steps in x and y direction
    num_steps = int(min(num_steps_x, num_steps_y))
    # total height of the terrain
    total_height = (num_steps + 1) * step_height

    # initialize list of meshes
    meshes_list = list()

    # generate the border if needed
    if cfg.border_width > 0.0 and not cfg.holes:
        # obtain a list of meshes for the border
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * step_height]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, step_height, border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders
    # generate the terrain
    # -- compute the position of the center of the terrain
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    terrain_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    # -- generate the stair pattern
    for k in range(num_steps):
        # check if we need to add holes around the steps
        if cfg.holes:
            box_size = (cfg.platform_width, cfg.platform_width)
        else:
            box_size = (terrain_size[0] - 2 * k * cfg.step_width, terrain_size[1] - 2 * k * cfg.step_width)
        # compute the quantities of the box
        # -- location
        box_z = terrain_center[2] - total_height / 2 - (k + 1) * step_height / 2.0
        box_offset = (k + 0.5) * cfg.step_width
        # -- dimensions
        box_height = total_height - (k + 1) * step_height
        # generate the boxes
        # top/bottom
        box_dims = (box_size[0], cfg.step_width, box_height)
        # -- top
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- bottom
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # right/left
        if cfg.holes:
            box_dims = (cfg.step_width, box_size[1], box_height)
        else:
            box_dims = (cfg.step_width, box_size[1] - 2 * cfg.step_width, box_height)
        # -- right
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- left
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # add the boxes to the list of meshes
        meshes_list += [box_top, box_bottom, box_right, box_left]
    # generate final box for the middle of the terrain
    box_dims = (
        terrain_size[0] - 2 * num_steps * cfg.step_width,
        terrain_size[1] - 2 * num_steps * cfg.step_width,
        step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] - total_height - step_height / 2)
    box_middle = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    meshes_list.append(box_middle)
    # origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], -(num_steps + 1) * step_height])

    return meshes_list, origin

# 沟壑（Gap）：描述为沟体宽度，参数范围0.1m-0.8m。
def gap_terrain(
    difficulty: float, cfg: cmoe_1_terrains_cfg.MeshGapTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a gap around the platform.

    The terrain has a ground with a platform in the middle. The platform is surrounded by a gap
    of width :obj:`gap_width` on all sides.

    .. image:: ../../_static/terrains/trimesh/gap_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])

    # initialize list of meshes
    meshes_list = list()
    # constants for terrain generation
    terrain_height = 1.0
    terrain_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)

    # Generate the outer ring
    inner_size = (cfg.platform_width + 2 * gap_width, cfg.platform_width + 2 * gap_width)
    meshes_list += make_border(cfg.size, inner_size, terrain_height, terrain_center)
    # Generate the inner box
    box_dim = (cfg.platform_width, cfg.platform_width, terrain_height)
    box = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(terrain_center))
    meshes_list.append(box)

    # specify the origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], 0.0])

    return meshes_list, origin

# 障碍（Hurdle）：包含两项参数，分别是障碍高度，范围0.2m-0.4m；以及障碍宽度，范围0.1m-0.3m。
def repeated_objects_terrain(
    difficulty: float, cfg: cmoe_1_terrains_cfg.MeshRepeatedObjectsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a set of repeated objects.

    The terrain has a ground with a platform in the middle. The objects are randomly placed on the
    terrain s.t. they do not overlap with the platform.

    Depending on the object type, the objects are generated with different parameters. The objects
    The types of objects that can be generated are: ``"cylinder"``, ``"box"``, ``"cone"``.

    The object parameters are specified in the configuration as curriculum parameters. The difficulty
    is used to linearly interpolate between the minimum and maximum values of the parameters.

    .. image:: ../../_static/terrains/trimesh/repeated_objects_cylinder_terrain.jpg
       :width: 30%

    .. image:: ../../_static/terrains/trimesh/repeated_objects_box_terrain.jpg
       :width: 30%

    .. image:: ../../_static/terrains/trimesh/repeated_objects_pyramid_terrain.jpg
       :width: 30%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).

    Raises:
        ValueError: If the object type is not supported. It must be either a string or a callable.
    """
    # import the object functions -- this is done here to avoid circular imports
    from . import cmoe_1_terrains_cfg as _cfg
    MeshRepeatedBoxesTerrainCfg = _cfg.MeshRepeatedBoxesTerrainCfg

    # if object type is a string, get the function: make_{object_type}
    if isinstance(cfg.object_type, str):
        object_func = globals().get(f"make_{cfg.object_type}")
    else:
        object_func = cfg.object_type
    if not callable(object_func):
        raise ValueError(f"The attribute 'object_type' must be a string or a callable. Received: {object_func}")

    # Resolve the terrain configuration
    # -- pass parameters to make calling simpler
    cp_0 = cfg.object_params_start
    cp_1 = cfg.object_params_end
    # -- common parameters
    num_objects = cp_0.num_objects + int(difficulty * (cp_1.num_objects - cp_0.num_objects))
    height = cp_0.height + difficulty * (cp_1.height - cp_0.height)
    platform_height = cfg.platform_height if cfg.platform_height >= 0.0 else height
    # -- object specific parameters
    # note: SIM114 requires duplicated logical blocks under a single body.
    if isinstance(cfg, MeshRepeatedBoxesTerrainCfg):
        cp_0: MeshRepeatedBoxesTerrainCfg.ObjectCfg
        cp_1: MeshRepeatedBoxesTerrainCfg.ObjectCfg
        object_kwargs = {
            "length": cp_0.size[0] + difficulty * (cp_1.size[0] - cp_0.size[0]),
            "width": cp_0.size[1] + difficulty * (cp_1.size[1] - cp_0.size[1]),
            "max_yx_angle": cp_0.max_yx_angle + difficulty * (cp_1.max_yx_angle - cp_0.max_yx_angle),
            "degrees": cp_0.degrees,
        }
    else:
        raise ValueError(f"Unknown terrain configuration: {cfg}")
    # constants for the terrain
    platform_clearance = 0.1

    # initialize list of meshes
    meshes_list = list()
    # compute quantities
    origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.5 * platform_height))
    platform_corners = np.asarray(
        [
            [origin[0] - cfg.platform_width / 2, origin[1] - cfg.platform_width / 2],
            [origin[0] + cfg.platform_width / 2, origin[1] + cfg.platform_width / 2],
        ]
    )
    platform_corners[0, :] *= 1 - platform_clearance
    platform_corners[1, :] *= 1 + platform_clearance
    # sample valid center for objects
    object_centers = np.zeros((num_objects, 3))
    # use a mask to track invalid objects that still require sampling
    mask_objects_left = np.ones((num_objects,), dtype=bool)
    # loop until no objects are left to sample
    while np.any(mask_objects_left):
        # only sample the centers of the remaining invalid objects
        num_objects_left = mask_objects_left.sum()
        object_centers[mask_objects_left, 0] = np.random.uniform(0, cfg.size[0], num_objects_left)
        object_centers[mask_objects_left, 1] = np.random.uniform(0, cfg.size[1], num_objects_left)
        # filter out the centers that are on the platform
        is_within_platform_x = np.logical_and(
            object_centers[mask_objects_left, 0] >= platform_corners[0, 0],
            object_centers[mask_objects_left, 0] <= platform_corners[1, 0],
        )
        is_within_platform_y = np.logical_and(
            object_centers[mask_objects_left, 1] >= platform_corners[0, 1],
            object_centers[mask_objects_left, 1] <= platform_corners[1, 1],
        )
        # update the mask to track the validity of the objects sampled in this iteration
        mask_objects_left[mask_objects_left] = np.logical_and(is_within_platform_x, is_within_platform_y)

    # generate obstacles (but keep platform clean)
    for index in range(len(object_centers)):
        # randomize the height of the object
        abs_height_noise = np.random.uniform(cfg.abs_height_noise[0], cfg.abs_height_noise[1])
        rel_height_noise = np.random.uniform(cfg.rel_height_noise[0], cfg.rel_height_noise[1])
        ob_height = height * rel_height_noise + abs_height_noise
        if ob_height > 0.0:
            object_mesh = object_func(center=object_centers[index], height=ob_height, **object_kwargs)
            meshes_list.append(object_mesh)

    # generate a ground plane for the terrain
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)
    # generate a platform in the middle
    dim = (cfg.platform_width, cfg.platform_width, 0.5 * platform_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.25 * platform_height)
    platform = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(platform)

    return meshes_list, origin

# 离散凸起地形（Discrete）：描述为不规则凸起高度，参数范围0.1m-0.2m。
@height_field_to_mesh
def discrete_obstacles_terrain(difficulty: float, cfg: cmoe_1_terrains_cfg.HfDiscreteObstaclesTerrainCfg) -> np.ndarray:
    """Generate a terrain with randomly generated obstacles as pillars with positive and negative heights.

    The terrain is a flat platform at the center of the terrain with randomly generated obstacles as pillars
    with positive and negative height. The obstacles are randomly generated cuboids with a random width and
    height. They are placed randomly on the terrain with a minimum distance of :obj:`cfg.platform_width`
    from the center of the terrain.

    .. image:: ../../_static/terrains/height_field/discrete_obstacles_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    obs_height = cfg.obstacle_height_range[0] + difficulty * (
        cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]
    )

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- obstacles
    obs_height = int(obs_height / cfg.vertical_scale)
    obs_width_min = int(cfg.obstacle_width_range[0] / cfg.horizontal_scale)
    obs_width_max = int(cfg.obstacle_width_range[1] / cfg.horizontal_scale)
    # -- center of the terrain
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    # create discrete ranges for the obstacles
    # -- shape
    obs_width_range = np.arange(obs_width_min, obs_width_max, 4)
    obs_length_range = np.arange(obs_width_min, obs_width_max, 4)
    # -- position
    obs_x_range = np.arange(0, width_pixels, 4)
    obs_y_range = np.arange(0, length_pixels, 4)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    # generate the obstacles
    for _ in range(cfg.num_obstacles):
        # sample size
        if cfg.obstacle_height_mode == "choice":
            height = np.random.choice([-obs_height, -obs_height // 2, obs_height // 2, obs_height])
        elif cfg.obstacle_height_mode == "fixed":
            height = obs_height
        else:
            raise ValueError(f"Unknown obstacle height mode '{cfg.obstacle_height_mode}'. Must be 'choice' or 'fixed'.")
        width = int(np.random.choice(obs_width_range))
        length = int(np.random.choice(obs_length_range))
        # sample position
        x_start = int(np.random.choice(obs_x_range))
        y_start = int(np.random.choice(obs_y_range))
        # clip start position to the terrain
        if x_start + width > width_pixels:
            x_start = width_pixels - width
        if y_start + length > length_pixels:
            y_start = length_pixels - length
        # add to terrain
        hf_raw[x_start : x_start + width, y_start : y_start + length] = height
    # clip the terrain to the platform
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)

# 混合地形1（Mix1）：为沟壑与台阶的混合地形，包含两项参数，分别是沟壑宽度，范围0.1m-0.8m；以及台阶高度，范围0.1m-0.15m。
def mix1_gap_stairs_terrain(
    difficulty: float, cfg: cmoe_1_terrains_cfg.MeshMix1TerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a mixed terrain with alternating gaps and stairs along the walking direction.

    The terrain consists of a runway where the robot encounters alternating sections
    of gaps (ditches) and stairs. The robot must cross gaps and climb/descend steps.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve terrain parameters based on difficulty
    gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])

    # terrain constants
    terrain_width = cfg.size[0]   # x direction (lateral)
    terrain_length = cfg.size[1]  # y direction (walking direction)
    terrain_depth = 1.0           # depth of ground below surface
    start_platform_length = cfg.start_platform_length
    step_width = cfg.step_width   # depth of each step along y

    meshes_list = list()

    # -- Starting platform (safe zone at the beginning, y from 0 to platform_width)
    start_dim = (terrain_width, start_platform_length, terrain_depth)
    start_pos = (terrain_width / 2, start_platform_length / 2, -terrain_depth / 2)
    start_platform = trimesh.creation.box(start_dim, trimesh.transformations.translation_matrix(start_pos))
    meshes_list.append(start_platform)

    # -- Generate alternating stairs and gaps along y direction
    # Layout: [platform] [stairs section] [gap] [stairs section] [gap] ... [end platform]
    y_cursor = start_platform_length  # start after the initial platform
    num_steps_per_section = cfg.num_steps_per_section
    num_sections = cfg.num_sections

    for i in range(num_sections):
        if i % 2 == 0:
            # === Stairs section ===
            for s in range(num_steps_per_section):
                step_z = (s + 1) * step_height
                step_dim = (terrain_width, step_width, step_z + terrain_depth)
                step_pos = (
                    terrain_width / 2,
                    y_cursor + step_width / 2,
                    step_z / 2 - terrain_depth / 2,
                )
                step_mesh = trimesh.creation.box(
                    step_dim, trimesh.transformations.translation_matrix(step_pos)
                )
                meshes_list.append(step_mesh)
                y_cursor += step_width

            # Then go back down (descending stairs)
            for s in range(num_steps_per_section - 1, -1, -1):
                step_z = s * step_height
                step_dim = (terrain_width, step_width, step_z + terrain_depth)
                step_pos = (
                    terrain_width / 2,
                    y_cursor + step_width / 2,
                    step_z / 2 - terrain_depth / 2,
                )
                step_mesh = trimesh.creation.box(
                    step_dim, trimesh.transformations.translation_matrix(step_pos)
                )
                meshes_list.append(step_mesh)
                y_cursor += step_width
        else:
            # === Gap section ===
            # A flat section before the gap
            flat_len = step_width
            flat_dim = (terrain_width, flat_len, terrain_depth)
            flat_pos = (terrain_width / 2, y_cursor + flat_len / 2, -terrain_depth / 2)
            flat_mesh = trimesh.creation.box(
                flat_dim, trimesh.transformations.translation_matrix(flat_pos)
            )
            meshes_list.append(flat_mesh)
            y_cursor += flat_len

            # The gap itself (no ground, robot falls if it doesn't cross)
            y_cursor += gap_width

            # A flat section after the gap
            flat_dim = (terrain_width, flat_len, terrain_depth)
            flat_pos = (terrain_width / 2, y_cursor + flat_len / 2, -terrain_depth / 2)
            flat_mesh = trimesh.creation.box(
                flat_dim, trimesh.transformations.translation_matrix(flat_pos)
            )
            meshes_list.append(flat_mesh)
            y_cursor += flat_len

    # -- Fill the remaining length with ground
    remaining = terrain_length - y_cursor
    if remaining > 0.1:
        end_dim = (terrain_width, remaining, terrain_depth)
        end_pos = (terrain_width / 2, y_cursor + remaining / 2, -terrain_depth / 2)
        end_mesh = trimesh.creation.box(
            end_dim, trimesh.transformations.translation_matrix(end_pos)
        )
        meshes_list.append(end_mesh)

    # origin at the starting platform center
    origin = np.array([terrain_width / 2, start_platform_length / 2, 0.0])
    return meshes_list, origin

# 混合地形2（Mix2）：为独木桥与台阶的混合地形，包含两项参数，分别是桥体宽度，范围0.5m-1.0m；以及桥上的台阶高度，范围0.1m-0.25m。
def mix2_bridge_stairs_terrain(
    difficulty: float, cfg: cmoe_1_terrains_cfg.MeshMix2TerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a mixed terrain with a narrow bridge that has stairs on it.

    The terrain consists of a narrow bridge (single-log bridge) spanning the walking
    direction with stairs on it. The areas on either side of the bridge are pits,
    forcing the robot to walk on the narrow bridge and navigate the steps.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # Note: bridge_width decreases with difficulty (narrower = harder)
    # So we invert: at difficulty=0, width=max; at difficulty=1, width=min
    bridge_width = cfg.bridge_width_range[1] - difficulty * (
        cfg.bridge_width_range[1] - cfg.bridge_width_range[0]
    )
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )

    # terrain constants
    terrain_width = cfg.size[0]   # x direction (lateral)
    terrain_length = cfg.size[1]  # y direction (walking direction)
    terrain_depth = 1.0           # depth below surface
    platform_width = cfg.platform_width
    step_width = cfg.step_width   # depth of each step along y
    num_steps = cfg.num_steps

    meshes_list = list()

    # -- Starting platform (full width, safe zone)
    start_dim = (terrain_width, platform_width, terrain_depth)
    start_pos = (terrain_width / 2, platform_width / 2, -terrain_depth / 2)
    start_mesh = trimesh.creation.box(
        start_dim, trimesh.transformations.translation_matrix(start_pos)
    )
    meshes_list.append(start_mesh)

    # -- Bridge section with stairs
    # The bridge is centered in x, with width = bridge_width
    bridge_start_y = platform_width
    bridge_center_x = terrain_width / 2

    y_cursor = bridge_start_y

    # Generate stairs going up on the bridge
    for s in range(num_steps):
        step_z = (s + 1) * step_height
        step_dim = (bridge_width, step_width, step_z + terrain_depth)
        step_pos = (
            bridge_center_x,
            y_cursor + step_width / 2,
            step_z / 2 - terrain_depth / 2,
        )
        step_mesh = trimesh.creation.box(
            step_dim, trimesh.transformations.translation_matrix(step_pos)
        )
        meshes_list.append(step_mesh)
        y_cursor += step_width

    # Flat section at the top of the bridge
    top_height = (num_steps) * step_height
    flat_length = step_width * 2  # a short flat section at top
    flat_dim = (bridge_width, flat_length, top_height + terrain_depth)
    flat_pos = (
        bridge_center_x,
        y_cursor + flat_length / 2,
        top_height / 2 - terrain_depth / 2,
    )
    flat_mesh = trimesh.creation.box(
        flat_dim, trimesh.transformations.translation_matrix(flat_pos)
    )
    meshes_list.append(flat_mesh)
    y_cursor += flat_length

    # Generate stairs going down on the bridge
    for s in range(num_steps - 1, -1, -1):
        step_z = s * step_height
        step_dim = (bridge_width, step_width, step_z + terrain_depth)
        step_pos = (
            bridge_center_x,
            y_cursor + step_width / 2,
            step_z / 2 - terrain_depth / 2,
        )
        step_mesh = trimesh.creation.box(
            step_dim, trimesh.transformations.translation_matrix(step_pos)
        )
        meshes_list.append(step_mesh)
        y_cursor += step_width

    # -- Ending platform (full width, safe zone)
    remaining = terrain_length - y_cursor
    if remaining > 0.1:
        end_dim = (terrain_width, remaining, terrain_depth)
        end_pos = (terrain_width / 2, y_cursor + remaining / 2, -terrain_depth / 2)
        end_mesh = trimesh.creation.box(
            end_dim, trimesh.transformations.translation_matrix(end_pos)
        )
        meshes_list.append(end_mesh)

    # -- Ground plane (only for the pit areas on either side of the bridge)
    # We add a low ground plane to catch the robot if it falls
    pit_ground = make_plane(cfg.size, height=-5.0, center_zero=False)
    meshes_list.append(pit_ground)

    # origin at the starting platform center
    origin = np.array([terrain_width / 2, platform_width / 2, 0.0])

    return meshes_list, origin
