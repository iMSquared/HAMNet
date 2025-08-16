#!/usr/bin/env python3

from typing import Tuple, Dict, List, Any, Optional, Iterable
from collections import OrderedDict
from dataclasses import dataclass, replace
from ham.util.config import ConfigBase
import pkg_resources
from pathlib import Path
import json
import os
import pickle
from functools import partial
from tempfile import TemporaryDirectory

import numpy as np
import einops
from tqdm.auto import tqdm
import trimesh
import itertools
import re
import math
import random

from isaacgym import gymtorch
from isaacgym import gymapi

import torch as th

from cho_util.math import transform as tx

from ham.env.env.base import EnvBase
from ham.env.scene.base import SceneBase
from ham.env.scene.tabletop_scene import TableTopScene
from ham.env.common import (
    create_camera, apply_domain_randomization, set_actor_friction,
    aggregate, set_actor_restitution)
from ham.data.transforms.io_xfm import load_mesh
from ham.util.torch_util import dcn, randu, randu_like
# from ham.env.task.util import (sample_goal_v2 as sample_goal)
from ham.util.math_util import (quat_multiply,
                                quat_rotate,
                                random_quat)

from isaacgym.gymutil import WireframeBoxGeometry, draw_lines
from isaacgym.torch_utils import quat_from_euler_xyz

from ham.env.scene.object_set import ObjectSet
from ham.env.scene.dgn_object_set import DGNObjectSet
from ham.env.scene.mesh_object_set import MeshObjectSet
from ham.env.scene.filter_object_set import FilteredObjectSet, FilterDims
from ham.env.scene.combine_object_set import CombinedObjectSet
from ham.env.scene.util import (
    create_bin,
    create_bump,
    create_step,
    create_cabinet,
    _is_stable,
    _foot_radius)
from ham.env.scene.sample_pose import (
    SampleRandomOrientation,
    SampleCuboidOrientation,
    SampleStableOrientation,
    SampleMixtureOrientation,
    RandomizeYaw,
    z_from_q_and_hull,
    sample_bump_xy,
    sample_flat_xy,
    sample_goal_xy_v2,
)

from ham.models.common import merge_shapes
from ham.data.transforms.sample_points_from_urdf import sample_surface_points_from_urdf

import nvtx
from icecream import ic
from tempfile import mkdtemp

DATA_ROOT = os.getenv('HAM_DATA_ROOT', '/input')

CLOUD_SIZE: int = 4096

# [1] plain table
# [2] bump
# [3] 'wall' or 'dancha' (staircase ?)


def array_to_string(x):
    return (np.array2string(x, precision=3, separator=',',)
            .replace(' ', '')
            .replace('[', '')
            .replace(']', ''))


def _to_id(*args):
    return ':'.join([array_to_string(np.asarray(dcn(x)))
                     for x in args])


def is_thin(extent, threshold: float = 2.5):
    size = np.sort(extent)
    return (size[1] >= threshold * size[0])


def sample_cuboid_poses(n: int, noise: float = np.deg2rad(5.0)):
    IRT2 = math.sqrt(1.0 / 2)
    canonicals = np.asarray([
        [0.000, 0.000, 0.000, 1.000],
        [1.000, 0.000, 0.000, 0.000],
        [-IRT2, 0.000, 0.000, +IRT2],
        [+IRT2, 0.000, 0.000, +IRT2],
        [0.000, -IRT2, 0.000, +IRT2],
        [0.000, +IRT2, 0.000, +IRT2],
    ], dtype=np.float32)
    indices = np.random.choice(
        len(canonicals),
        size=n)
    qs = canonicals[indices]

    # Add slight noise to break symmetry
    # in case of highly unstable configurations
    qzs = tx.rotation.axis_angle.random(size=n)
    # qzs[..., 3] *= noise
    qzs[..., 3] = np.random.uniform(-noise, +noise,
                                    size=qzs[..., 3].shape)
    qzs = tx.rotation.quaternion.from_axis_angle(qzs)
    qs = tx.rotation.quaternion.multiply(qzs, qs)

    return qs


def sample_points_from_urdf(urdf_file: str,
                            count: int = CLOUD_SIZE,
                            z_min: float = 0.0
                            ):
    return sample_surface_points_from_urdf(urdf_file, count,
                                           use_poisson=False,
                                           use_even=False,
                                           proc_mesh=lambda m: m.slice_plane(
                                               (0, 0, 0), (0, 0, 1))
                                           )


def _pad_hulls(hulls: Dict[str, trimesh.Trimesh]) -> Dict[str, np.ndarray]:
    n: int = max([len(h.vertices) for h in hulls.values()])
    out = {}
    for k, h in hulls.items():
        p = np.empty((n, 3), dtype=np.float32)
        v = h.vertices
        c = np.mean(v, axis=0, keepdims=True)
        p[:len(v)] = v
        p[len(v):] = c
        out[k] = p
    return out


def _array_from_map(
        keys: List[str],
        maps: Dict[str, th.Tensor],
        **kwds):
    if not isinstance(next(iter(maps.values())), th.Tensor):
        arr = np.stack([maps[k] for k in keys])
        return th.as_tensor(arr, **kwds)
    return th.stack([maps[k] for k in keys], dim=0).to(**kwds)


def _create_table_asset_options():
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.disable_gravity = True
    asset_options.enable_gyroscopic_forces = True
    asset_options.flip_visual_attachments = False
    asset_options.vhacd_enabled = False
    asset_options.thickness = 0.001  # ????
    asset_options.convex_decomposition_from_submeshes = True
    asset_options.override_com = True
    asset_options.override_inertia = True
    return asset_options


def _load_flat_asset(gym, sim, table_dims):
    suffix = _to_id(table_dims)
    asset_options = _create_table_asset_options()
    table_asset = gym.create_box(sim,
                                 *table_dims,
                                 asset_options)

    table_cloud, _ = trimesh.sample.sample_surface(
        trimesh.creation.box(table_dims).slice_plane(
            (0, 0, 0.5 * table_dims[2] - 0.01), (0, 0, 1)),
        CLOUD_SIZE)

    return table_asset, table_cloud


def _load_wall_asset(gym, sim, table_dims,
                     wall_width, wall_height,
                     base_dims=None):
    wall_width = float(wall_width)
    wall_height = float(wall_height)
    suffix = _to_id(table_dims, wall_width, wall_height)
    asset_options = _create_table_asset_options()
    bin_urdf_text = create_bin(table_dims=table_dims,
                               wall_width=wall_width,
                               wall_height=wall_height,
                               base_dims=base_dims)

    with TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir) / F'wall-{suffix}.urdf'
        with open(str(filename), 'w') as fp:
            fp.write(bin_urdf_text)
        table_asset = gym.load_urdf(sim,
                                    tmpdir,
                                    filename.name,
                                    asset_options)
        table_cloud = sample_points_from_urdf(filename,
                                              z_min=0.5 * table_dims[2])
    return table_asset, table_cloud


def _load_bin_urdf_asset(cfg, gym, sim):
    raise ValueError('This function is deprecated !')
    asset_options = _create_table_asset_options()
    table_asset = gym.load_urdf(sim,
                                str(cfg.asset_root),
                                str(cfg.table_file),
                                asset_options)
    table_cloud = sample_points_from_urdf(cfg.table_file,
                                          z_min=0.0
                                          )
    return table_asset, table_cloud


def _load_bump_asset(gym, sim, table_dims,
                     bump_width, bump_height,
                     bump_pos,
                     base_dims=None):
    bump_width = float(bump_width)
    bump_height = float(bump_height)
    bump_pos = float(bump_pos)
    suffix = _to_id(table_dims, bump_width, bump_height, bump_pos)
    asset_options = _create_table_asset_options()
    bump_urdf_text = create_bump(table_dims=table_dims,
                                 bump_width=bump_width,
                                 bump_height=bump_height,
                                 bump_pos=bump_pos,
                                 base_dims=base_dims)

    with TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir) / F'bump-{suffix}.urdf'
        with open(str(filename), 'w') as fp:
            fp.write(bump_urdf_text)
        table_asset = gym.load_urdf(sim,
                                    tmpdir,
                                    filename.name,
                                    asset_options)
        table_cloud = sample_points_from_urdf(filename,
                                              z_min=0.5 * table_dims[2]
                                              )
    return table_asset, table_cloud


def _load_step_asset(gym, sim, table_dims,
                     step_pos,
                     step_height,
                     base_dims=None,
                     step_tilt: float = 0.0):
    step_pos = float(step_pos)
    step_height = float(step_height)

    suffix = _to_id(table_dims, step_pos, step_height)
    asset_options = _create_table_asset_options()
    # FIXME(ycho): currently bump_height == step_height
    # ^^ forced and hardcoded
    step_urdf_text = create_step(table_dims=table_dims,
                                 step_pos=step_pos,
                                 step_height=step_height,
                                 base_dims=base_dims,
                                 step_tilt=step_tilt
                                 )

    with TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir) / F'step-{suffix}.urdf'
        with open(str(filename), 'w') as fp:
            fp.write(step_urdf_text)
        table_asset = gym.load_urdf(sim,
                                    tmpdir,
                                    filename.name,
                                    asset_options)
        table_cloud = sample_points_from_urdf(filename,
                                              z_min=0.5 * table_dims[2]
                                              )

    return table_asset, table_cloud


def _load_cabinet_asset(gym, sim, table_dims,
                        cabinet_height,
                        wall_width,
                        base_dims=None):
    cabinet_height = float(cabinet_height)
    wall_width = float(wall_width)
    suffix = _to_id(table_dims, cabinet_height)
    asset_options = _create_table_asset_options()
    cabinet_urdf_text = create_cabinet(
        table_dims=table_dims,
        cabinet_height=cabinet_height,
        wall_width=wall_width,
        base_dims=base_dims)
    with TemporaryDirectory() as tmpdir:
        filename = Path(tmpdir) / F'cabinet-{suffix}.urdf'
        with open(str(filename), 'w') as fp:
            fp.write(cabinet_urdf_text)
        table_asset = gym.load_urdf(sim,
                                    tmpdir,
                                    filename.name,
                                    asset_options)
        table_cloud = sample_points_from_urdf(filename,
                                              z_min=0.5 * table_dims[2]
                                              )
    return table_asset, table_cloud


@dataclass
class Properties:
    """
    Current environment properties.
    """
    ids: Optional[th.Tensor] = None
    handles: Optional[th.Tensor] = None
    radii: Optional[th.Tensor] = None
    stable_poses: Optional[th.Tensor] = None
    stable_masks: Optional[th.Tensor] = None
    foot_radius: Optional[th.Tensor] = None
    embeddings: Optional[th.Tensor] = None
    bboxes: Optional[th.Tensor] = None  # really more like 'keypoints'
    extent: Optional[th.Tensor] = None
    names: Optional[List[str]] = None
    cloud: Optional[th.Tensor] = None
    normal: Optional[th.Tensor] = None
    hulls: Optional[th.Tensor] = None
    predefined_goal: Optional[th.Tensor] = None
    object_friction: Optional[th.Tensor] = None
    table_friction: Optional[th.Tensor] = None
    object_restitution: Optional[th.Tensor] = None
    # currently these are fixed across all episodes
    # table_pos: Optional[th.Tensor] = None
    # table_dim: Optional[th.Tensor] = None
    yaw_only: Optional[th.Tensor] = None
    # scale: Optional[th.Tensor] = None
    # scene_type: Optional[List[str]] = None
    barrier_height: Optional[th.Tensor] = None


class TableTopWithObjectScene(TableTopScene):

    @dataclass
    class Config(TableTopScene.Config):
        data_root: str = F'{DATA_ROOT}/ACRONYM/urdf'
        # Convex hull for quickly computing initial placements.
        hull_root: str = F'{DATA_ROOT}/ACRONYM/hull'
        mesh_count: str = F'{DATA_ROOT}/ACRONYM/mesh_count.json'
        urdf_stats_file: str = F'{DATA_ROOT}/ACRONYM/urdf_stats.json'
        stable_poses_file: str = F'{DATA_ROOT}/ACRONYM/stable_poses.pkl'
        # stable_poses_file: str = F'/input/ACRONYM/train_chair_poses.pkl'
        embeddings_file: str = F'{DATA_ROOT}/ACRONYM/embedding.pkl'
        patch_center_file: str = '/input/ACRONYM/patch-v12.pkl'
        bbox_file: str = F'{DATA_ROOT}/ACRONYM/bbox.pkl'
        cloud_file: str = F'{DATA_ROOT}/ACRONYM/cloud.pkl'
        cloud_normal_file: str = F'{DATA_ROOT}/ACRONYM/cloud_normal.pkl'
        volume_file: str = F'{DATA_ROOT}/ACRONYM/volume.pkl'

        stable_poses_url: Optional[str] = 'https://drive.google.com/file/d/179nwfNJFibA9Y2_5B9OiuAARkC8dggmY/view?usp=share_link'
        embeddings_url: Optional[str] = 'https://drive.google.com/file/d/1J_d1gTJb8SZ14h0xIHZsCkTJoJmNVibI/view?usp=share_link'
        bbox_url: Optional[str] = 'https://drive.google.com/file/d/1T6oYQ2DI6a13HDGCcr6hs6zkMXwL51Sd/view?usp=share_link'
        cloud_url: Optional[str] = 'https://drive.google.com/file/d/11zoKfD_jh49X9YETNl7xhoDFJOPzrTmj/view?usp=share_link'
        cloud_normal_url: Optional[str] = 'https://drive.google.com/file/d/1VnmusxHKHkHCRIIuIsBh4lEjldi8hPeO/view?usp=share_link'
        volume_url: Optional[str] = 'https://drive.google.com/file/d/1Ue2fe2CPIB1OOEmmU9z9Mb4VmPWtS5_C/view?usp=share_link'

        use_wall: bool = False
        use_bin: bool = False
        use_bump: bool = False

        wall_width: float = 0.01
        wall_height: float = 0.04

        bump_width: float = 0.04
        bump_height: float = 0.04

        bump_width_bound: Tuple[float, float] = (0.01, 0.1)
        bump_height_bound: Tuple[float, float] = (0.0, 0.2)
        randomize_barrier_height: bool = True

        cabinet_height_bound: Tuple[float, float] = (0.3, 0.5)

        # wall_table_urdf: str = '../../data/assets/table-with-wall/robot.urdf'
        asset_root: str = pkg_resources.resource_filename('ham.data', 'assets')
        # table_file: str = 'table-with-wall/table-with-wall-open.urdf'
        # table_file: str = 'table-with-wall/table-with-wall.urdf'
        table_file: str = 'table-with-wall/table-with-wall-small.urdf'

        table_friction: Optional[float] = None
        object_friction: Optional[float] = None

        num_obj_per_env: int = 1
        z_eps: float = 1e-2

        # Or "sample", "zero", ...
        init_type: str = 'sample'
        goal_type: str = 'random'
        randomize_yaw: bool = False

        randomize_init_pos: bool = False
        randomize_init_orn: bool = False
        # In order to increase the likelihood of sampling dynamically
        # unstable goal configurations, sample init pose of dropping
        # as identity with given probability
        canonical_pose_prob: float = 0.2  # need to be randomize_init_orn True

        add_force_sensor_to_com: bool = False
        avoid_overlap: bool = True

        use_dr: bool = False
        use_dr_on_setup: bool = False
        use_mass_set: bool = False
        min_mass: float = 0.01
        max_mass: float = 2.0
        mass_set: Tuple[float, ...] = (0.1, 1.0, 10.0)
        use_scale_dr: bool = False
        min_scale: float = 0.09
        max_scale: float = 0.15

        min_object_friction: float = 0.2
        max_object_friction: float = 1.0
        min_table_friction: float = 0.6
        max_table_friction: float = 1.0
        min_object_restitution: float = 0.0
        max_object_restitution: float = 0.2

        # Old setting, load one single object
        diverse_object: bool = False

        # `filter_index` is only used during stable_poses generation...
        filter_index: Optional[Tuple[int, ...]] = None
        # `filter_class` to select categories...
        filter_class: Optional[Tuple[str, ...]] = None
        # `filter_key` to select specific objects ...
        filter_key: Optional[Tuple[str, ...]] = None
        # `filter_file` to select specific objects from file...
        filter_file: Optional[str] = None
        filter_complex: bool = True
        filter_dims: Optional[Tuple[float, float, float]] = None
        filter_pose_count: Optional[Tuple[int, int]] = None
        truncate_pose_count: int = 64
        filter_thin: bool = False

        # keys or file that contains key for yaw only objects
        yaw_only_key: Optional[Tuple[str, ...]] = None
        yaw_only_file: Optional[str] = None
        use_yaw_only_logic: bool = True
        thin_threshold: float = 2.5

        load_embedding: bool = False
        load_bbox: bool = False
        load_obb: bool = False
        load_cloud: bool = False
        load_normal: bool = False
        load_predefined_goal: bool = False
        load_stable_mask: bool = True
        load_foot_radius: bool = False

        # default_mesh: str =
        # 'Speaker_64058330533509d1d747b49524a1246e_0.003949258269301651.glb'
        default_mesh: str = 'RubiksCube_d7d3dc14748ec6d347cd142fcccd1cc2_8.634340549903529e-05.glb'
        # default_mesh: str = 'RubiksCube_cdda3ea70d829d5baa9ba8e71ae84fd3_0.02768212782072632.glb'
        # default_mesh: str =
        # 'RubiksCube_d060362a42a3ef0af0a72b757e578a97_0.05059491278967159.glb'

        # FIXME(ycho): `num_object_types` should usually match num_env.
        num_object_types: int = 512
        # max_vertex_count: int = 2500
        max_vertex_count: int = 8192
        max_chull_count: int = 128
        margin_scale: float = 0.95
        prevent_fall: bool = True

        load_convex: bool = False
        override_inertia: bool = False
        density: float = 200.0
        target_mass: Optional[float] = None

        base_set: Tuple[str, ...] = ('dgn',)
        dgn: DGNObjectSet.Config = DGNObjectSet.Config()
        mesh: MeshObjectSet.Config = MeshObjectSet.Config()
        need_attr: Optional[Tuple[str, ...]] = ('num_verts',)

        mode: str = 'train'
        num_valid_poses: int = 1

        restitution: Optional[float] = None

        # hmm...
        scene_types: Tuple[str, ...] = ('flat',)

        hack_climb_only: bool = False
        enable_climb_wall: bool = False
        hack_balance_types: bool = False
        hack_step_tilt: float = float(np.deg2rad(0.0))

        def __post_init__(self):
            if self.filter_dims is not None:
                d_min, d_max, r_max = self.filter_dims
                self.cuboid = replace(self.cuboid,
                                      min_dim=d_min,
                                      max_dim=d_max,
                                      max_aspect=r_max)
                self.cone = replace(self.cone,
                                    min_dim=d_min,
                                    max_dim=d_max,
                                    max_aspect=r_max)
                self.cylinder = replace(self.cylinder,
                                        min_dim=d_min,
                                        max_dim=d_max,
                                        max_aspect=r_max)
                self.prism = replace(self.prism,
                                     min_dim=d_min,
                                     max_dim=d_max,
                                     max_aspect=r_max)
            if self.use_bump:
                self.filter_thin = True

            if (self.use_bump) or 'bump' in self.scene_types:
                self.load_foot_radius = True

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # cfg.filter_index,
        self.meta = self.load_obj_set()

        self.keys: List[str] = []
        self.assets: Dict[str, Any] = {}
        self.hulls: th.Tensor = None
        self.radii: Dict[str, float] = None
        self.sensors = {}

        # WHAT WE NEED:
        # > ACTOR INDEX, for set_actor_root_state_tensor_index
        # > RIGID-BODY INDEX, for apply_rigid_body_force_at_pos_tensors
        self.obj_ids: th.Tensor = None
        self.obj_handles: th.Tensor = None
        self.obj_body_ids: th.Tensor = None
        self.body_ids: th.Tensor = None
        self._pos_scale: float = 1.0
        self.table_pos: th.Tensor = None
        self.table_dims: th.Tensor = None
        self.is_yaw_only: th.Tensor = None
        self._per_env_offsets: th.Tensor = None
        self.cur_props: Properties = Properties()
        self.max_barrier_height: float = cfg.bump_height_bound[1]
        self.barrier_cloud: Optional[th.Tensor] = None

    def load_obj_set(self) -> ObjectSet:
        cfg = self.cfg
        allowlist = None

        assert (len(cfg.base_set) > 0)

        obj_sets = []
        for base_set in cfg.base_set:
            if base_set == 'dgn':
                meta = DGNObjectSet(cfg.dgn)
            elif base_set == 'mesh':
                meta = MeshObjectSet(cfg.mesh)
            else:
                raise KeyError(F'Unknown base object set = {cfg.base_set}')
            keys = meta.keys()
            print(F'init : {len(keys)}')

            # First, filter by availability of all required fields.
            # Determine required attributes...
            need_attr = list(cfg.need_attr)
            if need_attr is None:
                need_attr = []
            if cfg.goal_type == 'stable':
                need_attr.append('pose')
            for attr in need_attr:
                query = getattr(meta, attr)
                fkeys = []
                for key in keys:
                    try:
                        if query(key) is None:
                            continue
                    except KeyError:
                        continue
                    fkeys.append(key)
                keys = fkeys
            print(F'after filter by "need" : {len(keys)}')

            # Filter by `filter_class`.
            if cfg.filter_class is not None:
                keys = [key for key in keys
                        if meta.label(key) in cfg.filter_class]
            print(F'after filter by "class" : {len(keys)}')

            if cfg.filter_key is not None:
                keys = [key for key in keys
                        if key in cfg.filter_key]
            print(F'after filter by "filter_key" : {len(keys)}')

            if cfg.filter_file is not None:
                with open(cfg.filter_file, 'r') as fp:
                    allowlist = [str(s) for s in json.load(fp)]
                keys = [key for key in keys
                        if key in allowlist]
            print(F'after filter by "filter_file" : {len(keys)}')

            # Filter by size, and remove degenerate mesh
            if cfg.filter_complex:
                keys = [key for key in keys if
                        (meta.num_verts(key) < cfg.max_vertex_count and
                         meta.num_hulls(key) < cfg.max_chull_count and
                         key != '4Shelves_fd0fd7b2c19cce39e3783ec57dd5d298_0.001818238917007018')
                        ]
            print(F'after filter by "complex" : {len(keys)}')

            if cfg.filter_dims is not None:
                d_min, d_max, r_max = cfg.filter_dims
                f = FilterDims(d_min, d_max, r_max)
                keys = [key for key in keys if f(meta, key)]
            print(F'after filter by "dims" : {len(keys)}')

            # Filter by size, and remove degenerate mesh
            if cfg.filter_pose_count:
                pmin, pmax = cfg.filter_pose_count
                keys = [key for key in keys if
                        meta.pose(key) is not None
                        and pmin <= meta.pose(key).shape[0]
                        and meta.pose(key).shape[0] < pmax
                        ]
            print(F'after filter by "pose_count" : {len(keys)}')
            if cfg.filter_thin:
                keys = [key for key in keys
                        if not is_thin(meta.obb(key)[1],
                                       threshold=cfg.thin_threshold)
                        ]
                print(F'after filter by "thin logic" : {len(keys)}')

            obj_sets.append(FilteredObjectSet(meta, keys=keys))
        if len(obj_sets) == 1:
            return obj_sets[0]
        return CombinedObjectSet(obj_sets)

    def setup(self, env: 'EnvBase'):
        cfg = self.cfg

        obj_ids = []
        obj_handles = []
        obj_body_ids = []
        self.tmpdir = None
        num_env: int = env.num_env

        for i in range(num_env):
            for j in range(cfg.num_obj_per_env):
                obj_id = env.gym.find_actor_index(
                    env.envs[i],
                    F'object-{j:02d}',
                    gymapi.IndexDomain.DOMAIN_SIM)
                obj_ids.append(obj_id)

                obj_handle = env.gym.find_actor_handle(
                    env.envs[i],
                    F'object-{j:02d}')
                obj_handles.append(obj_handle)

                obj_body_id = env.gym.find_actor_rigid_body_index(
                    env.envs[i],
                    obj_handle,
                    'base_link',
                    gymapi.IndexDomain.DOMAIN_ENV
                )
                obj_body_ids.append(obj_body_id)

        # actor indices
        self.obj_ids = th.as_tensor(obj_ids,
                                    dtype=th.int32,
                                    device=env.cfg.th_device).reshape(
            env.cfg.num_env, cfg.num_obj_per_env)

        # actor handles
        self.obj_handles = th.as_tensor(obj_handles,
                                        dtype=th.int32,
                                        device=env.cfg.th_device).reshape(
            env.cfg.num_env, cfg.num_obj_per_env)
        self.obj_body_ids = th.as_tensor(obj_body_ids,
                                         dtype=th.int32,
                                         device=env.cfg.th_device).reshape(
            env.cfg.num_env, cfg.num_obj_per_env)

        self.table_handles = [
            env.gym.find_actor_handle(env.envs[i], 'table')
            for i in range(env.cfg.num_env)]

        self.barrier_handles = [
            env.gym.find_actor_handle(env.envs[i], 'barrier')
            for i in range(env.cfg.num_env)]

        self.barrier_ids = th.as_tensor([
            env.gym.find_actor_index(
                env.envs[i],
                F'barrier',
                gymapi.IndexDomain.DOMAIN_SIM)
            for i in range(env.cfg.num_env)],
            dtype=th.long,
            device=env.cfg.th_device)

        # FIXME(ycho): it will be quite nontrivial
        # to figure out the domains of these ids
        # in the perspective of the external API...
        self.table_body_ids = [
            env.gym.get_actor_rigid_body_index(
                env.envs[i],
                self.table_handles[i],
                0,
                gymapi.IndexDomain.DOMAIN_SIM
            ) for i in range(env.cfg.num_env)
        ]

        self.mask = th.zeros(
            env.tensors['root'].shape[0],
            dtype=bool,
            device=env.cfg.th_device
        )
        self.scales = []
        self.cur_props.object_friction = randu(
            cfg.min_object_friction,
            cfg.max_object_friction,
            size=(num_env, cfg.num_obj_per_env),
            device=env.device)
        self.cur_props.table_friction = randu(
            cfg.min_table_friction,
            cfg.max_table_friction,
            size=(num_env, cfg.num_obj_per_env),
            device=env.device)
        self.cur_props.object_restitution = randu(
            cfg.min_object_restitution,
            cfg.max_object_restitution,
            size=(num_env, cfg.num_obj_per_env),
            device=env.device
        )
        self.cur_props.barrier_height = randu(
            cfg.bump_height_bound[0],
            self.max_barrier_height,
            size=(num_env,),
            device=env.device)
        # TODO(ycho): avoid recomputing scene clouds!
        # if not cfg.randomize_barrier_height:
        # self.__scene_cloud = cache_clouds()

        if cfg.use_dr_on_setup:
            # TODO(ycho): dcn() necessary??
            obj_fr = dcn(self.cur_props.object_friction)
            tbl_fr = dcn(self.cur_props.table_friction)
            obj_rs = dcn(self.cur_props.object_restitution)

            # Apply domain randomization.
            for i in range(env.cfg.num_env):
                apply_domain_randomization(
                    env.gym, env.envs[i],
                    self.table_handles[i],
                    enable_friction=True,
                    min_friction=tbl_fr[i],
                    max_friction=tbl_fr[i])

                apply_domain_randomization(
                    env.gym, env.envs[i],
                    self.barrier_handles[i],
                    enable_friction=True,
                    min_friction=tbl_fr[i],
                    max_friction=tbl_fr[i])
                for j in range(cfg.num_obj_per_env):
                    out = apply_domain_randomization(
                        env.gym, env.envs[i],
                        self.obj_handles[i, j],

                        enable_mass=True,
                        min_mass=cfg.min_mass,
                        max_mass=cfg.max_mass,
                        use_mass_set=cfg.use_mass_set,
                        mass_set=cfg.mass_set,

                        change_scale=cfg.use_scale_dr,
                        min_scale=cfg.min_scale,
                        max_scale=cfg.max_scale,
                        radius=self.radii[i],

                        enable_friction=True,
                        min_friction=obj_fr[i, j],
                        max_friction=obj_fr[i, j],

                        enable_restitution=True,
                        min_restitution=obj_rs[i, j],
                        max_restitution=obj_rs[i, j]
                    )
                    if 'scale' in out:
                        self.scales.append(out['scale'])

        if cfg.restitution is not None:
            for i in range(env.cfg.num_env):
                for j in range(cfg.num_obj_per_env):
                    set_actor_restitution(env.gym,
                                          env.envs[i],
                                          self.obj_handles[i, j],
                                          restitution=cfg.restitution)

        if len(self.scales) > 0:
            from xml.dom import minidom
            if self.tmpdir is None:
                self.tmpdir = mkdtemp()
            for idx, scale in enumerate(self.scales):
                key = self.keys[idx]
                urdf = self.meta.urdf(key)
                with open(urdf, 'r', encoding='utf-8') as f:
                    str_urdf = f.read()
                dom = minidom.parseString(str_urdf)
                meshes = dom.getElementsByTagName("mesh")
                for mesh in meshes:
                    mesh_scales = mesh.attributes['scale'].value.split(' ')
                    new_scale = [str(scale.item() * float(mesh_scale))
                                 for mesh_scale in mesh_scales]
                    mesh.attributes['scale'].value = \
                        ' '.join(new_scale)
                with open(f'{self.tmpdir}/{key}.urdf', "w") as f:
                    dom.writexml(f)
            self.scales = th.as_tensor(self.scales,
                                       device=env.device)
            self.radii = self.radii * self.scales
            self.hulls = self.hulls * self.scales[:, None, None]
            if cfg.load_bbox:
                self.bboxes[..., :] = (self.bboxes[..., :] *
                                       self.scales[:, None, None])
            if cfg.load_cloud:
                self.cloud[..., :] = (self.cloud[..., :] *
                                      self.scales[:, None, None])
            if cfg.goal_type == 'stable':
                poses = self.stable_poses.clone()
                table_h = cfg.table_dims[-1]
                poses[..., 2] -= table_h
                poses[..., 2] = poses[..., 2] * self.scales[:, None]
                self.stable_poses[..., 2] = poses[..., 2] + table_h
        # Validation.
        # masses = np.zeros(env.cfg.num_env)
        # for i in range(env.cfg.num_env):
        #     prop = env.gym.get_actor_rigid_body_properties(
        #         env.envs[i],
        #         self.obj_handles[i, 0].item()
        #     )
        #     masses[i] = prop[0].mass
        # print(F'masses = {masses}')
        self.table_pos = einops.repeat(
            th.as_tensor(np.asarray(cfg.table_pos),
                         dtype=th.float,
                         device=env.device),
            '... -> n ...', n=env.num_env)
        self.table_dims = einops.repeat(
            th.as_tensor(np.asarray(cfg.table_dims),
                         dtype=th.float,
                         device=env.device),
            '... -> n ...', n=env.num_env)
        self._per_env_offsets = th.zeros((env.num_env),
                                         dtype=th.long,
                                         device=env.device)

    def _reset_props(self, gym, sim, env,
                     env_ids: th.Tensor,
                     num_reset: int,
                     reset_all: bool):
        cfg = self.cfg

        with nvtx.annotate("build_nxt"):
            # <==> SELECT OBJECT TO MOVE TO TABLE <==>
            if cfg.mode == 'valid':
                # Cycle through objects in deterministic order.
                self._per_env_offsets[env_ids] += 1
                # self._per_env_offsets[env_ids] %= cfg.num_obj_per_env
                offsets = self._per_env_offsets[env_ids] % cfg.num_obj_per_env
            else:
                offsets = th.randint(cfg.num_obj_per_env,
                                     size=(num_reset,),
                                     device=env.device)
            nxt = {}

            nxt['ids'] = self.obj_ids[env_ids, offsets]
            nxt['handles'] = self.obj_handles[env_ids, offsets]
            nxt['body_ids'] = self.obj_body_ids[env_ids, offsets]
            # ni = nxt['ids'].long()

            # lookup-table indices
            lut_indices = env_ids * cfg.num_obj_per_env + offsets
            lut_indices_np = dcn(lut_indices)

            nxt['names'] = [self.keys[i] for i in lut_indices_np]
            nxt['radii'] = self.radii[lut_indices]
            nxt['hulls'] = self.hulls[lut_indices]
            if cfg.goal_type == 'stable':
                nxt['stable_poses'] = self.stable_poses[lut_indices]
                if self.is_yaw_only is not None:
                    nxt['yaw_only'] = self.is_yaw_only[lut_indices]
                if cfg.load_stable_mask:
                    nxt['stable_masks'] = self.stable_masks[lut_indices]
                if cfg.load_foot_radius:
                    nxt['foot_radius'] = self.foot_radius[lut_indices]
            if cfg.load_embedding:
                nxt['embeddings'] = self.object_embeddings[lut_indices]
            if cfg.load_bbox:
                nxt['bboxes'] = self.bboxes[lut_indices]
            if cfg.load_cloud:
                nxt['cloud'] = self.cloud[lut_indices]
                if cfg.load_normal:
                    nxt['normal'] = self.normal[lut_indices]
            if cfg.load_predefined_goal:
                nxt['predefined_goal'] = self.predefined_goal[lut_indices]

            # randomize
            if cfg.use_dr:
                nxt['object_friction'] = (
                    th.empty(size=(num_reset, cfg.num_obj_per_env),
                             device=env.device)
                    .uniform_(cfg.min_object_friction,
                              cfg.max_object_friction))
                nxt['table_friction'] = (
                    th.empty(size=(num_reset,), device=env.device)
                    .uniform_(cfg.min_table_friction, cfg.max_table_friction)
                )
                nxt['object_restitution'] = (
                    th.empty(
                        size=(num_reset, cfg.num_obj_per_env),
                        device=env.device)
                    .uniform_(cfg.min_object_restitution,
                              cfg.max_object_restitution))

            if cfg.randomize_barrier_height:
                nxt['barrier_height'] = randu(cfg.bump_height_bound[0],
                                              self.max_barrier_height,
                                              size=(num_reset,),
                                              device=env.device)

        # <==> DOMAIN RANDOMIZATION <==>
        with nvtx.annotate("apply_dr"):
            if cfg.use_dr:
                # one per env
                tbl_fr = dcn(nxt['table_friction']).ravel()
                # maybe many per env
                obj_frs = dcn(nxt['object_friction']).ravel()
                obj_rss = dcn(nxt['object_restitution']).ravel()
                with nvtx.annotate("c"):
                    for i, (env_id, obj_handle, obj_fr, obj_rs) in enumerate(
                            zip(env_ids, nxt['handles'], obj_frs, obj_rss)):
                        apply_domain_randomization(
                            gym, env.envs[int(env_id)], obj_handle,
                            enable_friction=True,
                            min_friction=obj_fr,
                            max_friction=obj_fr,
                            enable_restitution=True,
                            min_restitution=obj_rs,
                            max_restitution=obj_rs,
                        )
                        apply_domain_randomization(
                            gym, env.envs[int(env_id)],
                            self.table_handles[int(env_id)],
                            enable_friction=True,
                            min_friction=tbl_fr[i],
                            max_friction=tbl_fr[i]
                        )
                        apply_domain_randomization(
                            gym, env.envs[int(env_id)],
                            self.barrier_handles[int(env_id)],
                            enable_friction=True,
                            min_friction=tbl_fr[i],
                            max_friction=tbl_fr[i]
                        )

        with nvtx.annotate("commit_nxt"):
            # [2] Commit `nxt` objects.
            if not reset_all:
                for k, v in nxt.items():
                    buf = getattr(self.cur_props, k)
                    if isinstance(buf, th.Tensor):
                        buf[env_ids] = v
                    elif k == 'names':  # or check type:=List[str]
                        for i, j in enumerate(dcn(env_ids)):
                            buf[j] = v[i]
            else:
                for k, v in nxt.items():
                    setattr(self.cur_props, k, v)
        return nxt

    def _reset_poses(self, gym, sim, env,
                     env_ids: th.Tensor,
                     num_reset: int,
                     reset_all: bool,
                     prv_ids: th.Tensor,
                     nxt: Dict[str, th.Tensor]
                     ):
        cfg = self.cfg

        # [1] Reset prv objects' poses
        # to arbitrary positions in the environment.
        # Basically, the intent of this code is to
        # disable simulation for these objects.
        root_tensor = env.tensors['root']
        with nvtx.annotate("d"):
            if (cfg.num_obj_per_env > 1) and (prv_ids is not None):
                pi = None if (prv_ids is None) else prv_ids.long()
                # pos, orn, lin.vel/ang.vel --> 0
                root_tensor[pi] = 0
                # NOTE(ycho): "somewhere sufficiently far away"
                root_tensor[pi, 0] = (prv_ids + 1).float() * 100.0
                root_tensor[pi, 2] = 1.0
                # (0,1,2), (3,4,5,6)
                # Set orientation to unit quaternion
                root_tensor[pi, 6] = 1

        # [2] Nothing to do = return
        ni = nxt['ids'].long()
        if len(ni) <= 0:
            return self.barrier_ids.ravel()

        # [3] First set position + velocity to zeros
        with nvtx.annotate("1"):
            root_tensor[ni] = 0

        # [4] In `valid` mode,
        # Select from one of the preconfigured poses,
        # then exit early.
        if cfg.mode == 'valid':
            pose_index = ((self._per_env_offsets[env_ids]
                           // cfg.num_obj_per_env)
                          % self._valid_poses.shape[-2])
            root_tensor[ni, :7] = self._valid_poses[
                env_ids,
                pose_index
            ]
            return

        # [5] At this point, we assume we're in `train` mode.

        # 5A. Create orientation sampler.
        if cfg.init_type == 'random':
            sample_q = SampleRandomOrientation(env.device)
        elif cfg.init_type == 'cuboid':
            sample_q = SampleCuboidOrientation(env.device)
        elif cfg.init_type == 'stable':
            sample_q = SampleStableOrientation(partial(getattr, self.cur_props,
                                                       'stable_poses'))
        elif cfg.init_type == 'random+cuboid':
            sample_q = SampleMixtureOrientation(
                [SampleRandomOrientation(env.device),
                 SampleCuboidOrientation(env.device)],
                [1.0 - cfg.canonical_pose_prob, cfg.canonical_pose_prob])
        else:
            raise ValueError(F'Unknown init_type={cfg.init_type}')
        if cfg.randomize_yaw:
            sample_q = RandomizeYaw(sample_q, device=env.device)

        # 5B. Sample orientation.
        q_aux = {}
        qs = sample_q(env_ids, num_reset, aux=q_aux)
        which_pose = q_aux['pose_index']

        # 5C. Sample XY.
        xys = [None for _ in range(len(cfg.scene_types))]
        if 'bump' in cfg.scene_types:
            assert (cfg.init_type == 'stable')
            xy_bump = sample_bump_xy(
                self.table_pos[env_ids],
                self.table_dims[env_ids],
                self.bump_width[env_ids],
                self.cur_props.foot_radius[env_ids, which_pose],
                self.bump_pos[env_ids]
            )
            xys[cfg.scene_types.index('bump')] = xy_bump

        if 'step' in cfg.scene_types:
            assert (cfg.init_type == 'stable')
            side = None
            if cfg.hack_climb_only:
                side = 0
            xy_step = sample_bump_xy(
                self.table_pos[env_ids],
                self.table_dims[env_ids],
                (0.0 * self.bump_width[env_ids] +
                 self.cur_props.barrier_height[env_ids] * math.tan(cfg.hack_step_tilt)),
                self.cur_props.foot_radius[env_ids, which_pose],
                self.step_pos[env_ids],
                side=side
            )
            xys[cfg.scene_types.index('step')] = xy_step

        if 'flat' in cfg.scene_types:
            # franka
            body_tensors = env.tensors['body']
            hand_ids = env.robot.ee_body_indices.long()
            eef_pose = body_tensors[hand_ids, :]
            keepout_center = eef_pose[env_ids, ..., :2]
            # FIXME(ycho): assumes `robot_radius` is scalar
            keepout_radius = env.robot.robot_radius

            xy_flat = sample_flat_xy(
                self.table_pos[env_ids],
                self.table_dims[env_ids],
                self.cur_props.foot_radius[env_ids, which_pose],
                num_reset,
                env.device,
                th.float32,
                keepout_center,
                keepout_radius,
                cfg.prevent_fall,
                cfg.avoid_overlap,
                cfg.margin_scale,
                min(self._pos_scale, 1.0)
            )
            xys[cfg.scene_types.index('flat')] = xy_flat

        if 'wall' in cfg.scene_types:
            # franka
            body_tensors = env.tensors['body']
            hand_ids = env.robot.ee_body_indices.long()
            eef_pose = body_tensors[hand_ids, :]
            keepout_center = eef_pose[env_ids, ..., :2]
            # FIXME(ycho): assumes `robot_radius` is scalar
            keepout_radius = env.robot.robot_radius

            xy_wall = sample_flat_xy(
                self.table_pos[env_ids],
                self.table_dims[env_ids] - 2.0 * self.wall_width[env_ids, ..., None],
                self.cur_props.foot_radius[env_ids, which_pose],
                num_reset,
                env.device,
                th.float32,
                keepout_center,
                keepout_radius,
                cfg.prevent_fall,
                cfg.avoid_overlap,
                cfg.margin_scale,
                min(self._pos_scale, 1.0)
            )
            xys[cfg.scene_types.index('wall')] = xy_wall

        if 'cabinet' in cfg.scene_types:
            # franka
            body_tensors = env.tensors['body']
            hand_ids = env.robot.ee_body_indices.long()
            eef_pose = body_tensors[hand_ids, :]
            keepout_center = eef_pose[env_ids, ..., :2]
            # FIXME(ycho): assumes `robot_radius` is scalar
            keepout_radius = env.robot.robot_radius

            xy_cabinet = sample_flat_xy(
                self.table_pos[env_ids],
                self.table_dims[env_ids] - 2.0 * self.wall_width[env_ids, ..., None],
                self.cur_props.foot_radius[env_ids, which_pose],
                num_reset,
                env.device,
                th.float32,
                keepout_center,
                keepout_radius,
                cfg.prevent_fall,
                cfg.avoid_overlap,
                cfg.margin_scale,
                min(self._pos_scale, 1.0)
            )
            xys[cfg.scene_types.index('cabinet')] = xy_cabinet

        xy = th.take_along_dim(
            th.stack(xys, dim=0),  # T, R, 2
            self.scene_type[None, env_ids, None],  # 1, R, 1
            dim=0).squeeze(0)

        # 5D. Sample z from `q`.
        # Compute `z` from `q`.
        # TODO(ycho): in the case of bump/wall,
        # may require access to (x, y)
        # alternatively we can just modify `z` maybe
        # [2] Reset nxt objects' poses so that
        # the convex hull rests immediately on the tabletop surface.
        # NOTE(ycho): Only activated if init_type is not stable,
        # in which we need to use the z value from precomputed
        # stable poses.
        table_height = self.table_pos[env_ids, 2] + 0.5 * cfg.table_dims[2]

        z = z_from_q_and_hull(
            qs,
            self.cur_props.hulls[env_ids],
            table_height) + cfg.z_eps

        # +++++HACK+++++
        if 'step' in cfg.scene_types:
            is_step = (self.scene_type[env_ids]
                       == cfg.scene_types.index('step'))
            high_mask = th.logical_and(
                is_step,
                (xy[..., 1] > (self.table_pos[env_ids, 1]
                               + self.step_pos[env_ids]))
            )
            z += high_mask * self.cur_props.barrier_height[env_ids]

        # 5E. Apply z-offset based on terrain height.
        # primarily applicable for the "step" domain
        # z += self.terrain_height(self.scene_types, self.scene_params?, xy,
        # qs?)

        root_tensor[ni, :2] = xy
        root_tensor[ni, 2] = z
        root_tensor[ni, 3:7] = qs

    @nvtx.annotate('Scene.reset()', color="red")
    def reset(self, gym, sim, env,
              env_ids: Optional[Iterable[int]] = None) -> th.Tensor:
        """
        Current reset logic roughly based on:
            https://forums.developer.nvidia.com/t/
            would-it-be-possible-to-destroy-an-actor-
            and-add-a-new-one-during-simulation/169517/2

        What happens during this reset script?
        1. select the object to move to table.
        2. apply domain randomization on target object.
        3. apply camera-pose randomization.
        4. reset (prv/cur) object poses.
        5. Bookkeeping for indices...
        """
        cfg = self.cfg
        # Reset object poses and potentially
        # apply domain randomization.
        # set_actor_rigid_body_properties()
        # set_actor_rigid_shape_properties()
        # set_actor_root_state_tensor_indexed()

        with nvtx.annotate("a"):
            reset_all: bool = False
            if env_ids is None:
                env_ids = th.arange(env.num_env, device=env.device)
                reset_all = True
            num_reset: int = len(env_ids)

        # == CACHE PREV IDS.
        if reset_all:
            if self.cur_props.ids is not None:
                prv_ids = self.cur_props.ids.clone()
            else:
                prv_ids = None
        else:
            prv_ids = self.cur_props.ids[env_ids]

        nxt = self._reset_props(gym, sim, env, env_ids, num_reset, reset_all)
        self._reset_poses(gym, sim, env,
                          env_ids, num_reset, reset_all,
                          prv_ids, nxt)

        # Also update bump/table/barrier... height
        root_tensor = env.tensors['root']
        barrier_ids = self.barrier_ids[env_ids.long()]
        if 'flat' in cfg.scene_types:
            is_flat = (self.scene_type[env_ids] ==
                       cfg.scene_types.index('flat'))
            barrier_height = ((~is_flat).float() *
                              self.cur_props.barrier_height[env_ids])
        else:
            barrier_height = self.cur_props.barrier_height[env_ids]
        root_tensor[barrier_ids, ..., 2] = (
            (self.table_pos[..., 2] - self.table_dims[..., 2])[env_ids]
            + barrier_height
        )

        if 'step' in cfg.scene_types:
            is_step = (self.scene_type[env_ids] ==
                       cfg.scene_types.index('step'))
            # NOTE(ycho):
            # the reason we're plugging in `cfg.table_dims[2]`
            # is because that's the default-set step height
            # for whatever reason...
            # This is potentially quite dangerous and somewhat dumb.
            barrier_dy = (
                is_step.float() * 0.5 *
                (self.cur_props.barrier_height[env_ids] - cfg.table_dims[2]) *
                math.tan(cfg.hack_step_tilt))
            root_tensor[barrier_ids, ..., 1] = barrier_dy

        pi = None if (prv_ids is None) else prv_ids.long()
        ni = nxt['ids'].long()
        with nvtx.annotate("g"):
            # merge pi, ni
            if (cfg.num_obj_per_env > 1):
                mask = self.mask
                mask.fill_(0)
                if (prv_ids is not None):
                    mask[pi] = 1
                mask[ni] = 1
                set_ids = th.argwhere(mask).ravel().to(
                    dtype=th.int32)
            else:
                set_ids = ni.to(dtype=th.int32)

        if True:
            # sort?
            set_ids = th.cat([set_ids, barrier_ids], dim=-1)

        with nvtx.annotate("i"):
            return set_ids

    def create_actors(self, gym, sim, env,
                      env_id: int):
        cfg = self.cfg

        # Sample N objects from the pool.

        # Spawn table.
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(*cfg.table_pos)
        table_pose.r = gymapi.Quat(*cfg.table_orn)

        table_key = self.table_keys[env_id]
        table_actor = gym.create_actor(
            env,
            # self.assets['table'][table_key],
            self.__flat_asset,
            table_pose, F'table', env_id,
            0b0001
        )

        barrier_pose = gymapi.Transform()
        barrier_pose.p = table_pose.p
        barrier_pose.p.z -= cfg.table_dims[2]
        barrier_pose.r = table_pose.r
        barrier_actor = gym.create_actor(
            env,
            self.assets['table'][table_key],
            barrier_pose, F'barrier', env_id,
            0b0001
        )

        for actor in [table_actor, barrier_actor]:
            shape_props = gym.get_actor_rigid_shape_properties(
                env, actor)
            for p in shape_props:
                p.filter = 0b0001
            gym.set_actor_rigid_shape_properties(env, actor, shape_props)
            if cfg.table_friction is not None:
                set_actor_friction(gym, env, actor, cfg.table_friction)

        object_actors = []

        keys = self.keys[env_id * cfg.num_obj_per_env:]
        for i, key in enumerate(keys[:cfg.num_obj_per_env]):
            obj_pose = gymapi.Transform()

            # Spawn objects.
            # -1: load from asset; 0: enable self-collision; >0: disable self-collision
            obj_asset = self.assets['objects'][key]
            body_count = gym.get_asset_rigid_body_count(obj_asset)
            shape_count = gym.get_asset_rigid_shape_count(obj_asset)
            with aggregate(gym, env,
                           body_count,
                           shape_count,
                           False,
                           use=False):
                object_actor = gym.create_actor(
                    env,
                    obj_asset,
                    obj_pose,
                    F'object-{i:02d}',
                    env_id,
                    0b0010
                )

            if True:
                shape_props = gym.get_actor_rigid_shape_properties(
                    env, object_actor)
                for p in shape_props:
                    p.filter = 0b0010
                gym.set_actor_rigid_shape_properties(
                    env, object_actor, shape_props)

            if cfg.object_friction is not None:
                set_actor_friction(gym, env, object_actor,
                                   cfg.object_friction)
            gym.set_rigid_body_segmentation_id(env, object_actor,
                                               0, 1 + i)
            object_actors.append(object_actor)

        return {'table': table_actor,
                'object': object_actors,
                'barrier': barrier_actor}

    def create_assets(self, gym, sim, env: 'EnvBase',
                      counts: Optional[Dict[str, int]] = None
                      ):
        cfg = self.cfg

        # (1) Create table.
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False

        total_body_count: int = 0
        total_shape_count: int = 0

        # (2) (pre-)configure table params
        self.bump_width = randu(cfg.bump_width_bound[0],
                                cfg.bump_width_bound[1],
                                size=(env.num_env,),
                                device=env.device)
        # for now, alias wall_width == bump_width
        if cfg.enable_climb_wall:
            assert (cfg.bump_width_bound[0] >= 0.2)
        self.wall_width = 0.5 * self.bump_width

        self.bump_pos = randu_like(
            - 0.5 * cfg.table_dims[1] + 0.5 * self.bump_width + 0.1,
            + 0.5 * cfg.table_dims[1] - 0.5 * self.bump_width - 0.1,
            self.bump_width)
        self.step_pos = randu(
            -0.5 * cfg.table_dims[1] + 0.1,
            +0.5 * cfg.table_dims[1] - 0.1,
            # 0.0 + cfg.bump_width_bound[0],
            # cfg.table_dims[1] - cfg.bump_width_bound[0],
            size=(env.num_env,),
            device=env.device)
        #
        self.cabinet_height = randu(cfg.cabinet_height_bound[0],
                                    cfg.cabinet_height_bound[1],
                                    size=(env.num_env,),
                                    device=env.device)

        # FIXME(ycho) HACK HACK HACK!
        # Unfortunately, we __CANNOT__ set this to (0,0,0)
        # (i.e. "remove" the dummy base_link)
        # without encountering isaac gym issues.
        if True:
            barrier_base_dims = (0.01, 0.01, 0.01)

        load = {
            'flat': (
                lambda i: _load_flat_asset(
                    gym,
                    sim,
                    cfg.table_dims)),
            'bump': (
                lambda i: _load_bump_asset(
                    gym,
                    sim,
                    cfg.table_dims,
                    self.bump_width[i],
                    cfg.table_dims[2],
                    self.bump_pos[i],
                    base_dims=barrier_base_dims)),
            'wall': (
                lambda i: _load_wall_asset(
                    gym,
                    sim,
                    cfg.table_dims,
                    self.wall_width[i],
                    cfg.table_dims[2],
                    base_dims=barrier_base_dims)),
            'step': (
                lambda i: _load_step_asset(
                    gym,
                    sim,
                    cfg.table_dims,
                    self.step_pos[i],
                    cfg.table_dims[2],
                    base_dims=barrier_base_dims,
                    step_tilt=cfg.hack_step_tilt)),
            'cabinet': (
                lambda i: _load_cabinet_asset(
                    gym,
                    sim,
                    cfg.table_dims,
                    # for now 0.3 is hardcorded
                    cfg.table_dims[2] + 0.3,
                    # for now let's assume it share wall width with wall doamin
                    self.wall_width[i],
                    base_dims=barrier_base_dims))}

        if cfg.hack_balance_types:
            # FIXME(ycho): hard-coded balancing ratio...
            p = [0 for _ in range(len(cfg.scene_types))]
            p[cfg.scene_types.index('flat')] = 0.1
            p[cfg.scene_types.index('wall')] = 0.1
            p[cfg.scene_types.index('step')] = 0.4
            p[cfg.scene_types.index('bump')] = 0.4
            table_types = np.random.choice(len(cfg.scene_types),
                                           size=env.num_env,
                                           replace=True,
                                           p=p)
            ic(np.unique(table_types,
                         return_counts=True))
        else:
            table_types = np.random.choice(len(cfg.scene_types),
                                           size=env.num_env,
                                           replace=True)

        self.__flat_asset, _ = _load_flat_asset(
            gym, sim, cfg.table_dims)

        table_assets = OrderedDict()
        table_clouds = OrderedDict()
        for i in range(env.num_env):
            table_assets[i], table_clouds[i] = load[cfg.scene_types
                                                    [table_types[i]]](i)
        self.table_clouds = th.as_tensor(
            np.stack([table_clouds[i] for i in range(env.num_env)], axis=0),
            dtype=th.float,
            device=env.device
        )

        total_body_count += max([
            gym.get_asset_rigid_body_count(a)
            for a in table_assets.values()]) + gym.get_asset_rigid_body_count(self.__flat_asset)
        total_shape_count += max([
            gym.get_asset_rigid_shape_count(a)
            for a in table_assets.values()
        ]) + gym.get_asset_rigid_shape_count(self.__flat_asset)

        # scene_types = np.random.choice(len(cfg.scene_types),
        #                                size=env.cfg.num_env)
        # table_asset = [base_assets[i] for i in scene_types]

        # (3) Create objects.
        # TODO(ycho): do we need to load a bunch of
        # diverse-ish objects ??
        # TODO(ycho): probably need to make either a
        # dummy URDF file or auto-generate URDF file
        # based on inertial and mass properties (e.g.
        # assuming density = 0.1kg/m^3, domain randomization,
        # ...)
        obj_assets = {}
        force_sensors = {}

        max_load: int = cfg.num_obj_per_env * env.cfg.num_env
        urdfs = [self.meta.urdf(k) for k in self.meta.keys()]

        num_obj = min(max_load, len(urdfs), cfg.num_object_types)
        if cfg.mode == 'train':
            object_files = np.random.choice(
                urdfs,
                size=num_obj,
                replace=False
            )
        elif cfg.mode == 'valid':
            # Deterministic and ordered list of object_files
            object_files = list(itertools.islice(
                itertools.cycle(urdfs),
                num_obj))
        else:
            raise KeyError(F'Unknown mode = {cfg.mode}')

        max_obj_body_count: int = 0
        max_obj_shape_count: int = 0

        for index, filename in enumerate(
                tqdm(object_files, desc='create_object_assets')):

            # FIXME(ycho): relies on string parsing
            # to identify the key for the specific
            # URDF file
            # key = filename
            key = Path(filename).stem

            if key in obj_assets:
                continue

            asset_options = gymapi.AssetOptions()
            asset_options.disable_gravity = False
            asset_options.fix_base_link = False
            # FIXME(ycho): hardcoded `thickness`
            asset_options.thickness = 0.001

            # asset_options.linear_damping = 1.0
            # asset_options.angular_damping = 1.0

            if filename == 'cube':
                # CRM cube
                asset_options.density = cfg.density
                obj_asset = gym.create_box(sim,
                                           0.09, 0.09, 0.09,
                                           asset_options)
            else:
                if cfg.override_inertia:
                    asset_options.override_com = True
                    asset_options.override_inertia = True
                    if cfg.target_mass is None:
                        asset_options.density = cfg.density
                    else:
                        idx = 1 if cfg.load_convex else 0
                        volume = self.volume[key][idx]
                        asset_options.density = cfg.target_mass / volume
                else:
                    asset_options.override_com = False
                    asset_options.override_inertia = False
                # NOTE(ycho): set to `True` since we're directly using
                # the convex decomposition result from CoACD.
                asset_options.vhacd_enabled = False
                if cfg.load_convex:
                    asset_options.convex_decomposition_from_submeshes = False
                else:
                    asset_options.convex_decomposition_from_submeshes = True
                obj_asset = gym.load_urdf(sim,
                                          str(Path(filename).parent),
                                          str(Path(filename).name),
                                          asset_options)

            obj_body_count = gym.get_asset_rigid_body_count(obj_asset)
            obj_shape_count = gym.get_asset_rigid_shape_count(obj_asset)

            max_obj_body_count = max(max_obj_body_count,
                                     obj_body_count)
            max_obj_shape_count = max(max_obj_shape_count,
                                      obj_shape_count)

            # key = F'{index}-{filename}'
            obj_assets[key] = obj_asset

            # FIXME(ycho):
            # this might require the assumption that
            # the center of mass is located at the body origin.
            if cfg.add_force_sensor_to_com:
                props = gymapi.ForceSensorProperties()
                # FIXME(ycho): might be slightly unexpected
                props.use_world_frame = True

                # NOTE(ycho): should really be disabled !!!
                # props.enable_forward_dynamics_forces = False  # no gravity
                # props.enable_constraint_solver_forces = True  # contacts

                # FIXME(ycho): hardcoded `0` does not work
                # if the object is articulated
                force_sensor = gym.create_asset_force_sensor(
                    obj_asset, 0, gymapi.Transform(),
                    props)
                force_sensors[key] = force_sensor

        total_body_count += max_obj_body_count
        total_shape_count += max_obj_shape_count

        if counts is not None:
            counts['body'] = total_body_count
            counts['shape'] = total_shape_count

        # (2) Create objects.

        self.assets = {
            'table': table_assets,
            'objects': obj_assets,
            'force_sensors': force_sensors
        }

        self.keys = list(itertools.islice(
            itertools.cycle(list(obj_assets.keys())),
            env.cfg.num_env * cfg.num_obj_per_env))

        # NOTE(ycho):
        # the commented-out version is _probably_ fine,
        # but I'm using the below version "just in case".
        # self.table_keys = list(itertools.islice(
        #     itertools.cycle(list(table_assets.keys())),
        #     env.cfg.num_env))
        self.table_keys = list(table_assets.keys())

        convert = partial(_array_from_map,
                          self.keys,
                          dtype=th.float,
                          device=env.device)

        self.radii = convert({k: self.meta.radius(k) for k in self.keys})
        self.hulls = convert(_pad_hulls(
            {k: self.meta.hull(k) for k in self.keys}))
        self.scene_type = th.as_tensor(
            table_types,
            dtype=th.long,
            device=env.device
        )

        # FIXME(ycho): at some point, consider randomizing
        # table dimensions :)
        self.table_pos = einops.repeat(
            th.as_tensor(np.asarray(cfg.table_pos),
                         dtype=th.float,
                         device=env.device),
            '... -> n ...', n=env.num_env)
        self.table_dims = einops.repeat(
            th.as_tensor(np.asarray(cfg.table_dims),
                         dtype=th.float,
                         device=env.device),
            '... -> n ...', n=env.num_env)

        if cfg.goal_type == 'stable':
            stable_poses = {k: self.meta.pose(k) for k in self.keys}
            min_len = min([len(v) for v in stable_poses.values()])
            max_len = min(
                max([len(v) for v in stable_poses.values()]),
                cfg.truncate_pose_count)
            print(F'\tmin_len = {min_len}, max_len = {max_len}')

            def _pad(x: np.ndarray, max_len: int):
                if len(x) < max_len:
                    extra = max_len - len(x)
                    x = np.concatenate(
                        [x, x[np.random.choice(len(x), size=extra, replace=True)]],
                        axis=0)
                else:
                    x = x[np.random.choice(
                        len(x), size=max_len, replace=False)]
                return x
            stable_poses = {k: _pad(v[..., :7], max_len)
                            for k, v in stable_poses.items()}
            self.stable_poses = convert(stable_poses)

            # == additionally we pregenerate a bunch of validation poses ==
            if cfg.mode == 'valid':
                self._valid_poses = (self.stable_poses[:, :cfg.num_valid_poses]
                                     .detach().clone())
                self._valid_poses[..., 0:2] = self._get_xy(
                    env,
                    self._valid_poses.device,
                    self._valid_poses.dtype,
                    # self._valid_poses.shape[:-1],
                    (cfg.num_valid_poses, self.stable_poses.shape[0]),
                    env_ids=th.arange(env.num_env, device=env.device),
                    prevent_fall=False
                ).swapaxes(0, 1)

            is_yaw_only = None

            if cfg.use_yaw_only_logic:
                is_yaw_only = {
                    k: is_thin(self.meta.obb(k)[1], threshold=cfg.thin_threshold)
                    for k in self.keys
                }

            if cfg.yaw_only_key is not None:
                is_yaw_only = {
                    k: 1. if k in cfg.yaw_only_key else 0.
                    for k in self.keys
                }
                print(cfg.yaw_only_key)
                print(is_yaw_only)

            if cfg.yaw_only_file is not None:
                with open(cfg.filter_file, 'r') as fp:
                    yawonly_list = [str(s) for s in json.load(fp)]
                if is_yaw_only is not None:
                    for k in yawonly_list:
                        is_yaw_only[k] = 1.
                else:
                    is_yaw_only = {
                        k: 1. if k in yawonly_list else 0.
                        for k in self.keys
                    }

            if is_yaw_only is not None:
                self.is_yaw_only = convert(is_yaw_only).bool()
            else:
                self.is_yaw_only = th.zeros(len(self.keys), dtype=th.bool,
                                            device=env.device)

        if cfg.load_embedding:
            self.object_embeddings = convert(
                {k: self.meta.code(k) for k in self.keys})

        if cfg.load_bbox:
            self.bboxes = convert({k: self.meta.bbox(k) for k in self.keys})
        if cfg.load_obb:
            self.obbs = convert({k: self.meta.obb(k)[1] for k in self.keys})
        if cfg.load_cloud:
            self.cloud = convert({k: self.meta.cloud(k) for k in self.keys})
            if cfg.load_normal:
                self.normal = convert({k: self.meta.normal(k)
                                       for k in self.keys})
        if cfg.load_predefined_goal:
            self.predefined_goal = convert(
                {k: self.meta.predefined_goal(k) for k in self.keys})

        if cfg.load_stable_mask:
            num_obj = len(self.cloud)

            stable_masks_np = np.zeros(
                (num_obj, self.stable_poses.shape[1]),
                dtype=bool)

            for i in range(num_obj):
                cloud = dcn(self.cloud[i])
                for j in range(len(self.stable_poses[i])):
                    pose = dcn(self.stable_poses[i, j])
                    cloud_at_pose = tx.rotation.quaternion.rotate(
                        pose[None, 3:7],
                        cloud) + pose[None, 0:3]
                    stable_masks_np[i, j] = _is_stable(cloud_at_pose)

            self.stable_masks = th.as_tensor(stable_masks_np,
                                             dtype=bool,
                                             device=env.device)

        if cfg.load_foot_radius:
            num_obj = len(self.cloud)

            foot_radius_np = np.zeros(
                (num_obj, self.stable_poses.shape[1]), dtype=float)

            for i in range(num_obj):
                cloud = dcn(self.cloud[i])
                for j in range(len(self.stable_poses[i])):
                    pose = dcn(self.stable_poses[i, j])
                    cloud_at_orn = tx.rotation.quaternion.rotate(
                        pose[None, 3:7],
                        cloud)
                    foot_radius_np[i, j] = _foot_radius(
                        cloud_at_orn,
                        cfg.bump_height_bound[1]
                    )
            self.foot_radius = th.as_tensor(foot_radius_np,
                                            dtype=th.float,
                                            device=env.device)

        return self.assets

    def create_sensors(self, gym, sim, env, env_id: int):
        return {}


def main():
    scene = TableTopWithObjectScene(
        TableTopWithObjectScene.Config())


if __name__ == '__main__':
    main()
