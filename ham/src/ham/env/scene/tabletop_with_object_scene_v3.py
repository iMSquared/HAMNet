#!/usr/bin/env python3

from typing import Tuple, Dict, Optional, Iterable
from dataclasses import dataclass, replace
from ham.util.config import recursive_replace_map
import pkg_resources
import os

import numpy as np
import itertools
from tempfile import mkdtemp
import nvtx

from isaacgym import gymapi

import torch as th


# Isaac Gym bases
from ham.env.env.base import EnvBase
from ham.env.scene.tabletop_scene import TableTopScene
from ham.env.common import aggregate

# Import for scene assets (geometries)
from ham.env.scene.dgn_object_set import DGNObjectSet
from ham.env.scene.mesh_object_set import MeshObjectSet
from ham.env.scene.gen_both import (BothGen, BaseGen, CaseGen)
from ham.env.scene.gen_mesh import MeshGen
from ham.env.scene.scene_util import load_obj_set

# Episode loading
# A: meta objects
from ham.env.episode.compose import Compose
from ham.env.episode.sample_buffer import SampleBuffer
# B: table
from ham.env.episode.table_dim import TableDim
from ham.env.episode.table_pos import TablePos
from ham.env.episode.table_asset import TableAsset
from ham.env.episode.plate_pose import PlatePose
from ham.env.episode.scene_cloud import SceneCloud
# C: object
from ham.env.episode.object_asset import ObjectAsset
from ham.env.episode.load_object import LoadObject
from ham.env.episode.scale_object import ScaleObject
from ham.env.episode.select_object import SelectObject
from ham.env.episode.lookup_prop import LookupProp
from ham.env.episode.phys_prop import PhysProp
from ham.env.episode.init_obj_pose import InitObjectPose
from ham.env.episode.init_obj_orn import InitObjectOrn
from ham.env.episode.goal_obj_pose import GoalObjectPose
# D: robot
from ham.env.episode.robot_pos import RobotPos
from ham.env.episode.robot_dof import RobotDof
from ham.env.episode.env_code import EnvCode
from ham.env.episode.col_free import CollisionFree
from ham.env.episode.sel_pose import SelPose


DATA_ROOT = os.getenv('HAM_DATA_ROOT', '/input')

CLOUD_SIZE: int = 4096  # max size of scene clouds (before subsampling)
BASE_SIZE: int = 11  # number of base(horizontal) plates
CASE_SIZE: int = 5  # number of case(wall,ceiling) plates


@dataclass
class TableTopWithObjectScene(TableTopScene):

    @dataclass
    class Config(TableTopScene.Config):
        asset_root: str = pkg_resources.resource_filename('ham.data', 'assets')

        data_root: str = F'{DATA_ROOT}/ACRONYM/urdf'
        # Convex hull for quickly computing initial placements.
        hull_root: str = F'{DATA_ROOT}/ACRONYM/hull'
        mesh_count: str = F'{DATA_ROOT}/ACRONYM/mesh_count.json'
        urdf_stats_file: str = F'{DATA_ROOT}/ACRONYM/urdf_stats.json'
        stable_poses_file: str = F'{DATA_ROOT}/ACRONYM/stable_poses.pkl'
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

        gen: BothGen.Config = BothGen.Config(
            base=BaseGen.Config(base_thickness=0.02,
                                wall_thickness=0.02),
            case=CaseGen.Config(thickness=0.02)
        )
        mesh_gen: MeshGen.Config = MeshGen.Config(
            vis_file='/tmp/docker/v2/bookshelf.obj',
            col_file='/tmp/docker/v2/bookshelf-coacd.obj',
            divide_goal=True
        )

        randomize_barrier_height: bool = True

        table_friction: Optional[float] = None
        object_friction: Optional[float] = None

        num_obj_per_env: int = 1
        z_eps: float = 1e-2  # =0.01

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
        dr_period: int = 1
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

        use_table_height_dr: bool = False
        min_table_height_offset: float = 0.0
        max_table_height_offset: float = 0.6
        use_table_dim_dr: bool = False
        use_table_pos_dr: bool = False

        use_com_dr: bool = False
        com_dr_scale: float = 0.01

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
        load_obj_height: bool = False

        default_mesh: str = 'RubiksCube_d7d3dc14748ec6d347cd142fcccd1cc2_8.634340549903529e-05.glb'

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
        # scenario: ScenarioObjectSet.Config = ScenarioObjectSet.Config()
        mesh: MeshObjectSet.Config = MeshObjectSet.Config()
        need_attr: Optional[Tuple[str, ...]] = ('num_verts',)

        mode: str = 'train'
        num_valid_poses: int = 1

        restitution: Optional[float] = None

        load_faces: bool = False
        case_thickness_offset: float = 0.0
        apply_guide_force: bool = False
        force_prop: Tuple[float, float, float] = (0., 0., 5.)
        high_bias: float = 1.0
        env_level: Optional[int] = None
        fuse_env: bool = False
        mesh_env: bool = False
        use_tight_ceil: bool = False

        # NOTE(ycho):
        # ad-hoc margin to ensure franka end-effector
        # can fit within the cabinet
        wrist_width: float = 0.1

        export_scene: Optional[str] = None
        import_scene: Optional[str] = None

        sample_curobo: bool = False
        use_cuda_graph: bool = False
        near_obj: bool = True

        load_episode: Optional[str] = None
        save_episode: Optional[str] = None
        tight_prior: float = 0.3
        qrand_prior: float = 0.1

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
            # FIXME(ycho): always true for now?
            self.load_foot_radius = True
            if self.use_tight_ceil:
                self.gen = recursive_replace_map(
                    self.gen, {'case.use_tight_ceil': True}
                )
            if self.use_tight_ceil:
                self.load_obj_height = True
            if self.mesh_env:
                self.fuse_env = True

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # cfg.filter_index,
        self.obj_set = load_obj_set(self)
        # self.gen = MeshGen(cfg.mesh_gen)
        self.gen = BothGen(cfg.gen)

        # Episode generator...
        self.gen_eps = Compose([
            # == configure scene (table) ==
            TableDim(),
            TablePos(),
            TableAsset(),
            PlatePose(),
            SceneCloud(),

            # == configure scene (object) ==
            ObjectAsset(),
            LoadObject(),
            ScaleObject(),
            PhysProp(),

            SelectObject(),  # -> ids
            LookupProp(),  # -> properties
            RobotPos(),
            InitObjectOrn(),
            InitObjectPose(),
            GoalObjectPose(),
            RobotDof(),
            EnvCode(),
            (CollisionFree()
             if cfg.sample_curobo
             else SelPose())
        ])

        if cfg.load_episode is not None:
            specs = [
                SampleBuffer(
                    self.gen_eps,
                    cfg.load_episode,
                    cfg.tight_prior,
                    cfg.qrand_prior
                ),
                # Re-add expensive
                # un-saved derivative quantites
                # on the fly.
                SceneCloud(),
                LookupProp()
            ]
            self.gen_eps = Compose(specs)

        # episode information
        self.data = {}
        self.prev_data = None

        table_base_dims = np.asarray(cfg.table_dims)
        # FIXME(ycho): ad-hoc hardcoded `0.5` coef
        table_dim_bound = np.stack([table_base_dims * 0.5,
                                    table_base_dims])
        # FIXME(ycho): hardcoded table position offset ranges
        table_pos_bound = (
            (0.0, -0.15, cfg.min_table_height_offset),
            (0.1, 0.15, cfg.max_table_height_offset)
        )
        table_pos_bound = np.asarray(table_pos_bound)

        # FIXME(ycho): separate
        # `ctx` object --> handles (like gen, tmpdir, ...)
        # and `cfg` object --> configurations (options).
        self.ctx = dict(
            # These things are unknown
            # and will be populated when we have
            # `env` input.
            num_env=None,
            device=None,

            gen=self.gen,
            tmpdir=mkdtemp(),
            num_obj_per_env=cfg.num_obj_per_env,
            obj_set=self.obj_set,
            num_object_types=cfg.num_object_types,
            max_pose_count=cfg.truncate_pose_count,
            obj_load_mode=cfg.mode,
            table_dim_bound=table_dim_bound,
            table_pos_bound=table_pos_bound,
            scale_bound=(cfg.min_scale, cfg.max_scale),
            obj_mass_bound=(cfg.max_mass, cfg.max_mass),
            obj_friction_bound=(cfg.min_object_friction,
                                cfg.max_object_friction),

            # FIXME(ycho): hardcoded away from `franka.py`
            # ...consider updating later when we have
            # env.robot as input?
            hand_friction_bound=(1.0, 1.5),

            table_friction_bound=(
                cfg.min_table_friction,
                cfg.max_table_friction),
            obj_restitution_bound=(cfg.min_object_restitution,
                                   cfg.max_object_restitution),

            override_inertia=cfg.override_inertia,
            density=cfg.density,
            load_convex=cfg.load_convex,
            target_mass=cfg.target_mass,

            dr_period=cfg.dr_period,
            use_mass_dr=cfg.use_dr_on_setup,
            use_scale_dr=cfg.use_scale_dr,
            use_com_dr=cfg.use_com_dr,
            z_eps=cfg.z_eps,
            # FIXME(ycho): rename
            min_height=cfg.wrist_width,
            # FIXME(ycho): add to cfg
            cloud_size=CLOUD_SIZE,
            # FIXME(ycho): maybe avoid ad-hoc calculation
            max_ws_height=(cfg.gen.case.max_height
                           + cfg.gen.base.max_height),
            high_bias=cfg.high_bias,
            load_foot_radius=cfg.load_foot_radius,
            max_vertical_obstacle_height=(
                max(cfg.gen.base.max_height,
                    cfg.gen.case.max_wall_height)
            ),
            use_cuda_graph=cfg.use_cuda_graph,
            near_obj=cfg.near_obj
        )

        # self.keys: List[str] = []
        # self.assets: Dict[str, Any] = {}
        self.sensors = {}

        self.max_barrier_height: float = max(
            cfg.gen.base.max_height,
            cfg.gen.case.max_wall_height
        )
        self.env_level: Optional[int] = cfg.env_level
        self.high_bias = cfg.high_bias

        # DR + Curriculum related variables I guess...
        self.__dr_step = 0

    def export(self):
        pass

    def _lookup_indices(self, env):
        cfg = self.cfg
        num_env: int = env.num_env
        # lookup indices
        obj_ids = []
        obj_handles = []
        obj_body_ids = []
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

        # 16 = from base
        # 5  = from wall
        if cfg.fuse_env:
            barrier_handles = [[
                env.gym.find_actor_handle(env.envs[i], F'table')
            ]
                for i in range(env.cfg.num_env)]
            barrier_ids = th.as_tensor([
                [env.gym.find_actor_index(env.envs[i],
                                          F'table',
                                          gymapi.IndexDomain.DOMAIN_SIM
                                          )]
                for i in range(env.cfg.num_env)],
                dtype=th.long,
                device=env.cfg.th_device)  # N x 21
        else:
            barrier_handles = [[
                env.gym.find_actor_handle(env.envs[i], F'plate_{j:02d}')
                for j in range(BASE_SIZE + CASE_SIZE)]
                for i in range(env.cfg.num_env)]
            barrier_ids = th.as_tensor([
                [env.gym.find_actor_index(env.envs[i],
                                          F'plate_{j:02d}',
                                          gymapi.IndexDomain.DOMAIN_SIM
                                          )
                 for j in range(BASE_SIZE + CASE_SIZE)]
                for i in range(env.cfg.num_env)],
                dtype=th.long,
                device=env.cfg.th_device)  # N x 21

        # actor indices
        obj_ids = th.as_tensor(obj_ids,
                               dtype=th.int32,
                               device=env.cfg.th_device).reshape(
            env.cfg.num_env, cfg.num_obj_per_env)

        # actor handles
        obj_handles = th.as_tensor(obj_handles,
                                   dtype=th.int32,
                                   device=env.cfg.th_device).reshape(
            env.cfg.num_env, cfg.num_obj_per_env)
        obj_body_ids = th.as_tensor(obj_body_ids,
                                    dtype=th.long,
                                    device=env.cfg.th_device).reshape(
            env.cfg.num_env, cfg.num_obj_per_env)

        return (obj_ids, obj_handles, obj_body_ids,
                barrier_handles, barrier_ids)

    def setup(self, env: 'EnvBase'):
        # Setup/parse
        self.cfg
        self.tmpdir = None
        num_env: int = env.num_env

        # Lookup indices.
        # TODO(ycho): alternatively, wrap this logic
        # inside episode Spec(...) logic
        (self.obj_ids,
         self.obj_handles,
         self.obj_body_ids,
         self.barrier_handles,
         self.barrier_ids) = self._lookup_indices(env)

        self.ctx['obj_id'] = self.obj_ids
        self.ctx['obj_handle'] = self.obj_handles
        self.ctx['obj_body_id'] = self.obj_body_ids
        self.ctx['barrier_id'] = self.barrier_ids
        self.ctx['barrier_handle'] = self.barrier_handles
        self.ctx['envs'] = env.envs
        self.ctx['root_tensor'] = env.tensors['root']
        # FIXME(ycho): introspection
        self.ctx['dof_tensor'] = env.tensors['dof']
        self.ctx['box_min'] = env.robot.cfg.box_min
        self.ctx['box_max'] = env.robot.cfg.box_max

        # FIXME(ycho): HACK -- steal from `Franka`
        robot_cfg = env.robot.cfg
        self.ctx['robot_id'] = env.robot.indices
        self.ctx['keepout_radius'] = robot_cfg.keepout_radius
        self.ctx['base_height'] = robot_cfg.base_height
        self.ctx['hand_friction_bound'] = (robot_cfg.min_hand_friction,
                                           robot_cfg.max_hand_friction)

        # FIXME(ycho): HACK -- steal from `PushTask`
        task_cfg = env.task.cfg
        self.ctx['samples_per_rejection'] = task_cfg.samples_per_rejection
        self.ctx['goal_radius_bound'] = (task_cfg.goal_radius,
                                         task_cfg.goal_radius)

        self.ctx['goal_angle_bound'] = (task_cfg.goal_angle,
                                        task_cfg.goal_angle)

        # NOTE(ycho):
        # Delayed setup as a workaround to current spawning infra.
        # PhysProp apply() currently requires access to obj_ids et al.
        # To avoid this, we need to enable create_actor(...) invocations
        # inside apply_setup() which breaks abstractions.
        print('Apply delayed PhysProp setup...')
        for spec in self.gen_eps.find(
            (lambda s: isinstance(s, PhysProp)),
                recurse=True):
            self.data = spec.apply_setup(self.ctx, self.data)

        # Maintain a small buffer
        # for tracking which objects need to be reset.
        self._reset_mask_buf = th.zeros(
            env.tensors['root'].shape[0],
            dtype=bool,
            device=env.cfg.th_device
        )

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

        if env_ids is None:
            env_ids = th.arange(env.num_env, device=env.device)

        num_reset: int = len(env_ids)
        if num_reset <= 0:
            # This means there's nothing to reset
            return th.empty((0,),
                            dtype=th.long,
                            device=env.device)

        prv_ids = None
        if 'obj_id' in self.data:
            prv_ids = self.data['obj_id'][env_ids]

        self.data['reset_ids'] = env_ids
        self.data = self.gen_eps.sample_reset(self.ctx, self.data)
        self.data = self.gen_eps.apply_reset(self.ctx, self.data)

        cur_ids = self.data['obj_id'][env_ids]

        # NOTE(ycho): `table_pos` adjustment
        # may not work with `import_scene`
        pi = None if (prv_ids is None) else prv_ids.long()
        ni = cur_ids.long()

        with nvtx.annotate("g"):
            # merge pi, ni
            if (cfg.num_obj_per_env > 1):
                mask = self._reset_mask_buf
                mask.fill_(0)
                if (prv_ids is not None):
                    mask[pi] = 1
                mask[ni] = 1
                set_ids = th.argwhere(mask).ravel().to(
                    dtype=th.int32)
            else:
                set_ids = ni.to(dtype=th.int32)

        # TODO(ycho): is `sort` necessary or useful here?
        # set_ids = th.cat([set_ids, barrier_ids.ravel()], dim=-1)
        barrier_ids = self.barrier_ids[env_ids.long()]  # R x (16+5)
        set_ids = th.cat([barrier_ids.ravel(), set_ids], dim=-1)

        return set_ids

    def create_actors(self, gym, sim, env,
                      env_id: int):
        cfg = self.cfg

        # Spawn table.
        table_key = self.table_keys[env_id]

        # Would `aggregate` help?
        with aggregate(gym, env,
                       # FIXME(ycho): there may be len() check
                       # might be invalid (it's fine for now,
                       # since this code section is not used)
                       16,
                       16,
                       False,
                       use=False):
            barrier_pose = gymapi.Transform()
            if cfg.fuse_env:
                barrier_actors = [gym.create_actor(
                    env,
                    self.assets['table'][table_key],
                    barrier_pose,
                    F'table',
                    env_id,
                    0b0001
                )]
            else:
                barrier_actors = [gym.create_actor(
                    env,
                    self.assets['table'][table_key][i],
                    barrier_pose, F'plate_{i:02d}', env_id,
                    0b0001
                ) for i in range(len(self.assets['table'][table_key]))]

        for actor in barrier_actors:
            shape_props = gym.get_actor_rigid_shape_properties(
                env, actor)
            # FIXME(ycho): _very_ brittle string-parsing logic
            actor_key = gym.get_actor_name(env, actor).split('_')[-1]
            if cfg.fuse_env:
                # FIXME(ycho): technically a bug:
                # might be necessary to split the bodies?
                for p in shape_props:
                    p.filter = 0b0001
            else:
                # FIXME(ycho): _very_ brittle string-parsing logic
                if actor_key in [str(n) for n in range(11, 15)]:
                    for p in shape_props:
                        # collide with everything
                        p.filter = 0b1000
                        #               ^--- table
                        #              ^---- object
                        #             ^---- robot
                        #            ^----- wall
                else:
                    for p in shape_props:
                        # allow collision with table
                        p.filter = 0b1001
            gym.set_actor_rigid_shape_properties(env, actor, shape_props)

        # Sample N objects from the pool.
        object_actors = []
        keys = self.keys[env_id * cfg.num_obj_per_env:]
        for i, key in enumerate(keys[:cfg.num_obj_per_env]):
            obj_pose = gymapi.Transform()

            # Spawn objects.
            # -1: load from asset
            # 0: enable self-collision
            # >0: disable self-collision
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

            shape_props = gym.get_actor_rigid_shape_properties(
                env, object_actor)
            for p in shape_props:
                p.filter = 0b0010
            gym.set_actor_rigid_shape_properties(
                env, object_actor, shape_props)

            gym.set_rigid_body_segmentation_id(env, object_actor,
                                               0, 1 + i)
            object_actors.append(object_actor)

        return {'object': object_actors,
                'barrier': barrier_actors}

    def create_assets(self, gym, sim, env: 'EnvBase',
                      counts: Optional[Dict[str, int]] = None
                      ):
        cfg = self.cfg

        # NOTE(ycho): we call setup() here since this is
        # the first entry-point to `env.scene.setup(...)`.
        # related logic...
        self.ctx['num_env'] = env.num_env
        self.ctx['device'] = env.device
        self.ctx['gym'] = gym
        self.ctx['sim'] = sim
        self.data['reset_ids'] = th.arange(env.num_env,
                                           dtype=th.long,
                                           device=env.device)
        self.data = self.gen_eps.sample_setup(
            self.ctx,
            self.data
        )
        self.data = self.gen_eps.apply_setup(
            self.ctx,
            self.data
        )

        # (3) Create objects and tables.
        obj_assets = self.data['object_asset']
        table_assets = self.data['table_asset']

        # add body counts
        if counts is not None:
            if cfg.fuse_env:
                table_body_count = max([
                    gym.get_asset_rigid_body_count(a)
                    for a in table_assets.values()])
                table_shape_count = max([
                    gym.get_asset_rigid_shape_count(a)
                    for a in table_assets.values()])
            else:
                table_body_count = max([sum([
                    gym.get_asset_rigid_body_count(b)
                    for b in a])
                    for a in table_assets.values()])
                table_shape_count = max([sum([
                    gym.get_asset_rigid_shape_count(b)
                    for b in a])
                    for a in table_assets.values()
                ])
            max_obj_body_count = max([gym.get_asset_rigid_body_count(x)
                                      for x in obj_assets.values()])
            max_obj_shape_count = max([gym.get_asset_rigid_shape_count(x)
                                       for x in obj_assets.values()])
            total_body_count = (table_body_count
                                + max_obj_body_count)
            total_shape_count = (table_shape_count
                                 + max_obj_shape_count)
            counts['body'] = total_body_count
            counts['shape'] = total_shape_count

        # (2) Create objects.
        force_sensors = {}
        self.assets = {
            'table': table_assets,
            'objects': obj_assets,
            'force_sensors': force_sensors
        }

        self.keys = list(itertools.islice(
            itertools.cycle(list(obj_assets.keys())),
            env.cfg.num_env * cfg.num_obj_per_env))

        self.table_keys = list(itertools.islice(
            itertools.cycle(list(table_assets.keys())),
            self.ctx['num_env']))

        return self.assets

    def create_sensors(self, gym, sim, env, env_id: int):
        return {}


def main():
    scene = TableTopWithObjectScene(
        TableTopWithObjectScene.Config())


if __name__ == '__main__':
    main()
