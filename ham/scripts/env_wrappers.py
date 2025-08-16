#!/usr/bin/env python3

from isaacgym import (gymtorch, gymapi, gymutil)
from isaacgym.gymutil import (AxesGeometry, WireframeBBoxGeometry, draw_lines)

from typing import (Optional, Iterable, Callable,
                    Tuple, Dict, Union, Sequence)
from dataclasses import dataclass
from ham.util.config import ConfigBase, recursive_replace_map
from skimage.color import label2rgb

import pickle
from gym import spaces
import trimesh
import einops

import numpy as np
import torch as th
import torch.nn.functional as F

from pytorch3d.ops.points_alignment import iterative_closest_point
from pytorch3d.ops.knn import knn_points
from pytorch3d.ops.sample_farthest_points import sample_farthest_points

from ham.env.env.base import EnvBase, EnvIface
from ham.env.env.wrap.base import (
    WrapperEnv,
    ObservationWrapper,
    ActionWrapper,
    add_obs_field
)
from ham.env.env.wrap.monitor_env import MonitorEnv
from ham.env.env.wrap.normalize_env import NormalizeEnv
from ham.env.episode.sample_buffer import SampleBuffer

from ham.util.path import ensure_directory, get_path
from ham.util.torch_util import (
    dcn, dot, merge_shapes,
    randu, randu_like)
from ham.util.math_util import (
    matrix_from_quaternion,
    quaternion_from_matrix,
    quat_from_axa,

    apply_pose,
    compose_pose_tq,
    invert_pose_tq,
    apply_pose_tq,

    quat_rotate,
    quat_multiply,
    quat_inverse
)
from ham.env.util import draw_keypoints, draw_sphere
from functools import partial
from ham.env.scene.tabletop_with_object_scene import _array_from_map
from ham.train.ckpt import load_ckpt, last_ckpt

# FIXME(ycho): better import path
from ham.data.transforms.aff import get_gripper_mesh
from ham.data.transforms.col import point_triangle_distance
from ham.models.cloud.point_mae import (
    subsample
)

import nvtx
from icecream import ic

from omegaconf import OmegaConf
import sys

from ham.models.common import transfer
from ham.models.cloud.unicorn import MLPEncoder

def _copy_if(src: th.Tensor, dst: Optional[th.Tensor]) -> th.Tensor:
    if dst is None:
        return src.detach().clone()
    else:
        dst.copy_(src)
    return dst


def _top_crop(x: th.Tensor,
              z_min: th.Tensor,
              num_sample: int,
              eps: float = 1e-6
              ):
    is_top = (x[..., 2] >= z_min[..., None])
    indices = th.multinomial(is_top.float() + eps, num_sample,
                             replacement=True)
    return th.take_along_dim(x, indices[..., None], dim=-2)


def _local_crop(x: th.Tensor,
                center: th.Tensor,
                radius: th.Tensor,
                z_min: th.Tensor,
                num_sample: int,
                eps: float = 1e-6,
                project_2d: bool = True
                ):
    if project_2d:
        is_local = (th.linalg.norm(x[..., :2] - center[..., None, :2],
                                   dim=-1) <= radius)
    else:
        is_local = (th.linalg.norm(x - center[..., None, :],
                                   dim=-1) <= radius)
    is_local = is_local & (x[..., 2] >= z_min[..., None])
    indices = th.multinomial(is_local.float() + eps, num_sample,
                             replacement=True)
    return th.take_along_dim(x, indices[..., None], dim=-2)


class DrawPose(WrapperEnv):
    """
    Args:
        env: Base environment to wrap.
        eps: below this mass, boxes will not be drawn.
    """

    def __init__(self, env, pose_fn: Callable[[None], th.Tensor],
                 check_viewer: bool = True):
        super().__init__(env)
        self.check_viewer = check_viewer
        self.pose_fn = pose_fn

    def __draw(self):
        if (self.check_viewer) and (self.viewer is None):
            return
        poses = self.pose_fn()
        if poses is None:
            return

        gym = self.gym
        viewer = self.viewer
        for i in range(self.num_env):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*poses[i, 0:3])
            pose.r = gymapi.Quat(*poses[i, 3:7])
            geom = AxesGeometry(0.5, pose)
            draw_lines(geom, gym, viewer,
                       self.envs[i], None)

    def step(self, *args, **kwds):
        out = super().step(*args, **kwds)
        self.__draw()
        return out


class DrawGoalPose(DrawPose):
    def __init__(self, env, check_viewer: bool = True):
        super().__init__(env, self.__get_pose, check_viewer)

    def __get_pose(self):
        if self.task.goal.shape[-1] == 7:
            return self.task.goal
        else:
            return None


class DrawObjPose(DrawPose):
    def __init__(self, env, check_viewer: bool = True):
        super().__init__(env, self.__get_pose, check_viewer)

    def __get_pose(self):
        obj_ids = self.scene.data['obj_id'].long()
        return self.tensors['root'][obj_ids, :7]


class DrawDebugLines(WrapperEnv):
    @dataclass
    class Config(ConfigBase):
        draw_workspace: bool = False
        draw_wrench_target: bool = False
        draw_cube_action: bool = False

    def __init__(self, cfg: Config, env: EnvIface,
                 check_viewer: bool = True):
        super().__init__(env)
        self.check_viewer = check_viewer
        self.cfg = cfg
        self._prev_cube_action: th.Tensor = None

    def draw(self):
        cfg = self.cfg
        # FIXME(ycho): introspection
        if isinstance(self.env, EnvBase):
            dt: float = self.env.cfg.dt
        else:
            dt: float = self.env.unwrap(target=EnvBase).cfg.dt

        if (self.check_viewer) and (self.viewer is None):
            return

        if cfg.draw_workspace:
            wss = dcn(self.task.ws_bound)
            for i in range(self.num_env):
                ws = wss[i]
                box = WireframeBBoxGeometry(ws)
                draw_lines(box, self.gym, self.viewer,
                           self.envs[i], None)
            draw_lines(box, self.gym, self.viewer,
                       self.envs[0], None)

        if cfg.draw_wrench_target:
            object_ids = self.scene.data['obj_id']
            obj_pos = dcn(self.tensors['root'][object_ids.long(), :3])
            wrench = dcn(self._prev_wrench_target)
            k = dt / 0.17
            line = np.stack([obj_pos, obj_pos + k * wrench[..., :3]],
                            axis=-2)  # Num_env X 2 X 3
            line[..., 2] += 0.1
            for i in range(self.num_env):
                self.gym.add_lines(self.viewer,
                                   self.envs[i],
                                   1,
                                   line[None, i],
                                   np.asarray([1, 0, 1], dtype=np.float32)
                                   )

        if cfg.draw_cube_action:
            if self._prev_cube_action is not None:
                cube_state = self.tensors['root'][
                    self.robot.actor_ids.long()]
                cube_pos = dcn(cube_state[..., :3])

                wrench = dcn(self._prev_cube_action)
                k = dt / 0.17
                line = np.stack([cube_pos, cube_pos + k * wrench[..., :3]],
                                axis=-2)  # Num_env X 2 X 3
                line[..., 2] += 0.1
                for i in range(self.num_env):
                    self.gym.add_lines(
                        self.viewer, self.envs[i],
                        1, line[None, i],
                        np.asarray([1, 0, 1],
                                   dtype=np.float32))

    def step(self, action):
        self.draw()
        out = super().step(action)
        self._prev_cube_action = action
        return out


class DrawTargetPose(DrawPose):
    def __init__(self, env, check_viewer: bool = True):
        super().__init__(env, self.__get_pose, check_viewer)

    def __get_pose(self):
        # if isinstance(self.robot.pose_error, CartesianControlError):
        target = None
        if self.robot.cfg.ctrl_mode in ['osc', 'CI']:
            target = self.robot.pose_error.target
        else:
            target = self.robot.pose_error.pose_error.target
        return target


class DrawPosBound(WrapperEnv):
    def __init__(self, env, check_viewer: bool = True):
        super().__init__(env)
        self.check_viewer = check_viewer

    def __draw(self):
        if (self.check_viewer) and (self.viewer is None):
            return
        bounds = self.robot.pose_error.pos_bound
        if bounds is None:
            return
        if len(bounds.shape) == 3:
            for i in range(self.num_env):
                bound = bounds[i]
                box = WireframeBBoxGeometry(bound)
                draw_lines(box, self.gym, self.viewer,
                           self.envs[i], None)

    def step(self, action):
        self.__draw()
        return super().step(action)

class DrawPatchCenter(WrapperEnv):
    """
    Args:
        env: Base environment to wrap.
        eps: below this mass, boxes will not be drawn.
    """

    def __init__(self, env, check_viewer: bool = True):
        super().__init__(env)
        self.check_viewer = check_viewer
        self._net = None

    def register(self, net):
        self._net = net

    def __get_pose(self):
        obj_ids = self.scene.data['obj_id'].long()
        return self.tensors['root'][obj_ids, :7]

    def __draw(self):
        if (self.check_viewer) and (self.viewer is None):
            return
        if self.scene.patch_centers is None:
            return
        gym = self.gym
        viewer = self.viewer
        obj_pose = self.__get_pose()
        centers = 1.1 * self.scene.data['obj_patch_center']
        cur_centers = dcn(obj_pose[..., None, 0:3] +
                          quat_rotate(obj_pose[..., None, 3:7], centers))

        alpha = None
        if self._net is not None:
            if self._net._attn is not None:
                alpha = dcn(self._net._attn)
                patch_indices = alpha.argmax(axis=-1)
                head_colors = label2rgb(np.arange(patch_indices.shape[-1]))

        for index, env in enumerate(self.envs):
            if alpha is not None:
                if True:
                    for hi, pi in enumerate(patch_indices[index]):
                        draw_sphere(gym, viewer, env,
                                    pos=cur_centers[index, pi],
                                    color=tuple(head_colors[hi]),
                                    radius=0.005)
                else:
                    # aggregate attention on that
                    # specific patch across 4 heads
                    net_attn = alpha[index].max(axis=-2)
                    alpha_i = (net_attn / net_attn.max())
                    draw_keypoints(
                        gym,
                        viewer,
                        env,
                        cur_centers[index],
                        alpha=alpha_i,
                        min_alpha=0.9
                    )
            else:
                draw_keypoints(gym, viewer, env,
                               cur_centers[index])

    def step(self, *args, **kwds):
        out = super().step(*args, **kwds)
        self.__draw()
        return out

class DrawCom(WrapperEnv):
    """
    Args:
        env: Base environment to wrap.
        eps: below this mass, boxes will not be drawn.
    """

    def __init__(self, env):
        super().__init__(env)
        self.__coms = None

    def __get_coms(self):
        gym = self.gym
        coms = []
        actor_handles = dcn(self.scene.data['obj_handle'])
        for (env, actor_handle) in zip(self.envs, actor_handles):
            prop = gym.get_actor_rigid_body_properties(env, actor_handle)
            assert (len(prop) == 1)
            com = np.asarray([prop[0].com.x,
                              prop[0].com.y,
                              prop[0].com.z])
            coms.append(com)
        coms = np.stack(coms, axis=0)
        return th.as_tensor(coms,
                            dtype=th.float,
                            device=self.device)

    def __get_pos(self):
        obj_ids = self.scene.data['obj_id'].long()
        obp = self.tensors['root'][obj_ids, :7]
        if self.__coms is None:
            self.__coms = self.__get_coms()
        com = self.__coms
        g_com = apply_pose_tq(obp,
                              com)
        return g_com

    def __draw(self):
        if self.viewer is None:
            return
        gym = self.gym
        viewer = self.viewer
        com_pos = self.__get_pos()
        for index, env in enumerate(self.envs):
            draw_sphere(gym, viewer, env,
                        pos=com_pos[index],
                        radius=0.01,
                        color=(0, 0, 0)
                        )

    def step(self, *args, **kwds):
        self.__draw()
        out = super().step(*args, **kwds)
        return out


class AddObjectKeypoint(ObservationWrapper):
    """
    Add object keypoint to info.

    Args:
        env: Base environment to wrap.
    """

    def __init__(self,
                 env: EnvIface,
                 key: str = 'keypoint'):
        super().__init__(env, self._wrap_obs)
        obs_space, update_fn = add_obs_field(
            env.observation_space, key, spaces.Box(
                -float('inf'), +float('inf'), (24,)))
        self._obs_space = obs_space
        self._update_fn = update_fn

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        obj_ids = self.scene.data['obj_id'].long()
        obj_pose = self.tensors['root'][obj_ids, :7]
        bboxes = self.scene.data['obj_bbox']
        # new_bboxes = rotate_keypoint(bboxes, obj_pose)
        new_bboxes = (quat_rotate(obj_pose[..., None, 3:7], bboxes)
                      + obj_pose[..., None, 0:3])
        return self._update_fn(obs,
                               new_bboxes.reshape(-1, 24))


class AddObjectFullCloud(ObservationWrapper):
    """
    Add object point cloud to info.

    Args:
        env: Base environment to wrap.
    """

    def __init__(self,
                 env: EnvIface,
                 key: str = 'cloud',
                 goal_key: Optional[str] = None,
                 num_point: Optional[int] = 512,
                 hack_canonical: bool = False,
                 hack_color_cloud: bool = False
                 ):
        super().__init__(env, self._wrap_obs)

        # NOTE(ycho): for now, we do not support online sampling
        # or cloud-size configuration in general.
        # assert (self.scene.cloud.shape[-2] == num_point)
        obs_space, update_fn = add_obs_field(
            env.observation_space, key,
            spaces.Box(-float('inf'), +float('inf'), (num_point, 3))
        )

        update_color_fn = None
        if hack_color_cloud:
            obs_space, update_color_fn = add_obs_field(
                obs_space, 'color_cloud',
                spaces.Box(-float('inf'), +float('inf'), (num_point, 6))
            )

        update_goal_fn = None
        if goal_key is not None:
            point_dim: int = 6 if hack_color_cloud else 3
            ic(F'goal_key={goal_key}')
            obs_space, update_goal_fn = add_obs_field(
                obs_space,
                goal_key,
                spaces.Box(-float('inf'), +float('inf'), (num_point, point_dim))
            )
            ic(obs_space)

        self._obs_space = obs_space
        self._update_fn = update_fn
        self._update_color_fn = update_color_fn
        self._update_goal_fn = update_goal_fn
        self.__hack_canonical = hack_canonical

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        obj_ids = self.scene.data['obj_id'].long()
        obj_pose = self.tensors['root'][obj_ids, :7]
        base_cloud = self.scene.data['obj_cloud']  # Nx512x3

        # [1] Add object point cloud at object pose.
        if self.__hack_canonical:
            obj_cloud = base_cloud.clone()
            obj_cloud[..., 2] += 0.55
            # FIXME(ycho): hardcoded constant!!
            # ^^ this is the sampling average of unicorn
        else:
            obj_cloud = apply_pose(obj_pose[..., None, 3:7],
                                   obj_pose[..., None, 0:3],
                                   base_cloud[..., :3])
        out = self._update_fn(obs, obj_cloud)

        if self._update_color_fn is not None:
            col_cloud = th.cat([obj_cloud,
                                base_cloud[..., 3:]],
                               dim=-1)
            out = self._update_color_fn(out, col_cloud)

        # [2] Also add object point cloud at goal pose.
        if self._update_goal_fn is not None:
            goal_pose = self.task.goal
            goal_cloud = apply_pose(goal_pose[..., None, 3:7],
                                    goal_pose[..., None, 0:3],
                                    base_cloud[..., :3])
            if base_cloud.shape[-1] > 3:
                goal_cloud = th.cat([goal_cloud,
                                    base_cloud[..., 3:]],
                                    dim=-1)
            out = self._update_goal_fn(out, goal_cloud)

        return out

class AddSceneFullClouds(ObservationWrapper):
    """
    Add scene points to info.

    Args:
        env: Base environment to wrap.
    """

    @dataclass
    class Config:
        key_focal: str = 'focal_cloud'
        key_peripheral: str = 'peripheral_cloud'
        # 'hand_focal_cloud'
        key_hand_focal: Optional[str] = None

        # NOTE(ycho):
        # Should this be constant? or different
        # based on `focal`<->`peripheral`?
        num_point: Optional[int] = 512
        num_points: Optional[Dict[str, int]] = None
        # One of `abs`, `rel`,
        # `knn`, `plane`, `chamfer`
        crop_style: str = 'abs'
        rel_crop_scale: float = 1.25  # used for `rel`
        abs_crop_radius: float = 0.25  # used for `abs`
        save_cloud: bool = False
        project_2d: bool = False  # used for `rel`/`abs`
        # use_knn: bool = False
        # abs_crop: bool = False
        plane_cloud_size: int = 4096

    def __init__(self, cfg: Config, env: EnvIface):
        super().__init__(env, self._wrap_obs)
        self.cfg = cfg
        obs_space = env.observation_space
        update_fns = {}
        for key in [cfg.key_focal, cfg.key_peripheral,
                    cfg.key_hand_focal]:
            if key is None:
                continue
            obs_space, update_fns[key] = add_obs_field(
                obs_space, key,
                spaces.Box(-float('inf'), +float('inf'),
                           (self._num_point(key), 3))
            )

        self._obs_space = obs_space
        self._update_fns = update_fns
        self.__count = 0
        self.__kpts = None

        if cfg.key_hand_focal is not None:
            urdf_path = get_path(
                'assets/franka_description/robots/franka_panda_custom_v3.urdf'
            )
            m = get_gripper_mesh(cat=True,
                                 frame='panda_hand',
                                 urdf_path=urdf_path,
                                 links=['panda_link7',
                                        'panda_hand',
                                        'panda_leftfinger',
                                        'panda_rightfinger'])
            self.__hand_kpts, _ = trimesh.sample.sample_surface(
                m, 64)
            self.__hand_kpts = th.as_tensor(self.__hand_kpts,
                                            dtype=th.float,
                                            device=self.device)

        if cfg.save_cloud:
            self.hand_mesh = get_gripper_mesh(cat=True, frame='panda_hand')
            self.hand_pcd, _ = trimesh.sample.sample_surface(
                self.hand_mesh,
                cfg.num_point)
            self.hand_pcd = th.as_tensor(self.hand_pcd,
                                         dtype=th.float,
                                         device=self.device)

    def _num_point(self, k: str):
        cfg = self.cfg
        if k is None:
            return cfg.num_point
        if cfg.num_points is None:
            return cfg.num_point
        return cfg.num_points.get(k, cfg.num_point)

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        cfg = self.cfg

        # if True:
        #    # base cloud + wall cloud how?

        num_focal = self._num_point(cfg.key_focal)
        num_peripheral = self._num_point(cfg.key_peripheral)
        num_hand_focal = self._num_point(cfg.key_hand_focal)

        if True:
            # focal_cloud = th.randn((self.num_env, cfg.num_point, 3),
            #                       device=self.device)
            # peripheral_cloud = th.randn((self.num_env, cfg.num_point, 3),
            #                       device=self.device)
            peripheral_cloud = subsample(
                self.scene.data['scene_cloud'],
                num_peripheral
            )
            focal_cloud = peripheral_cloud

            # FIXME(ycho): `focal_cloud` is unused for now!
            assert (cfg.crop_style == 'chamfer')
            if self.__kpts is None:
                self.__kpts, _ = sample_farthest_points(
                    self.scene.data['obj_cloud'], K=64)

            # == chamfer-like logic ==
            obj_ids = self.scene.data['obj_id'].long()
            obj_pose = self.tensors['root'][obj_ids, :7]
            obj_cloud = apply_pose(obj_pose[..., None, 3:7],
                                   obj_pose[..., None, 0:3],
                                   self.__kpts[..., :3])
            with th.no_grad():
                dists = knn_points(
                    self.scene.data['scene_cloud'],
                    obj_cloud,
                    K=1,
                    return_nn=False,
                    return_sorted=False).dists.squeeze(dim=-1)
                # N, T
                focal_idx = th.argsort(dists, dim=-1)[..., :num_focal, None]
                focal_cloud = th.take_along_dim(
                    self.scene.data['scene_cloud'],
                    focal_idx,
                    dim=-2)

            if cfg.key_hand_focal is not None:
                body_tensors = self.tensors['body']
                body_indices = self.robot.ee_body_indices.long()
                hand_pose = body_tensors[body_indices, :]
                hand_kpts = apply_pose(hand_pose[..., None, 3:7],
                                       hand_pose[..., None, 0:3],
                                       self.__hand_kpts[None, ..., :3])
                with th.no_grad():
                    dists = knn_points(
                        self.scene.data['scene_cloud'],
                        hand_kpts,
                        K=1,
                        return_nn=False,
                        return_sorted=False).dists.squeeze(dim=-1)
                    # N, T
                    hand_focal_idx = th.argsort(
                        dists, dim=-1)[..., :num_focal, None]
                    hand_focal_cloud = th.take_along_dim(
                        self.scene.data['scene_cloud'],
                        hand_focal_idx,
                        dim=-2)
        else:
            # Generate points that belong to the table plane
            center = self.scene.table_pos.clone()
            center[..., 2] += 0.5 * self.scene.table_dims[..., 2]
            radius = 0.5 * self.scene.table_dims
            radius[..., 2] = 0.0
            plane_cloud = center[..., None, :] + radius[..., None, :] * th.empty(
                (self.num_env, cfg.plane_cloud_size, 3),
                dtype=th.float, device=self.device
            ).uniform_(-1.0, +1.0)

            if 'flat' in self.scene.cfg.scene_types:
                is_flat = (self.scene.scene_type ==
                           self.scene.cfg.scene_types.index('flat'))
                barrier_height = (
                    (~is_flat).float() *
                    self.scene.data['barrier_height'])
            else:
                barrier_height = self.scene.data['barrier_height']
            barrier_height = (barrier_height +
                              self.scene.table_pos[..., 2] -
                              self.scene.table_dims[..., 2])
            barrier_cloud = self.scene.table_clouds.clone()
            barrier_cloud[..., 2] += barrier_height[..., None]
            total_cloud = th.cat([barrier_cloud, plane_cloud], dim=-2)

            # Global cloud
            # Subsample from larger cloud then add plane
            z_min = (self.scene.table_pos[..., 2]
                     + 0.5 * self.scene.table_dims[..., 2]
                     - 0.01)

            peripheral_cloud = _top_crop(total_cloud,
                                         z_min,
                                         num_peripheral
                                         )

            # subsample(total_cloud, self.__num_point)
            # peripheral_cloud = th.where(
            #     th.rand(peripheral_cloud.shape[:-1],
            #             dtype=th.float,
            #             device=self.device) < 0.2,
            #     plane_cloud,
            #     peripheral_cloud
            # )

            # Local crop
            obj_ids = self.scene.data['obj_id'].long()
            obj_pose = self.tensors['root'][obj_ids, :7]

            if cfg.crop_style == 'chamfer':
                if self.__kpts is None:
                    self.__kpts, _ = sample_farthest_points(
                        self.scene.data['obj_cloud'], K=64)
                # == chamfer-like logic ==
                obj_cloud = apply_pose(obj_pose[..., None, 3:7],
                                       obj_pose[..., None, 0:3], self.__kpts)
                with th.no_grad():
                    dists = knn_points(
                        total_cloud,
                        obj_cloud,
                        K=1,
                        return_nn=False,
                        return_sorted=False).dists.squeeze(dim=-1)
                # N, T
                focal_cloud = th.take_along_dim(total_cloud, th.argsort(
                    dists, dim=-1)[..., :num_focal, None], dim=-2)
            elif cfg.crop_style == 'knn':
                idx = knn_points(
                    obj_pose[..., None, :3],
                    total_cloud,
                    K=num_focal,
                    return_nn=False,
                    return_sorted=False).idx.squeeze(dim=-2)
                focal_cloud = th.take_along_dim(total_cloud,
                                                idx[..., None],
                                                dim=-2)
            elif cfg.crop_style in ['abs', 'rel']:
                if cfg.crop_style == 'abs':
                    radius = cfg.abs_crop_radius
                else:
                    radius = (cfg.rel_crop_scale *
                              self.scene.data['obj_radius'][..., None])
                focal_cloud = _local_crop(
                    total_cloud, obj_pose[..., : 3],
                    radius, z_min, num_focal,
                    project_2d=cfg.project_2d)
            else:
                raise ValueError(F'Unknown crop style={cfg.crop_style}')

        obs = self._update_fns[cfg.key_focal](obs, focal_cloud)
        obs = self._update_fns[cfg.key_peripheral](obs, peripheral_cloud)
        if cfg.key_hand_focal is not None:
            obs = self._update_fns[cfg.key_hand_focal](obs, hand_focal_cloud)

        if cfg.save_cloud:
            body_tensors = self.tensors['body']
            body_indices = self.robot.ee_body_indices.long()
            hand_pose = body_tensors[body_indices, :]
            hand_cloud = apply_pose(hand_pose[..., None, 3:7],
                                    hand_pose[..., None, 0:3],
                                    self.hand_pcd[None])

            sav = dict(o_cloud=dcn(obs['cloud']),
                       l_cloud=dcn(obs['focal_cloud']),
                       g_cloud=dcn(obs['peripheral_cloud']),
                       h_cloud=dcn(hand_cloud))
            Path('/tmp/logcloud4').mkdir(
                parents=True,
                exist_ok=True)
            with open(F'/tmp/logcloud4/{self.__count:05d}.pkl', 'wb') as fp:
                pickle.dump(sav, fp)
            self.__count += 1

        return obs


class UnicornEmbedClouds(ObservationWrapper):
    @dataclass
    class Config:
        encoder: MLPEncoder.Config = MLPEncoder.Config()
        load_cfg: Optional[str] = None
        load_ckpt: Optional[str] = None
        scale_level: int = 0
        # obj_scale_level: Optional[int] = None
        scale_levels: Optional[Dict[str, int]] = None
        cache_z: bool = False
        keys: Tuple[str, ...] = ('cloud', 'focal_cloud', 'peripheral_cloud')
        log_fps_nn_idx: bool = False

    def __init__(self,
                 env: EnvIface,
                 cfg: Config):
        super().__init__(env, self._wrap_obs)
        self.cfg = cfg

        # Load config and create model
        model_cfg = cfg.encoder
        if cfg.load_cfg is not None:
            model_cfg = OmegaConf.structured(model_cfg)
            train_cfg = OmegaConf.load(cfg.load_cfg)
            model_cfg = OmegaConf.merge(model_cfg, (train_cfg.model.encoder))
            model_cfg: MLPEncoder.Config = OmegaConf.to_object(model_cfg)
        model = MLPEncoder(model_cfg).to(self.device)

        # Load weight
        assert (cfg.load_ckpt is not None)
        params = th.load(last_ckpt(cfg.load_ckpt),
                         map_location='cpu')
        self.query_token = None
        if 'query_token' in params['model']:
            # to extract global information
            self.query_token = th.as_tensor(
                params['model']['query_token'],
                dtype=th.float,
                device=self.device).detach().clone().requires_grad_(False)
        xfer_output = transfer(model, params['model'], prefix_map={
            'encoder.': '',
        }, strict=False, freeze=True)
        ic(xfer_output)
        model.eval()
        self.model: MLPEncoder = model

        obs_space = env.observation_space
        update_fns = {}
        for key in cfg.keys:
            embed_key = F'embed_{key}'
            num_token = obs_space[key].shape[-2] // model_cfg.patch_size
            if self.query_token is not None:
                num_token += 1
            obs_space, update_fns[(key, embed_key)] = add_obs_field(
                obs_space, embed_key,
                spaces.Box(-float('inf'), +float('inf'),
                           (num_token, model_cfg.model_dim))
            )
        self._obs_space = obs_space
        self._update_fns = update_fns
        self.__zs = {}
        self._aux = {}

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        cfg = self.cfg
        with th.inference_mode():

            kis=[]
            ufs=[]
            for (ki, ko), update in self._update_fns.items():
                if ki not in obs:
                    continue
                kis.append(ki)
                ufs.append(update)

            x = th.cat([obs[ki] for ki in kis], dim=0)
            if self.query_token is not None:
                z_ctx = self.query_token[None, None].expand(
                    *x.shape[:-2], 1, self.query_token.shape[-1])
            else:
                z_ctx = None

            scale_level = cfg.scale_level
            assert(cfg.scale_levels is None)
            z = self.model(x, z_ctx=z_ctx,
                            group_level=scale_level,
                            patch_level=scale_level)

            z = einops.rearrange(z,
                                   '(r b) ... -> r b ...',
                                   r = len(kis))
            z = th.unbind(z, dim=0)
            for (update, _z) in zip(ufs, z):
                obs=update(obs, _z)
        return obs



class AddPhysParams(ObservationWrapper):
    """
    Add physics parameters to observation.

    Args:
        env: Base environment to wrap.
    """

    def __init__(self, env: EnvIface,
                 key: str = 'phys_params',
                 min_mass: float = 0.0,
                 max_mass: float = 2.0,
                 min_friction: float = 0.0,
                 max_friction: float = 2.0,
                 min_restitution: float = 0.0,
                 max_restitution: float = 1.0,
                 add_com: bool = False,
                 ):
        super().__init__(env, self._wrap_obs)
        self.__masses = None

        # NOTE(ycho): `3` stands for
        # table friction, object friction, hand friction,
        # respsectively.
        lo = np.asarray([min_mass] + [min_friction] * 3 + [min_restitution],
                        dtype=np.float32)
        hi = np.asarray([max_mass] + [max_friction] * 3 + [max_restitution],
                        dtype=np.float32)
        dim_input = 5
        if add_com:
            lo = np.concatenate([lo, [-1.0] * 3])
            hi = np.concatenate([hi, [1.0] * 3])
            dim_input = 8
            self.__coms = None
        self.__add_com = add_com
        obs_space, update_fn = add_obs_field(
            env.observation_space, key, spaces.Box(lo, hi, (dim_input,))
        )
        self._obs_space = obs_space
        self._update_fn = update_fn

    def __get_masses(self):
        gym = self.gym
        masses = []
        actor_handles = dcn(self.scene.data['obj_handle'])
        for (env, actor_handle) in zip(self.envs, actor_handles):
            prop = gym.get_actor_rigid_body_properties(env, actor_handle)
            assert (len(prop) == 1)
            mass = prop[0].mass
            masses.append(mass)
        return th.as_tensor(masses,
                            dtype=th.float,
                            device=self.device)

    def __get_coms(self):
        gym = self.gym
        coms = []
        actor_handles = dcn(self.scene.data['obj_handle'])
        for (env, actor_handle) in zip(self.envs, actor_handles):
            prop = gym.get_actor_rigid_body_properties(env, actor_handle)
            assert (len(prop) == 1)
            com = np.asarray([prop[0].com.x,
                              prop[0].com.y,
                              prop[0].com.z])
            coms.append(com)
        coms = np.stack(coms, axis=0)
        return th.as_tensor(coms,
                            dtype=th.float,
                            device=self.device)

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        if self.__masses is None:
            self.__masses = self.__get_masses()

        mass = self.__masses[..., None]
        tbl_fr = self.scene.data['table_friction'][..., None]
        obj_fr = self.scene.data['obj_friction']
        hnd_fr = self.robot.cur_hand_friction[..., None]
        obj_rs = self.scene.data['obj_restitution']
        out = [mass, tbl_fr, obj_fr, hnd_fr, obj_rs]
        if self.__add_com:
            if self.__coms is None:
                self.__coms = self.__get_coms()
            obj_ids = self.scene.data['obj_id'].long()
            obp = self.tensors['root'][obj_ids, :7]
            com = apply_pose_tq(obp, self.__coms)
            out.append(com)
        phys = th.cat(out, dim=-1)
        return self._update_fn(obs, phys)

class AddPrevAction(ObservationWrapper):
    def __init__(self, env: EnvIface,
                 key: str = 'previous_action',
                 zero_out: bool = False):
        super().__init__(env, self._wrap_obs)
        self._zero_out = zero_out

        if isinstance(env.robot.action_space, spaces.Discrete):
            # By default, __prev_action will be stored as one-hot.
            act_obs_space = spaces.Box(0.0, 1.0, (env.robot.action_space.n,))
            self.__prev_action = th.zeros(
                (env.num_env, env.robot.action_space.n),
                dtype=th.float,
                device=env.device)
        else:
            act_obs_space = env.robot.action_space
            self.__prev_action = th.zeros(
                merge_shapes(env.num_env, env.robot.action_space.shape),
                dtype=th.float,
                device=env.device)

        obs_space, update_fn = add_obs_field(env.observation_space,
                                             key,
                                             act_obs_space)
        self._obs_space = obs_space
        self._update_fn = update_fn

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        if self.__prev_action is None:
            return obs
        # FIXME(ycho): we multiply by 0.0 only for compatibility
        if self._zero_out:
            return self._update_fn(obs, 0.0 * self.__prev_action)
        else:
            return self._update_fn(obs, self.__prev_action)

    def reset_indexed(self,
                      indices: Optional[Iterable[int]] = None):
        # FIXME(ycho): this will fail in case
        # `0` is a meaningful action in the domain.
        if indices is None:
            self.__prev_action.fill_(0)
        else:
            self.__prev_action.index_fill_(0, indices, 0)
        return super().reset_indexed(indices)

    def step(self, act):
        obs, rew, done, info = super().step(act)
        self.__prev_action[...] = act
        obs = self._wrap_obs(obs)
        return (obs, rew, done, info)

class RelGoal(ObservationWrapper):
    """
    Rewrite goal so that it's relative to the
    current pose of the object.
    """

    def __init__(self, env,
                 key_goal: str = 'goal',
                 use_6d: bool = True,
                 add_abs_goal: bool = False):
        super().__init__(env, self._wrap_obs)
        self.__key_goal = key_goal
        self.__use_6d = use_6d

        n: int = 9 if use_6d else 7
        obs_space, update_fn = add_obs_field(
            env.observation_space,
            'goal', spaces.Box(-1.0, 1.0, (n,))
        )
        self._update_fn = update_fn
        self._update_fn2 = None

        if add_abs_goal:
            obs_space, update_fn2 = add_obs_field(
                obs_space,
                'abs_goal', spaces.Box(-1.0, 1.0, (n,))
            )
            self._update_fn2 = update_fn2
        self._obs_space = obs_space

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        # T0 = obs[self.__key_obj]
        # T0 =
        obj_ids = self.scene.data['obj_id'].long()
        T0 = self.tensors['root'][obj_ids, :7]
        T1 = obs[self.__key_goal]
        # dq = quat_multiply(
        #     quat_inverse(T1[..., 3:7]),
        #     T0[..., 3:7])
        dq = quat_multiply(T1[..., 3:7], quat_inverse(T0[..., 3:7]))
        # dt = T1[..., 0:3] - quat_rotate(dq, T0[..., 0:3])
        dt = T1[..., 0:3] - T0[..., 0:3]
        if self.__use_6d:
            rot_mat = matrix_from_quaternion(dq)
            dq = rot_mat[..., :, :2].reshape(*rot_mat.shape[:-2], -1)
        dT = th.cat([dt, dq], dim=-1)
        if self._update_fn2 is None:
            return self._update_fn(obs, dT)
        else:
            obs = self._update_fn(obs, dT)
            gp = T1[..., :3]
            gq = T1[..., 3:7]
            if self.__use_6d:
                rot_mat = matrix_from_quaternion(T1)
                gq = rot_mat[..., :, :2].reshape(*rot_mat.shape[:-2], -1)
            g = th.cat([gp, gq], dim=-1)
            return self._update_fn2(obs, g)

class TuneDomainSampling(WrapperEnv):
    @dataclass
    class Config(ConfigBase):
        decay_factor: float = 0.5 # only for `decay_mode=success`
        log_period: int = 1024
        decay_mode: str = 'linear' # or success or mul
        p_tight_target: float = 5.0
        p_qrand_target: float = 10.0
        # param for linear schedule 
        total_scale_step: float = 2e6  
        n_step_per_update: int = 1024


    def __init__(self,
                 env: EnvIface,
                 cfg: Config):
        super().__init__(env)
        self.cfg = cfg

        scene = self.scene
        specs = scene.gen_eps.find(
            (lambda s: isinstance(s, SampleBuffer)),
            recurse=True)
        assert (len(specs) == 1)
        self.spec = specs[0]  # SampleBuffer
        self.__step = 0
        n_update = (cfg.total_scale_step
                    / cfg.n_step_per_update)
        if cfg.decay_mode == 'linear':
            self._tight_prior = scene.cfg.tight_prior
            self._p_tight_addend = ((cfg.p_tight_target
                                    - self._tight_prior)
                                    / n_update)
            self._qrand_prior = scene.cfg.qrand_prior
            self._p_qrand_addend = ((cfg.p_qrand_target
                                    - self._qrand_prior)
                                    / n_update)
            
        elif cfg.decay_mode == 'mul':
            self._tight_prior = scene.cfg.tight_prior
            self._p_tight_multiplier = np.power((cfg.p_tight_target
                                                 /self._tight_prior),
                                                1.0 / n_update)
            self._qrand_prior = scene.cfg.qrand_prior
            self._p_qrand_multiplier = np.power((cfg.p_qrand_target
                                                 /self._qrand_prior),
                                                1.0 / n_update)

    def step(self, actions: th.Tensor):
        cfg = self.cfg
        spec = self.spec
        obsn, rewd, done, info = super().step(actions)
        succ = info['success']

        # Which episodes are 'done'?
        if cfg.decay_mode == 'success':
            env_idx = th.argwhere(done)
            eps_idx = self.scene.data['buf_idx'][env_idx]
            p_sel = self.spec._p_sel
            p_sel[eps_idx, env_idx] = th.where(succ[env_idx],
                                            p_sel[eps_idx, env_idx] * cfg.decay_factor,
                                            p_sel[eps_idx, env_idx])
            if (self.__step % cfg.log_period == 0):
                step=self.__step
                if self.writer is not None:
                    self.writer.add_scalar(F'log/p_sel_min',
                                        p_sel[p_sel > 0].min().item(),
                                        global_step=step)
                    self.writer.add_scalar(F'log/p_sel_max',
                                        p_sel.max().item(),
                                        global_step=step)
        elif cfg.decay_mode == 'linear':
            if (self.__step % cfg.n_step_per_update == 0):
                self._qrand_prior += self._p_qrand_addend
                self._tight_prior += self._p_tight_addend
        elif cfg.decay_mode == 'mul':
            if (self.__step % cfg.n_step_per_update == 0):
                self._qrand_prior *= self._p_qrand_multiplier
                self._tight_prior *= self._p_tight_multiplier

        if cfg.decay_mode in ['linear', 'mul'] :
            if (self.__step % cfg.n_step_per_update == 0):
                self._qrand_prior = min(self._qrand_prior,
                                        cfg.p_qrand_target)
                self._tight_prior = min(self._tight_prior,
                                        cfg.p_tight_target)
                self.spec._setup_p_sel(self._tight_prior,
                                        self._qrand_prior) 
            if (self.__step % cfg.log_period) == 0:
                step=self.__step
                if self.writer is not None:
                    self.writer.add_scalar(F'log/p_tight',
                                        self._tight_prior,
                                        global_step=step)
                    self.writer.add_scalar(F'log/p_qrand',
                                        self._qrand_prior,
                                        global_step=step)

        self.__step += 1
        return (obsn, rewd, done, info)
