#!/usr/bin/env python3

from typing import Iterable, Optional
from isaacgym import gymapi, gymtorch

import torch as th
import numpy as np
import itertools

import cv2
from icecream import ic

from ham.env.env.wrap.base import WrapperEnv
from ham.env.util import (draw_one_inertia_box, from_vec3, from_mat3,
                          draw_cloud_with_sphere,
                          draw_cloud_with_ray,
                          draw_cloud_with_cross,
                          )
from ham.util.torch_util import dcn
from ham.util.math_util import apply_pose_tq
from ham.env.env.wrap.normalize_env import NormalizeEnv


class DrawClouds(WrapperEnv):
    def __init__(self, env, check_viewer: bool = True,
                 cloud_key: str = 'cloud',
                 stride: int = 8,
                 radius: float = 0.005,
                 style: str = 'sphere',
                 draw_robot: Optional[str] = None):
        super().__init__(env)
        self.__stride = stride
        self.__radius = radius
        self.__key = cloud_key
        self.__style = style
        self.__draw_robot = draw_robot
        self.check_viewer = check_viewer

        CMAP_VIRIDIS = (
            cv2.applyColorMap(
                np.arange(255, dtype=np.uint8),
                colormap=cv2.COLORMAP_VIRIDIS)[..., :: -1] / 255.0)
        self.__colormap = th.as_tensor(CMAP_VIRIDIS,
                                       dtype=th.float32,
                                       device=self.device)

    def _apply_colmap(self, value, max_val=0.1):
        indices = ((th.abs(value) / max_val
                    * self.__colormap.shape[0])
                   .to(dtype=th.long)
                   .clamp_(0, self.__colormap.shape[0] - 1)
                   )
        return self.__colormap[indices]

    def __draw(self, obs):
        if (self.check_viewer) and (self.viewer is None):
            return

        # clouds = {'init': dcn(obs['object_state']), 'goal': dcn(obs['goal'])}

        nenv = self.unwrap(target=NormalizeEnv)
        if isinstance(nenv, NormalizeEnv):
            obs = {
                k: obs[k] for k in nenv.normalizer.obs_rms.keys()
            }
            nobs = nenv.normalizer.unnormalize_obs(obs)
        else:
            nobs = obs
        clouds = {'init': dcn(nobs[self.__key])}
        if 'goal' in nobs and nobs['goal'].shape[-1] == 3:
            clouds['goal'] = dcn(nobs['goal'])
        if 'partial_cloud' in nobs:
            clouds['partial'] = dcn(nobs['partial_cloud_1'])
        # if 'color_cloud' in nobs:
        #     clouds['color'] = dcn(nobs['color_cloud'])

        colors = {'init': (1.0, 0.0, 0.0),
                  'goal': (0.0, 0.0, 1.0),
                  'partial': (0.0, 1.0, 0.0),
                  }
        if self.__draw_robot:
            color_list = list(itertools.product([0, 1, 0.5], repeat=3))
            body_tensors = self.env.tensors['body']
            for c in colors.values():
                color_list.remove(c)
            if True:
                start_idx = 0
                for k, v in self.env.robot.cloud.items():
                    body_indices = self.env.robot.link_ids[k]
                    link_pose = body_tensors[body_indices, :]
                    link_pcd = apply_pose_tq(link_pose[..., None, :7],
                                             v)  # (N, N_points, 3)
                    n_env, n_cloud = link_pcd.shape[:2]
                    end_idx = start_idx + n_cloud
                    # (N, N_points)
                    is_col = self.env.task.is_col[:, start_idx:end_idx]
                    # (N, N_points)
                    depth = self.env.task.depth[:, start_idx:end_idx]
                    ncol_color = th.tensor(
                        [0, 0, 1],
                        dtype=th.float, device=self.device).expand(
                        *depth.shape, 3)
                    if self.__draw_robot == 'depth':
                        col_color = self._apply_colmap(depth).squeeze()
                    else:
                        col_color = th.tensor(
                            [0, 1, 0],
                            dtype=th.float, device=self.device).expand_as(
                            ncol_color)
                    cloud_color = th.where(
                        is_col[..., None].expand(*is_col.shape, 3),
                        col_color, ncol_color)
                    clouds[k] = dcn(link_pcd)
                    colors[k] = dcn(cloud_color)
                    start_idx = end_idx

        if self.__style == 'sphere':
            for k, v in clouds.items():
                cloud_color = None
                if k in colors:
                    cloud_color = np.asarray(colors[k])
                if v.shape[-1] == 6:
                    cloud_color = v[..., 3:6]
                for i, (c, e) in enumerate(zip(v, self.envs)):
                    if len(cloud_color.shape) > 2:
                        color = cloud_color[i][::self.__stride]
                    else:
                        color = cloud_color
                    draw_cloud_with_sphere(self.gym, self.viewer,
                                           c[::self.__stride, ..., :3],
                                           e,
                                           radius=self.__radius,
                                           color=color
                                           )
        elif self.__style == 'cross':
            for k, v in clouds.items():
                cloud_color = None
                if k in colors:
                    cloud_color = np.asarray(colors[k])
                if v.shape[-1] == 6:
                    cloud_color = v[..., 3:6]
                for i, (c, e) in enumerate(zip(v, self.envs)):
                    if len(cloud_color.shape) > 2:
                        color = cloud_color[i][::self.__stride]
                    else:
                        color = cloud_color
                    draw_cloud_with_cross(self.gym, self.viewer,
                                           c[::self.__stride, ..., :3],
                                           e,
                                           radius=self.__radius,
                                           color=color)
        elif self.__style == 'ray':
            raise ValueError('ray no longer supported.')
        else:
            raise ValueError(F'Unknown style = {self.__style}')

    def step(self, *args, **kwds):
        out = super().step(*args, **kwds)
        obs, rew, done, info = out
        self.__draw(obs)
        return out
