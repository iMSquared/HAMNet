#!/usr/bin/env python3

from isaacgym import gymapi, gymutil

from typing import Tuple, Callable, Dict

import numpy as np
import torch as th

import open3d as o3d

from ham.env.env.iface import EnvIface
from ham.env.env.wrap.base import WrapperEnv
from ham.util.torch_util import dcn


class WireframeMeshGeometry(gymutil.LineGeometry):

    def __init__(self,
                 mesh=None,
                 pose=None,
                 color=None):

        if isinstance(mesh, str):
            mesh = o3d.io.read_triangle_mesh(mesh)
        lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        points = np.asarray(lineset.points)
        lines = np.asarray(lineset.lines)
        num_lines: int = len(lines)
        vs = points[lines]

        if color is None:
            color = (1, 0, 0)

        verts = np.empty((num_lines, 2),
                         gymapi.Vec3.dtype)
        for i in range(num_lines):
            verts[i][0] = tuple(vs[i, 0])
            verts[i][1] = tuple(vs[i, 1])
        colors = np.empty(num_lines, gymapi.Vec3.dtype)
        colors.fill(color)

        if pose is None:
            self.verts = verts
        else:
            self.verts = pose.transform_points(verts)

        self._colors = colors

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors


def scene_pose(env, _):
    root_tensor = env.tensors['root']
    barrier_ids = env.scene.barrier_ids
    poses = dcn(root_tensor[barrier_ids.ravel()])
    if env.scene.cfg.mesh_env:
        poses[..., :3] += np.asarray(env.scene.cfg.mesh_gen.origin_offset)
    return poses


class DrawMeshEdge(WrapperEnv):
    """
    Draw a copy of the same mesh edges,
    across all envs at different poses.
    """

    def __init__(self, env, check_viewer: bool = True,
                 mesh_file: str = '/tmp/docker/domain/duktig/coacd3.obj',
                 color: Tuple[float, float, float] = (0, 1, 0),
                 pose_fn: Callable[[EnvIface, Dict[str, th.Tensor]], th.Tensor] = scene_pose
                 ):
        super().__init__(env)
        self.check_viewer = check_viewer
        self.geom = WireframeMeshGeometry(mesh_file,
                                          color=color)
        self.pose = scene_pose

    def __draw(self, obs):
        if (self.check_viewer) and (self.viewer is None):
            return
        poses = self.pose(self, obs)
        for i, env in enumerate(self.envs):
            pose = gymapi.Transform(gymapi.Vec3(*poses[i, :3]),
                                    gymapi.Quat(*poses[i, 3:7]))
            gymutil.draw_lines(self.geom,
                               self.gym,
                               self.viewer,
                               env,
                               pose)

    def step(self, *args, **kwds):
        out = super().step(*args, **kwds)
        obs, rew, done, info = out
        self.__draw(obs)
        return out
