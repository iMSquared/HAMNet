#!/usr/bin/env python3

from typing import Optional, Dict, Tuple
from dataclasses import dataclass

import numpy as np
import torch as th
import torch.nn as nn
from ham.util.math_util import (quat_diff_rad,
                                quat_rotate,
                                apply_pose_tq
                                )


def pose_error(src_pos: th.Tensor,
               dst_pos: th.Tensor,
               src_orn: Optional[th.Tensor] = None,
               dst_orn: Optional[th.Tensor] = None,
               planar: bool = True,
               skip_orn: bool = False):
    ndim = (2 if planar else 3)
    pos_err = th.linalg.norm(dst_pos[..., :ndim] - src_pos[..., :ndim], dim=-1)
    if skip_orn:
        orn_err = th.zeros_like(pos_err)
    else:
        orn_err = th.abs(quat_diff_rad(dst_orn, src_orn))
    return (pos_err, orn_err)


def geodesic_step_keypoint_error(hull,
                                 y_step, z_step,
                                 src_pos: th.Tensor, dst_pos: th.Tensor,
                                 src_orn: th.Tensor, dst_orn: th.Tensor,
                                 planar: bool, fix: bool):
    """
    Geodesic keypoint error,
    only valid for `step` type scenes.
    """
    src_hull = quat_rotate(src_orn[..., None, :], hull) + src_pos[..., None, :]
    zmin = src_hull[..., 2].amin(dim=-1)
    ymax = src_hull[..., 1].amax(dim=-1)

    dy = (y_step - ymax).clamp_min(0)  # phase 1
    dz = (z_step - zmin).clamp_min(0)  # phase 2

    # phase 3
    src_pos = src_pos.clone()
    src_pos[..., 1].add_(dy)
    src_pos[..., 2].add_(dz)
    d3 = keypoint_error(hull, src_pos, dst_pos,
                        src_orn, dst_orn, planar=planar, fix=fix)
    return dy[..., None] + dz[..., None] + d3


def keypoint_error(kpt: th.Tensor,
                   src_pos: th.Tensor, dst_pos: th.Tensor,
                   src_orn: th.Tensor, dst_orn: th.Tensor,
                   planar: bool = True,
                   fix: bool = True):
    if planar:
        t_src = src_pos.clone()
        t_src[..., 2] = 0
        t_dst = dst_pos.clone()
        t_dst[..., 2] = 0
    else:
        t_src = src_pos
        t_dst = dst_pos
    q_src = src_orn[..., None, :]
    q_dst = dst_orn[..., None, :]
    src_kpt = quat_rotate(q_src, kpt) + t_src[..., None, :]
    dst_kpt = quat_rotate(q_dst, kpt) + t_dst[..., None, :]

    if fix:
        pos_err = th.linalg.norm(dst_kpt - src_kpt, dim=-1)
    else:
        pos_err = th.linalg.norm(dst_kpt - src_kpt, dim=-1).mean(dim=-1)
    return pos_err


def penetration_depth(
        has_ceil: th.Tensor,
        box_faces: th.Tensor,
        body_tensors: th.Tensor,
        cloud: Dict[str, th.Tensor],
        link_ids: Dict[str, int],
        reduce: bool = True
):
    # body_tensors = env.tensors['body']

    keys = list(cloud.keys())
    sizes = [cloud[k].shape[-2] for k in keys]
    index1 = np.cumsum(sizes)
    index0 = index1 - sizes
    i_hand = keys.index('link7')
    hand_range: Tuple[int, int] = [index0[i_hand], index1[i_hand]]

    robot_pcd = th.cat([
        apply_pose_tq(
            body_tensors[link_ids[k], ..., None, :7],
            cloud[k])
        for k in keys],
        dim=-2)
    # has_ceil = env.scene.cur_props.env_code[..., 18:19]
    # box_faces = env.scene.cur_props.env_faces
    offset = th.einsum(
        '...ij,...ij->...i',
        box_faces[..., : 3],
        box_faces[..., 3:])  # (N, num_box, 6)
    d = (th.einsum('...ik,...abk->...iab',
                   robot_pcd, box_faces[..., :3])
         - offset[:, None])  # (N, N_points, num_box, 6)
    d[..., -1:, :] = has_ceil[:, None, :, None] * d[..., -1:, :]
    depth = d.amax(-1).amin(-1).clamp_max(0)  # (N, N_points, num_box)

    full_depth = depth
    hand_depth = depth[..., hand_range[0]:hand_range[1]]
    if reduce:
        full_depth = full_depth.sum(-1)
        hand_depth = hand_depth.sum(-1)
        else_depth = full_depth - hand_depth
    else:
        # FIXME(ycho): hardcoded
        assert (hand_range[1] == depth.shape[-1])
        else_depth = depth[..., :hand_range[0]]
    return (hand_depth, else_depth)


class PointDistance(nn.Module):
    @dataclass
    class Config:
        pass

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(self, *,
                src_point: th.Tensor,
                dst_point: th.Tensor,
                **kwds):
        """
        kpts: [(B...), P, 3] canonical keypoints
        path: [(T...), (B...), 7] object pose
        goal: [(T...), (B...), 7] task goal (pose)
        """
        return th.linalg.norm(
            src_point - dst_point,
            dim=-1)[..., None]


class KeypointDistance(nn.Module):

    @dataclass
    class Config:
        planar: bool = False
        fix: bool = True

        def __post_init__(self):
            if self.planar:
                raise ValueError(F'self.planar={self.planar} deprecated.')
            if not self.fix:
                raise ValueError(F'self.fix={self.fix} deprecated.')

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(self, *,
                kpts: th.Tensor,
                path: th.Tensor,
                goal: th.Tensor, **kwds):
        """
        kpts: [(B...), P, 3] canonical keypoints
        path: [(T...), (B...), 7] object pose
        goal: [(T...), (B...), 7] task goal (pose)
        """
        cfg = self.cfg
        return keypoint_error(kpts[None],
                              path[..., 0:3],
                              goal[..., 0:3],
                              path[..., 3:7],
                              goal[..., 3:7],
                              planar=cfg.planar,
                              fix=cfg.fix)


class PoseDistance(nn.Module):

    @dataclass
    class Config:
        planar: bool = False

        def __post_init__(self):
            if self.planar:
                raise ValueError(F'self.planar={self.planar} deprecated.')

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, *, path: th.Tensor, goal: th.Tensor,
                **kwds):
        cfg = self.cfg
        pos_err, orn_err = pose_error(
            path[..., 0:3],
            goal[..., 0:3],
            path[..., 3:7],
            goal[..., 3:7],
            planar=cfg.planar,
            skip_orn=False,
        )
        return th.stack([pos_err, orn_err], dim=-1)
