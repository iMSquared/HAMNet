#!/usr/bin/env python3

from typing import Tuple, List, Optional

import math
import numpy as np
import torch as th
import einops
from functools import partial

from ham.models.common import merge_shapes
from ham.util.torch_util import randu_like, randu, dcn
from ham.util.math_util import (quat_multiply,
                                quat_rotate,
                                quat_inverse,
                                random_quat,
                                rejection_sample)

from icecream import ic


def adjust_bound(lo, hi, scale=None, offset=None):
    center = 0.5 * (lo + hi)
    radius = 0.5 * (hi - lo)
    if scale is not None:
        radius.mul_(scale)
    if offset is not None:
        radius.add_(offset)
    lo, hi = (center - radius,
              center + radius)
    invalid = (lo >= hi)
    lo = th.where(invalid, center, lo)
    hi = th.where(invalid, center, hi)
    return lo, hi


def ray_box_intersection(
        origin: th.Tensor,
        raydir: th.Tensor,
        box: th.Tensor):
    c = origin
    d = raydir
    b = box  # .reshape(..., 2, 2)
    i = 1 / d
    s = (i < 0).to(dtype=th.long)
    txmax = (b[th.arange(b.shape[0]),
               1 - s[..., 0], 0] - c[..., 0]) * i[..., 0]
    tymax = (b[th.arange(b.shape[0]),
               1 - s[..., 1], 1] - c[..., 1]) * i[..., 1]
    tmax = th.minimum(tymax, txmax)
    return c + d * tmax[..., None]


def sample_yaw(n: int, dtype=th.float, device=None):
    h = th.empty((n,), dtype=dtype, device=device).uniform_(
        -th.pi / 2, +th.pi / 2)
    z = th.sin(h)
    w = th.cos(h)
    q_z = th.stack([0 * z, 0 * z, z, w], dim=-1)
    return q_z


def sample_cuboid(n: int, dtype=th.float, device=None):
    h = - th.randint(low=1, high=2, size=(n,),
                     dtype=dtype, device=device) * th.pi / 4
    z = th.sin(h)
    w = th.cos(h)
    q_z = th.stack([0 * z, z, 0 * z, w], dim=-1)
    return q_z


def sample_random_along_axis(n: int, dtype=th.float,
                             axis=th.tensor, device=None):
    h = th.empty((n,), dtype=dtype, device=device).uniform_(
        -th.pi / 2, +th.pi / 2)
    z = th.sin(h)[..., None] * axis
    w = th.cos(h)[..., None]
    q_z = th.cat([z, w], dim=-1)
    return q_z


def sample_box_xy(
        bound: th.Tensor,
        z: float,
        n: Optional[int] = None,
        out: Optional[th.Tensor] = None
):
    if n is None:
        n = bound.shape[0]
    scale = bound[..., 1, :] - bound[..., 0, :]
    point_xy = bound[..., 0, :] + scale * th.rand((n, bound.shape[-1]),
                                                  dtype=bound.dtype,
                                                  device=bound.device)
    if out is None:
        out = th.zeros((*point_xy.shape[:-1], 3),
                       dtype=bound.dtype,
                       device=bound.device)
    out[..., :2] = point_xy[..., :2]
    out[..., 2] = z
    return out


def sample_goal_xy(
        bound: th.Tensor,
        center: th.Tensor,
        radius: float,
        eps: float = 1e-6,
        out: Optional[th.Tensor] = None
):
    """
    Sample points outside of the goal radius.
    Unlike rejection sampling, this method is not iterative,
    but at the cost of a slightly distorted distribution
    and potentially infeasible goals outside of the table boundary
    due to projection.
    """
    scale = (bound[..., 1, :] - bound[..., 0, :])
    p = bound[..., 0, :] + scale * th.rand_like(center)
    d = p - center
    r = th.linalg.norm(d, dim=-1,
                       keepdim=True)
    raydir = d.div_(r + eps)
    dst = ray_box_intersection(center, raydir, bound)

    # Explicitly sampled points at the
    # exterior of the goal.
    ext = th.lerp(center + radius * raydir, dst,
                  th.rand_like(center[..., :1])
                  )
    if out is None:
        out = th.zeros((*center.shape[:-1], 3),
                       dtype=th.float,
                       device=center.device)
    out[..., :2] = th.where(((0 * (r > radius)).bool()), p, ext)
    return out


def sample_goal_xy_v2(
        bound: th.Tensor,
        center: th.Tensor,
        radius: float,
        eps: float = 1e-6,
        out: Optional[th.Tensor] = None,
        num_samples: int = 4):
    """
    Sample points outside of the goal radius.
    Non-iterative rejection sampling,
    via oversampling by a factor of `num_samples`.
    """
    scale = (bound[..., 1, :] - bound[..., 0, :])
    p = bound[..., 0, :] + scale * th.rand(
        ((num_samples,) + center.shape),
        dtype=center.dtype,
        device=center.device)
    d = p - center
    r2 = th.einsum('...i,...i->...', d, d)
    mask = (r2 > radius**2)  # 1=bad, 0=good
    if out is None:
        out = th.zeros((*center.shape[:-1], 2),
                       dtype=th.float,
                       device=center.device)
    out[..., :2] = p[
        th.argmax(mask.float(), dim=0),
        th.arange(center.shape[0])
    ]
    return out


def sample_flat_xy(table_pos: th.Tensor,
                   table_dim: th.Tensor,
                   obj_radius: th.Tensor,

                   n: int,
                   device,
                   dtype,

                   keepout_center: th.Tensor,
                   keepout_radius: th.Tensor,
                   prevent_fall: bool = False,
                   avoid_overlap: bool = False,
                   margin_scale: float = 1.0,
                   pos_scale: float = 1.0
                   ):
    shape = merge_shapes(n, 2)

    # == Compute sample bounds ==
    center = table_pos[..., :2]
    radius = 0.5 * margin_scale * table_dim[..., :2]
    lo = center - radius
    hi = center + radius

    # == Inset bounds by `radius` ==
    if prevent_fall:
        lo, hi = adjust_bound(lo, hi,
                              scale=pos_scale,
                              offset=-obj_radius[..., None])

    # == Prevent overlap ==
    if avoid_overlap:
        bound = th.stack([lo, hi], dim=-2)
        return sample_goal_xy_v2(bound,
                                 keepout_center,
                                 keepout_radius,
                                 num_samples=16)
    else:
        return randu(lo, hi, size=(n, 2),
                     dtype=dtype, device=device)


def sample_bump_xy(table_pos: th.Tensor,
                   table_dim: th.Tensor,
                   bump_width: float,
                   foot_radius: th.Tensor,
                   bump_pos: Optional[th.Tensor] = None,
                   side: Optional[bool] = None
                   ):
    x = table_pos[..., 0] + randu_like(
        -0.5 * table_dim[..., 0] + foot_radius,
        +0.5 * table_dim[..., 0] - foot_radius,
        table_pos[..., 0]
    )

    if bump_pos is None:
        y = randu_like(
            0.5 * bump_width + foot_radius,
            0.5 * table_dim[..., 1] - foot_radius,
            table_pos[..., 1]
        )
        y *= th.sign(th.randn_like(y))
        y += table_pos[..., 1]
    else:
        lo_lo = -0.5 * table_dim[..., 1]
        lo_hi = bump_pos - 0.5 * bump_width
        lo_lo, lo_hi = adjust_bound(lo_lo, lo_hi,
                                    offset=-foot_radius)
        y_lo = randu_like(lo_lo, lo_hi,
                          table_pos[..., 1])

        hi_lo = bump_pos + 0.5 * bump_width
        hi_hi = +0.5 * table_dim[..., 1]
        hi_lo, hi_hi = adjust_bound(hi_lo, hi_hi,
                                    offset=-foot_radius)
        y_hi = randu_like(hi_lo, hi_hi,
                          table_pos[..., 1])
        if side is not None:
            if side == 0:
                y = y_lo
            if side == 1:
                y = y_hi
        else:
            y = th.where(th.randn_like(y_lo) < 0, y_lo, y_hi)

    return th.stack([x, y], dim=-1)


def sample_wall_xy(table_pos: th.Tensor,
                   table_dim: th.Tensor,
                   wall_width: float,
                   foot_radius: th.Tensor
                   ):

    lo_lo = 0.
    lo_hi = (0.5 * table_dim[..., 0]
             - wall_width)
    lo_lo, lo_hi = adjust_bound(lo_lo, lo_hi,
                                offset=-foot_radius)
    x_lo = randu_like(lo_lo, lo_hi,
                      table_pos[..., 1])

    hi_lo = (0.5 * table_dim[..., 0] - wall_width)
    hi_hi = 0.5 * table_dim[..., 0]
    hi_lo, hi_hi = adjust_bound(hi_lo, hi_hi,
                                offset=-foot_radius)
    x_hi = randu_like(hi_lo, hi_hi,
                      table_pos[..., 1])

    x = th.where(th.randn_like(x_lo) < 0, x_lo, x_hi)

    x[th.randn_like(x) < 0] *= -1
    x += table_pos[..., 0]

    lo_lo = 0.
    lo_hi = (0.5 * table_dim[..., 1]
             - wall_width)
    lo_lo, lo_hi = adjust_bound(lo_lo, lo_hi,
                                offset=-foot_radius)
    y_lo = randu_like(lo_lo, lo_hi,
                      table_pos[..., 1])

    hi_lo = (0.5 * table_dim[..., 1]
             - wall_width)
    hi_hi = (0.5 * table_dim[..., 1])
    hi_lo, hi_hi = adjust_bound(hi_lo, hi_hi,
                                offset=-foot_radius)
    y_hi = randu_like(hi_lo, hi_hi,
                      table_pos[..., 1])

    y = th.where(th.randn_like(y_lo) < 0, y_lo, y_hi)

    y[th.randn_like(y) < 0] *= -1
    y += table_pos[..., 1]

    return th.stack([x, y], dim=-1)


def z_from_q_and_hull(q: th.Tensor, hull: th.Tensor,
                      table_height: th.Tensor):
    zvec = th.as_tensor([0, 0, 1], dtype=q.dtype, device=q.device)
    # zvec = einops.repeat(zvec, 'd -> n d', n=q.shape[0])
    zvec = zvec.broadcast_to(*q.shape[:-1], 3)
    zvec = quat_rotate(quat_inverse(q), zvec)
    zmin = th.einsum('...ni, ...i -> ...n', hull, zvec).amin(dim=-1)
    z = table_height - zmin
    return z


class SampleCuboidOrientation:
    def __init__(self, device: str = None):
        IRT2 = math.sqrt(1.0 / 2)
        canonicals = np.asarray([
            [0.000, 0.000, 0.000, 1.000],
            [1.000, 0.000, 0.000, 0.000],
            [-IRT2, 0.000, 0.000, +IRT2],
            [+IRT2, 0.000, 0.000, +IRT2],
            [0.000, -IRT2, 0.000, +IRT2],
            [0.000, +IRT2, 0.000, +IRT2],
        ], dtype=np.float32)
        self.canonicals = th.as_tensor(canonicals,
                                       dtype=th.float,
                                       device=device)

    def __call__(self, env_id: th.Tensor,
                 shape: Tuple[int, ...],
                 aux=None) -> np.ndarray:
        indices = th.randint(high=self.canonicals.shape[0],
                             size=(np.prod(shape),),
                             device=self.canonicals.device)
        qs = self.canonicals[indices]
        return qs.reshape(merge_shapes(shape, 4))


class SampleRandomOrientation:
    def __init__(self, device: str = None):
        self.device = device

    def __call__(self, env_id: th.Tensor,
                 shape: Tuple[int, ...],
                 aux=None) -> th.Tensor:
        return (random_quat(size=np.prod(shape, initial=1),
                            device=self.device)
                .reshape(merge_shapes(shape, 4)))


class SampleStableOrientation:
    def __init__(self, stable_poses_fn):
        self.stable_poses_fn = stable_poses_fn

    def __call__(self, env_id: th.Tensor,
                 shape: Tuple[int, ...] = (),
                 aux=None) -> np.ndarray:
        stable_poses = self.stable_poses_fn()
        n_env = len(env_id)
        count: int = int(np.prod(shape, initial=1)
                         / n_env)

        pose_index = th.randint(
            stable_poses.shape[1],
            size=(n_env, count),
            dtype=th.long,
            device=stable_poses.device,
        )
        if aux is not None:
            aux['pose_index'] = pose_index.reshape(
                merge_shapes(shape)
            )
        # return (stable_poses[env_id, pose_index, 3:7]
        #         .reshape(merge_shapes(shape, 4)))
        return th.take_along_dim(stable_poses[env_id, ..., 3:7],
                                 pose_index[..., None],
                                 -2).reshape(merge_shapes(shape, 4))


class SampleDefaultOrientation:
    def __init__(self, device: str = None):
        self.q_i = th.as_tensor([0, 0, 0, 1],
                                dtype=th.float,
                                device=device)
    def __call__(self, obj_id: th.Tensor,
                 shape: Tuple[int, ...] = ()) -> th.Tensor:
        return (einops.repeat(self.q_i, 'n d', n=np.prod(shape))
                .reshape(merge_shapes(shape, 4)))


class SampleWeightedStableOrientation:
    def __init__(self,
                 stable_poses_fn,
                 weights_fn):
        self.stable_poses_fn = stable_poses_fn
        self.weights_fn = weights_fn

    def __call__(self, env_id: th.Tensor,
                 shape: Tuple[int, ...] = (),
                 aux=None) -> np.ndarray:
        stable_poses = self.stable_poses_fn()
        probs = self.weights_fn()

        n_env = len(env_id)
        count: int = int(np.prod(shape, initial=1)
                         / n_env)
        replace = True if (count > probs.shape[-1]) else False
        pose_index = th.multinomial(
            probs, count, replacement=replace)[env_id]
        if aux is not None:
            aux['pose_index'] = pose_index
        # return (stable_poses[env_id, pose_index, 3:7]
        #         .reshape(merge_shapes(shape, 4)))
        return th.take_along_dim(stable_poses[env_id, ..., 3:7],
                                 pose_index[..., None],
                                 -2).reshape(merge_shapes(shape, 4))


class SampleInitialOrientation:
    def __init__(self, env):
        self.env = env

    def __call__(self, env_id: th.Tensor,
                 shape: Tuple[int, ...],
                 aux=None) -> np.ndarray:
        env = self.env
        n_env = len(env_id)
        count: int = int(np.prod(shape, initial=1)
                         / n_env)
        obj_ids = env.scene.cur_props.ids[env_id].long()
        if aux is not None:
            aux['pose_index'] = env.scene.cur_pose_index[env_id].expand(
                n_env, count)
        return (env.tensors['root'][obj_ids, 3:7][:, None]
                .expand(n_env, count, 4)
                .reshape(merge_shapes(shape, 4)))


class SampleMixtureOrientation:
    def __init__(self,
                 sample_fns,
                 probs: List[float]):
        assert (len(sample_fns) == len(probs))

    def __call__(self, obj_id: th.Tensor,
                 shape: Tuple[int, ...] = (),
                 aux=None):
        count: int = np.prod(shape, initial=1)
        qss = th.stack([f(obj_id, count, aux=aux)
                       for f in self.sample_fns], dim=0)  # F,S,4
        sel = th.randint(qss.shape[0],
                         size=qss.shape[1],
                         device=qss.device)  # S
        return (th.take_along_dim(qss, sel[None, :, None], dim=0)
                .reshape(merge_shapes(shape, 4)))


class RandomizeYaw:
    def __init__(self,
                 sample_fn,
                 device: str = None):
        self.sample_fn = sample_fn
        self.device = device

    def __call__(self, obj_id: th.Tensor,
                 shape: Tuple[int, ...] = (),
                 *args, **kwds):
        q = self.sample_fn(obj_id, shape, *args, **kwds)
        count: int = np.prod(shape, initial=1)
        qz = sample_yaw(count, device=self.device).reshape(
            q.shape)
        return (quat_multiply(qz, q)
                .reshape(merge_shapes(shape, 4)))


class CuboidRoll:
    def __init__(self,
                 sample_fn,
                 device: str = None):
        self.sample_fn = sample_fn
        self.device = device

    def __call__(self, obj_id: th.Tensor,
                 shape: Tuple[int, ...] = (),
                 *args, **kwds):
        q = self.sample_fn(obj_id, shape, *args, **kwds)
        count: int = np.prod(shape, initial=1)
        qz = sample_cuboid(count, device=self.device)
        return (quat_multiply(qz, q)
                .reshape(merge_shapes(shape, 4)))


class RotateOverPrimaryAxis:
    def __init__(self,
                 sample_fn,
                 device: str = None):
        self.sample_fn = sample_fn
        self.device = device

    def __call__(self, obj_id: th.Tensor,
                 shape: Tuple[int, ...] = (),
                 eps: float = 1e-9,
                 *args, **kwds):
        q = self.sample_fn(obj_id, shape, *args, **kwds)
        count: int = np.prod(shape, initial=1)
        axis = th.zeros_like(q[..., :3])
        axis[..., -2] = 1
        axis = quat_rotate(q, axis)
        primary_axis = th.argmax(th.abs(axis), -1, keepdim=True)
        axis = th.zeros_like(axis).scatter_(-1, primary_axis, 1.)
        # sin_half = th.linalg.norm(axis, dim=-1, keepdim=True)
        # axis = th.where(sin_half < eps,
        #             th.tensor([0,0,1], dtype=th.float, device=self.device),
        #             axis.div_(sin_half))
        qz = sample_random_along_axis(count, axis=axis,
                                      device=self.device)
        return (quat_multiply(qz, q)
                .reshape(merge_shapes(shape, 4)))


def test_sample_flat_xy():
    from matplotlib import pyplot as plt

    device = 'cpu'
    n: int = 512
    to_torch = partial(th.as_tensor,
                       dtype=th.float,
                       device=device)
    table_pos = einops.repeat(to_torch([0.3, 0.5, 0.25]),
                              'd -> n d', n=n)
    table_dim = einops.repeat(to_torch([0.4, 0.5, 0.50]),
                              'd -> n d', n=n)
    obj_radius = th.full((n,), 0.03)
    keepout_center = einops.repeat(to_torch([0.2, 0.4]),
                                   'd -> n d', n=n)
    keepout_radius = 0.15
    margin_scale = 1.0
    scene_type: str = 'flat'
    bump_width: float = 0.05

    if scene_type == 'flat':
        xy = sample_flat_xy(table_pos,
                            table_dim,
                            obj_radius,
                            n,
                            device,
                            th.float,
                            keepout_center,
                            keepout_radius,
                            prevent_fall=True,
                            avoid_overlap=True,
                            margin_scale=margin_scale,
                            pos_scale=1.0)
    elif scene_type == 'bump':
        def accept_keepout(xy):
            delta = xy - keepout_center[None]
            sqd = th.einsum('...i,...i->...', delta, delta)
            return (sqd > (keepout_radius * keepout_radius))
        xy = rejection_sample(
            partial(sample_bump_xy,
                    einops.repeat(table_pos, '... -> r ...', r=8),
                    table_dim,
                    bump_width,
                    obj_radius),
            accept_keepout)
    xy = dcn(xy)

    if True:
        for pos, r in zip(xy, dcn(obj_radius)):
            obj = plt.Circle(pos,
                             r,
                             fill=False,
                             color='b')
            plt.gca().add_patch(obj)

    if True:
        plt.scatter(xy[..., 0], xy[..., 1])

    keepout = plt.Circle(dcn(keepout_center[0]),
                         keepout_radius, fill=False, color='r')
    table = plt.Rectangle(table_pos[0, :2] - 0.5 * table_dim[0, :2],
                          table_dim[0, 0],
                          table_dim[0, 1],
                          fill=False,
                          color='k')
    margin = plt.Rectangle(
        table_pos[0, : 2] - 0.5 * margin_scale * table_dim[0, : 2],
        margin_scale * table_dim[0, 0],
        margin_scale * table_dim[0, 1],
        fill=False, color='k')

    plt.gca().add_patch(keepout)
    plt.gca().add_patch(table)
    plt.gca().add_patch(margin)
    if scene_type == 'bump':
        tp = dcn(table_pos[0, :2])
        bump = plt.Rectangle(
            tp - (0.5 * table_dim[0, 0], 0.5 * bump_width),
            table_dim[0, 0],
            bump_width,
            fill=True,
            color='g')
        plt.gca().add_patch(bump)
    plt.grid()
    plt.axis('equal')
    plt.show()


def test_sample_bump_xy():
    pass


def main():
    test_sample_flat_xy()


if __name__ == '__main__':
    main()
