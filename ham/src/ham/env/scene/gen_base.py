#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from functools import partial

import numpy as np
import torch as th
import einops

from ham.util.torch_util import randu
from ham.util.math_util import (
    quat_from_axa,
    rejection_sample
)
from ham.env.scene.gen_util import (
    rotation_matrix_2d,
    sample_cloud_from_triangles)

from icecream import ic


@dataclass
class BaseGenConfig:
    min_height: float = 0.07
    max_height: float = 0.15
    min_ramp: float = float(0.0)
    max_ramp: float = float(np.deg2rad(30.0))
    base_thickness: float = 0.05
    wall_thickness: float = 0.05
    min_base_length: float = 0.1
    place_by_area: bool = True
    z_max: float = 1.05
    p_high: float = 0.5
    recenter: bool = True
    noflat: bool = False
    p_open_base: float = 0.0

    @property
    def wall_length(self) -> float:
        return float(self.max_height / np.cos(self.max_ramp))


def wall_pos(ramp_angle: th.Tensor,
             h_prev: th.Tensor,
             h_next: th.Tensor,
             ramp_pos: th.Tensor,
             thickness: th.Tensor,
             wall_length: th.Tensor):
    # if h_prev=0 h_next=1, then ramp_angle is CW
    # otherwise CCW
    dh = h_next - h_prev
    sign = th.sign(dh)
    ramp_angle = -ramp_angle * sign
    qx = (ramp_pos - sign * 0.5 * dh * th.tan(ramp_angle))
    qy = th.maximum(h_prev, h_next)
    q = th.stack([qx, qy], dim=-1)
    R = rotation_matrix_2d(ramp_angle)

    # ++ if from 1->0 else -+
    local_corner = th.stack([-sign * (0.5 * thickness),
                             th.ones_like(sign) * 0.5 * wall_length],
                            dim=-1)
    return q - (R @ local_corner[..., None]).squeeze(dim=-1)


class BaseGen:
    Config = BaseGenConfig

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # NOTE(ycho):
        # we make `min_ramp` into a member variable
        # to allow for adaptation via future curricula.
        self._min_ramp = cfg.min_ramp

    def geom(self, dim_table: th.Tensor) -> th.Tensor:
        cfg = self.cfg

        device = dim_table.device
        B = dim_table.shape[:-1]
        wall_length = th.as_tensor(cfg.wall_length, device=device)

        # NOTE(ycho):
        # This version guarantees minimum length of 0.1
        base_length = th.rand((*B, 2, 3), dtype=th.float, device=device)
        numer = dim_table[..., :2] - cfg.min_base_length * 3
        denom = base_length.sum(dim=-1)
        base_length *= (numer / denom)[..., None]
        base_length += cfg.min_base_length

        # Output geoms buffer
        geoms = []

        # [0] add 1 main-body
        main_dims = th.zeros((*B, 3), dtype=th.float,
                             device=device)
        main_dims[..., 0] = dim_table[..., 0]
        main_dims[..., 1] = dim_table[..., 1]
        main_dims[..., 2] = dim_table[..., 2]
        geoms.append(main_dims)

        # [1:11] add base/wall plates
        for PLATE_DIR in [0, 1]:
            # add 3 base-plates
            for BASE_NUM in range(3):
                base_dims = th.zeros((*B, 3), dtype=th.float,
                                     device=device)
                base_dims[...,
                          PLATE_DIR] = base_length[...,
                                                   PLATE_DIR,
                                                   BASE_NUM]
                base_dims[..., 1 - PLATE_DIR] = dim_table[..., 1 - PLATE_DIR]
                base_dims[..., 2] = cfg.base_thickness
                geoms.append(base_dims)

            # add 2 wall-plates
            for EDGE_NUM in range(2):
                wall_dims = th.zeros((*B, 3), dtype=th.float, device=device)
                wall_dims[..., :2] = cfg.wall_thickness
                wall_dims[..., 1 - PLATE_DIR] = dim_table[..., 1 - PLATE_DIR]
                wall_dims[..., 2] = wall_length
                geoms.append(wall_dims)

        # B X G x 3
        # Format: B x (PLATE_DIR) x (BASE_NUM+EDGE_NUM) x (3=XYZ)
        geoms = th.stack(geoms, dim=-2)
        return geoms

    def pose(self,
             geom: th.Tensor,
             dim_table: th.Tensor,
             z_table: th.Tensor,
             level: Optional[int] = None,
             tight_ceil: Optional[th.Tensor] = None,
             object_height: Optional[th.Tensor] = None
             ):
        cfg = self.cfg

        # Parse configuration.
        B = dim_table.shape[:-1]
        device = dim_table.device

        # Configure layout, height, tilt angle.
        # to balance lower-upper surfaces,
        # increase the threshold at which `layout`
        # becomes positive. (1/sqrt(2))

        def sample_layout(*Q):
            # 000 // level 0
            # 001 // level 1 (1)
            # 011 // level 1 (3)
            # 100 // level 1 (4)
            # 110 // level 1 (6)
            # 010 // level 2
            # 101 // level 2
            # 111 // none, or level 0
            S = (*Q, *B, 2, 3)

            if level is None:
                l = th.empty(S, device=device).uniform_() > cfg.p_high
                if tight_ceil is None:
                    return l
                else:
                    num_t = tight_ceil.count_nonzero()
                    src0 = th.as_tensor([
                        [0, 0, 1],
                        [0, 1, 1]
                    ], dtype=bool, device=device)
                    idx0 = th.randint(len(src0),
                                      size=(*Q, num_t))
                    l[tight_ceil, 0, :] = src0[idx0]
                    src1 = th.as_tensor([
                        [0, 0, 0],
                        [0, 1, 1],
                        [1, 1, 0]
                    ], dtype=bool, device=device)
                    is_bump = ((~l[..., 1, 0]
                               & l[..., 1, 1])
                               & ~l[..., 1, 2])
                    is_bump &= tight_ceil
                    num_b = is_bump.count_nonzero()
                    idx1 = th.randint(len(src1),
                                      size=(*Q, num_b))
                    l[is_bump, 1] = src1[idx1]
                    return l
            else:
                if level == 0:
                    return th.zeros(S, device=device)
                elif level == 1:
                    src0 = th.as_tensor([
                        [0, 0, 1],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 1, 0]
                    ], dtype=bool, device=device)  # 4x3

                    if True:
                        # sample each part then combine
                        src1 = th.as_tensor([
                            [0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 1],
                            [1, 0, 0],
                            [1, 1, 0]
                        ], dtype=bool, device=device)
                        idx0 = th.randint(len(src0), size=S[:-2])
                        idx1 = th.randint(len(src1), size=S[:-2])

                        out = th.stack([src0[idx0], src1[idx1]], dim=-2)
                        noise = th.rand_like(out[..., :1], dtype=th.float)
                        indices = th.argsort(noise, dim=-2)
                        out = th.take_along_dim(out, indices, dim=-2)
                    else:
                        # sample pair simultaneously
                        idx = src0[th.randint(len(src0), size=S[:-1])]
                        out = src0[idx]
                    return out
                elif level == 2:
                    src0 = th.as_tensor([
                        [0, 1, 0],
                        [1, 0, 1],
                    ], dtype=bool, device=device)  # 4x3

                    if True:
                        # sample each part then combine
                        src1 = th.as_tensor([
                            [0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 1],
                            [1, 0, 0],
                            [1, 1, 0],
                            [0, 1, 0],
                            [1, 0, 1],
                        ], dtype=bool, device=device)
                        idx0 = th.randint(len(src0), size=S[:-2])
                        idx1 = th.randint(len(src1), size=S[:-2])

                        out = th.stack([src0[idx0], src1[idx1]], dim=-2)
                        noise = th.rand_like(out[..., :1], dtype=th.float)
                        indices = th.argsort(noise, dim=-2)
                        out = th.take_along_dim(out, indices, dim=-2)
                    else:
                        # sample pair simultaneously
                        idx = src0[th.randint(len(src0), size=S[:-1])]
                        out = src0[idx]
                    return out
                elif level == 999:
                    src0 = th.as_tensor([
                        [0, 0, 1],
                    ], dtype=bool, device=device)  # 4x3

                    # sample each part then combine
                    src1 = th.as_tensor([
                        [1, 0, 1],
                    ], dtype=bool, device=device)
                    idx0 = th.randint(len(src0), size=S[:-2])
                    idx1 = th.randint(len(src1), size=S[:-2])
                    out = th.stack([src0[idx0], src1[idx1]], dim=-2)
                    # noise = th.rand_like(out[..., :1], dtype=th.float)
                    # indices = th.argsort(noise, dim=-2)
                    # out = th.take_along_dim(out, indices, dim=-2)
                    return out
                else:
                    raise ValueError(F'invalid level={level}')

        if cfg.noflat:
            def accept_nonflat(layout):
                # [000], [000] -> not ok
                # [111], [Any] -> not ok
                # [Any], [111] -> not ok
                # [111], [111] -> not ok
                grid = th.logical_or(
                    layout[..., 0, :][..., :, None],
                    layout[..., 1, :][..., None, :]
                )
                return ((grid != grid[..., :1, :1])
                        .reshape(*grid.shape[:-2], -1)
                        .any(dim=-1))
            layout = rejection_sample(partial(sample_layout, 4),
                                      accept_nonflat,
                                      batched=True)
        else:
            layout = sample_layout()

        height = layout * (
            th.empty((*B, 1, 1), device=device).uniform_(
                cfg.min_height,
                cfg.max_height)
        )

        bound_hi = cfg.z_max - z_table[..., None, None]
        height = height.clamp_max_(bound_hi)
        ramp_angle = th.empty(
            (*B, 2, 2),
            device=device).uniform_(
            self._min_ramp,
            cfg.max_ramp)

        # Derived parameters
        dh = height[..., 1:] - height[..., :-1]
        abs_dh = dh.abs()
        # 'run' as in 'rise over run'
        ramp_run = abs_dh * th.tan(ramp_angle)

        geom_dims = geom.reshape(*geom.shape[:-2], 11, 3)
        part_dims = geom_dims[..., 1:, :].reshape(
            *geom_dims.shape[:-2], 2, 5, 3)
        part_dims = part_dims[..., :3, :]

        # Create output transform buffer.
        xfm = th.as_tensor([0, 0, 0, 0, 0, 0, 1],
                           dtype=th.float,
                           device=device).expand(*B, 11, 7).clone()

        # main slice (unused)
        xfm_main = xfm[..., 0, :]

        if cfg.p_open_base > 0:
            open_base = th.full((*B,), cfg.p_open_base,
                                device=device).bernoulli().bool()
            xfm_main[..., 2] = th.where(
                open_base,
                # lower the base body to open the base plates
                z_table - 1.5 * dim_table[..., 2],
                # legacy default: always close the base plates
                z_table - 0.5 * dim_table[..., 2]
            )
        else:
            xfm_main[..., 2] = z_table - 0.5 * dim_table[..., 2]

        # part slice, comprising base-plates and wall-plates.
        xfm_part = xfm[..., 1:, :].reshape(
            *xfm.shape[:-2], 2, 5, xfm.shape[-1])

        xfm_base = xfm_part[..., :, :3, :]  # BAPD

        # P = PLANE_DIR (axis dir)
        # middle are all centered at (0, 0)
        offsets = [None, None]
        for P in range(2):
            xfm_base[..., P, 0, P] = (
                - 0.5 * part_dims[..., P, 1, P]
                - 1.0 * ramp_run[..., P, 0]
                - 0.5 * part_dims[..., P, 0, P]
            )
            xfm_base[..., P, 2, P] = (
                + 0.5 * part_dims[..., P, 1, P]
                + 1.0 * ramp_run[..., P, 1]
                + 0.5 * part_dims[..., P, 2, P]
            )

            # Recenter plate bounds to fit within the table.
            if cfg.recenter:
                vmin = xfm_base[..., P, 0, P] - 0.5 * part_dims[..., P, 0, P]
                vmax = xfm_base[..., P, 2, P] + 0.5 * part_dims[..., P, 2, P]
                vavg = 0.5 * (vmin + vmax)
                xfm_base[..., P, :, P] -= vavg[..., None]
                offsets[P] = vavg

        xfm_base[..., :, :, 2] = (
            height - 0.5 * cfg.base_thickness + z_table[..., None, None]
        )

        xfm_wall = xfm_part[..., :, -2:, :]

        # part_view: (..., xy, G, xyz)
        part_view = th.einsum('...dpd->...dp',
                              part_dims[..., :2, :3, :2])

        ramp_min = (
            - 0.5 * part_view[..., 1]
            - 0.5 * ramp_run[..., 0]
        )
        ramp_max = (
            + 0.5 * part_view[..., 1]
            + 0.5 * ramp_run[..., 1]
        )

        # # <batch...> x <num_axes> x <num_ramp> x <dim>
        ramp_pos = th.stack([ramp_min, ramp_max], dim=-1)
        if cfg.recenter:
            for P in range(2):
                ramp_pos[..., P, :] -= offsets[P][..., None]

        wp = wall_pos(ramp_angle,
                      height[..., :-1],
                      height[..., 1:],
                      ramp_pos,
                      cfg.wall_thickness,
                      cfg.wall_length)
        # ==> (..., num_axes, num_wall, dim(x/y,z))

        wall_view = th.einsum('...dpd->...dp', xfm_wall[..., :2])
        # Set displacement along main(x or y) axis
        wall_view[...] = wp[..., 0]
        # Set wall height
        xfm_wall[..., :2, :, 2] = wp[..., 1] + z_table[..., None, None]

        sign = th.sign(dh)
        axis = th.as_tensor([
            [0, 1, 0],
            [-1, 0, 0]
        ], dtype=th.float,
            device=device)
        xfm_wall[..., 3:] = quat_from_axa(
            sign[..., None] * ramp_angle[..., None] * axis[None, :, None])

        out = {
            'pose': xfm.reshape(xfm.shape[0], -1, 7),
            'layout': layout,
            'height': height,
            'ramp_pos': ramp_pos,
            'ramp_ang': ramp_angle,
            'ramp_run': ramp_run,
            'part_dim': part_dims,
        }
        # print('--> [gen_base] out', out['part_dim'].shape)
        if tight_ceil is not None:
            out['tight_ceil'] = tight_ceil
        if cfg.p_open_base > 0:
            out['open_base'] = open_base
        return out

    def face(self,
             workspace: th.Tensor,
             dim_table: th.Tensor,
             meta: Dict[str, th.Tensor]) -> th.Tensor:
        cfg = self.cfg
        # FIXME(ycho):
        # Refactor this part basically,
        # Which is exceedingly ugly.
        xfm = None
        if 'pose' in meta:
            xfm = meta['pose']
        elif 'base_pose' in meta:
            xfm = meta['base_pose']
        xfm_part = xfm[..., 1:, :].reshape(
            *xfm.shape[:-2], 2, 5, xfm.shape[-1])
        part_dims = meta['part_dim']

        # Extract diagonal views
        xy_view = th.einsum('...dpd->...pd',
                            xfm_part[..., :2, :3, :2])[..., None, :]
        dd_view = th.einsum('...dpd->...pd',
                            part_dims[..., :2, :, :2])[..., None, :]

        # xy: ..., (PLATE_NUM=3, SIGN=2, AXES=2)
        xy = th.cat([xy_view - 0.5 * dd_view,
                    xy_view + 0.5 * dd_view], dim=-2)
        xy = xy.reshape(*xy.shape[:-3], 6, 2)
        x, y = th.broadcast_tensors(xy[..., :, None, 0],
                                    xy[..., None, :, 1])

        z = th.maximum(
            xfm_part[..., 0, :3, None, 2],
            xfm_part[..., 1, None, :3, 2]
        ) + 0.5 * cfg.base_thickness
        z = einops.repeat(z,
                          '... i0 i1 -> ... (i0 r0) (i1 r1)',
                          r0=2, r1=2)

        xyz = th.stack([x, y, z], dim=-1)
        xyz = xyz.clip_(workspace[..., 0, None, None, :],
                        workspace[..., 1, None, None, :])

        p00 = xyz[..., :-1, :-1, :]
        p01 = xyz[..., :-1, 1:, :]
        p10 = xyz[..., 1:, :-1, :]
        p11 = xyz[..., 1:, 1:, :]

        # NOTE(ycho): here we're just trying to determine
        # the direction of the diagonal of the triangulation.
        # Maybe there's a smarter way to do this...
        m = (
            (p00[..., 2] - p11[..., 2]).abs()
            >
            (p01[..., 2] - p10[..., 2]).abs()
        )[..., None]

        # Triangle origin
        o = th.where(m, p00, p01)

        # NOTE(ycho): `uvs` here does _not_ particularly
        # build 'oriented' triangles, in the sense that
        # their cross product would produce consistent(outward) normals.
        uvs = th.where(m[..., None, :],
                       th.stack([p11, p10, p11, p01], dim=-2),
                       th.stack([p10, p00, p10, p11], dim=-2)).reshape(
            *m.shape[:-1], 2, 2, -1)

        u = uvs[..., 0, :] - o[..., None, :]
        v = uvs[..., 1, :] - o[..., None, :]

        o = einops.repeat(o, '... t d -> ... t two d', two=2)
        ouv = th.stack([o, u, v], dim=-2).reshape(*o.shape[:-4],
                                                  -1, 3, 3)
        return ouv

    def cloud(self,
              workspace: th.Tensor,
              dim_table: th.Tensor,
              meta: Dict[str, th.Tensor],
              num_samples: int = 1,
              eps: float = 1e-6):
        ouv = self.face(workspace,
                        dim_table,
                        meta)
        return sample_cloud_from_triangles(
            ouv, num_samples, eps)

    def place(self,
              workspace: th.Tensor,
              dim_table: th.Tensor,
              obj_radius: th.Tensor,
              meta: Dict[str, th.Tensor],
              has_wall: Optional[th.Tensor] = None,
              num_samples: int = 1,
              eps: float = 1e-3,
              high_scale: float = 1.0,
              init: bool = False):
        cfg = self.cfg

        # dim_table: Bx2
        # workspace: Bx2x3
        # n pose per env
        n_poses_per_env = obj_radius.shape[-1]
        layout = meta['layout']
        layout = einops.repeat(layout,
                               'B ...-> B n ...',
                               n=n_poses_per_env)
        ramp_run = meta['ramp_run']
        ramp_run = einops.repeat(ramp_run,
                                 'B ...-> B n ...',
                                 n=n_poses_per_env)
        workspace = einops.repeat(workspace,
                                  'B ...-> B n ...',
                                  n=n_poses_per_env).clone()
        dim_table = einops.repeat(dim_table,
                                  'B ...-> B n ...',
                                  n=n_poses_per_env)
        ramp_pos = einops.repeat(meta['ramp_pos'],
                                 'B ...-> B n ...',
                                 n=n_poses_per_env)

        grid_layout = th.logical_or(
            layout[..., 0, :][..., :, None],
            layout[..., 1, :][..., None, :]
        )

        lo_0 = workspace[..., 0, :2]
        if has_wall is not None:
            has_wall = einops.repeat(has_wall,
                                     'B ...-> B n ...',
                                     n=n_poses_per_env)
            if init:
                # While sampling initial poses,
                # Apply margin near the `dim_table` border,
                # which generally ensures that the
                # full body of the object is within the tabletop.
                margin = obj_radius[..., None]
                lo_0.clamp_min_(-0.5 * dim_table[..., :2] + margin)
            else:
                # While sampling goal poses,
                # Only apply margin if `has_wall` is True.
                margin = (obj_radius + cfg.wall_thickness)[..., None]
                lo_0.clamp_min_(-0.5 * dim_table[..., :2]
                                + has_wall[..., 0] * margin)
        hi_0 = (
            ramp_pos[..., 0]
        )
        lo_1 = (
            ramp_pos[..., 0]
        )
        hi_1 = (
            ramp_pos[..., 1]
        )
        lo_2 = (
            ramp_pos[..., 1]
        )
        hi_2 = workspace[..., 1, :2]
        if has_wall is not None:
            if init:
                margin = obj_radius[..., None]
                hi_2.clamp_max_(0.5 * dim_table[..., :2] - margin)
            else:
                margin = (obj_radius + cfg.wall_thickness)[..., None]
                hi_2.clamp_max_(0.5 * dim_table[..., :2]
                                - has_wall[..., 1] * margin)
        lo = th.stack([lo_0, lo_1, lo_2], dim=-2)
        hi = th.stack([hi_0, hi_1, hi_2], dim=-2)

        lo_ax0 = einops.repeat(lo[..., 0], '... d -> ... d x', x=3)
        lo_ax1 = einops.repeat(lo[..., 1], '... d -> ... x d', x=3)
        lo = th.stack([lo_ax0, lo_ax1], dim=-1)

        hi_ax0 = einops.repeat(hi[..., 0], '... d -> ... d x', x=3)
        hi_ax1 = einops.repeat(hi[..., 1], '... d -> ... x d', x=3)
        hi = th.stack([hi_ax0, hi_ax1], dim=-1)

        # major axis / lo
        lo[..., 1:, :, 0] += ((grid_layout[..., 1:, :] <
                               grid_layout[..., : -1, :]) * obj_radius[..., None, None] +
                              (grid_layout[..., 1:, :] > grid_layout[..., : -1, :]) * eps +
                              (grid_layout[..., 1:, :] != grid_layout[..., : -1, :]) * 0.5 *
                              ramp_run[..., 0, :, None])
        # minor axis / lo
        lo[..., :, 1:, 1] += ((grid_layout[..., :, 1:] <
                               grid_layout[..., :, : -1]) * obj_radius[..., None, None] +
                              (grid_layout[..., :, 1:] > grid_layout[..., :, : -1]) * eps +
                              (grid_layout[..., :, 1:] != grid_layout[..., :, : -1]) * 0.5 *
                              ramp_run[..., 1, None, :])
        # major axis / hi
        hi[..., : -1, :, 0] -= ((grid_layout[..., 1:, :] >
                                grid_layout[..., : -1, :]) * obj_radius[..., None, None] +
                                (grid_layout[..., 1:, :] < grid_layout[..., : -1, :]) * eps +
                                (grid_layout[..., 1:, :] != grid_layout[..., : -1, :]) * 0.5 *
                                ramp_run[..., 0, :, None])
        # minor axis / hi
        hi[..., :, : -1, 1] -= ((grid_layout[..., :, 1:] >
                                grid_layout[..., :, : -1]) * obj_radius[..., None, None] +
                                (grid_layout[..., :, 1:] < grid_layout[..., :, : -1]) * eps +
                                (grid_layout[..., :, 1:] != grid_layout[..., :, : -1]) * 0.5 *
                                ramp_run[..., 1, None, :])
        lo = lo.reshape(*lo.shape[:-3], -1, lo.shape[-1])
        hi = hi.reshape(*hi.shape[:-3], -1, hi.shape[-1])

        msk = (lo > hi)
        center = 0.5 * (lo + hi)
        lo[msk] = center[msk]
        hi[msk] = center[msk]

        if cfg.place_by_area:
            # area-weighted sampling
            area = (hi - lo)
            area = area[..., 0] * area[..., 1]

            # NOTE(ycho): Optionally apply height bias
            ss = area.shape
            if 'tight_ceil' not in meta:
                area[grid_layout.reshape(area.shape) >= 1] *= high_scale
            else:
                # high bias
                is_tight_ceil = einops.repeat(
                    meta['tight_ceil'],
                    'b ... -> b n ...',
                    n=n_poses_per_env
                )
                hb = ((grid_layout >= 1) &
                      ~is_tight_ceil[..., None, None]).reshape(area.shape)
                area[hb] *= high_scale
                # low bias
                lb = ((grid_layout < 1) &
                      is_tight_ceil[..., None, None]).reshape(area.shape)
                # HACK(JH) Arbitrary high value
                area[lb] *= 1e6
                # area[lb] *= 1.0 / high_scale

            area = area.reshape(-1, *ss[2:])
            i = th.multinomial(area + 1e-6, num_samples,
                               replacement=True)[..., None]
            i = i.reshape(*ss[:2], *i.shape[1:])
        else:
            # index-based sampling
            i = th.randint(9,
                           size=(*dim_table.shape[:-1], num_samples, 1),
                           device=dim_table.device)

        lo, hi = th.take_along_dim(
            lo, i, dim=-2), th.take_along_dim(hi, i, dim=-2)
        xy = lo + th.rand_like(lo) * (hi - lo)

        i0 = th.div(i.squeeze(dim=-1), 3, rounding_mode='trunc')
        i1 = i.squeeze(dim=-1) % 3
        i01 = th.stack([i0, i1], dim=-2)

        xfm = meta['pose']
        xfm_part = xfm[..., 1:, :].reshape(
            *xfm.shape[:-2], 2, 5, xfm.shape[-1])
        z_base = xfm_part[..., :, :3, 2]
        z_base = einops.repeat(z_base,
                               'B ...-> B n ...',
                               n=n_poses_per_env)
        height = th.take_along_dim(z_base + 0.5 * cfg.base_thickness,
                                   i01, dim=-1).amax(dim=-2)
        points = th.cat([xy, height[..., None]], dim=-1)
        return points


def balance_layout():
    from matplotlib import pyplot as plt
    B = (16384,)
    vs = np.linspace(0.0, 1.0, num=16)
    ps = []
    for v in np.linspace(0.0, 1.0, num=16):
        layout = th.empty((*B, 2, 3)).uniform_() > v
        grid_layout = th.logical_or(
            layout[..., 0, :][..., :, None],
            layout[..., 1, :][..., None, :]
        )
        ps.append(grid_layout.float().mean())
    plt.plot(vs, ps)
    plt.grid()
    # plt.plot(vs, 1.414 * np.log1p(ps))  # np.sqrt(1-vs))
    plt.plot(vs, 1 / vs)  # np.sqrt(1-vs))
    plt.xlabel('thresh')
    plt.ylabel('proportion of high plates')
    plt.show()


def main():
    from ham.util.torch_util import dcn, set_seed
    import trimesh
    from ham.util.math_util import matrix_from_pose

    device: str = 'cpu'
    B: int = 16
    seed: int = 1
    set_seed(seed)

    dim_table = [0.4, 0.5, 0.4]
    dim_table = th.as_tensor(dim_table, device=device)
    dim_table = einops.repeat(dim_table, '... -> B ...',
                              B=B)
    z_table = th.as_tensor([0.2] * B, device=device)

    workspace = th.as_tensor(
        [[-0.2, -0.25, 0.15],
         [+0.2, +0.25, 0.15 + 0.9]],
        dtype=th.float32)
    workspace = einops.repeat(workspace,
                              '... -> B ...',
                              B=B)
    obj_radius = th.as_tensor(0.03)
    obj_radius = einops.repeat(obj_radius,
                               '... -> B ...',
                               B=B)[..., None]

    cfg = BaseGenConfig(noflat=True)
    gen = BaseGen(cfg)

    geom = gen.geom(dim_table)
    meta = gen.pose(geom, dim_table, z_table,
                    level=2)
    cloud = gen.cloud(workspace, dim_table,
                      meta, 256)
    place = gen.place(workspace, dim_table,
                      obj_radius, meta, num_samples=256).squeeze(dim=-3)

    for B_i in range(B):
        draw = []
        for G_i in range(geom.shape[1]):
            box = trimesh.creation.box(
                dcn(geom[B_i, G_i]),
                transform=dcn(matrix_from_pose(
                    meta['pose'][B_i, G_i, :3],
                    meta['pose'][B_i, G_i, 3:])))
            draw.append(box)

        pcd1 = trimesh.PointCloud(dcn(cloud[B_i]))
        pcd1.colors = [0, 0, 255]
        draw.append(pcd1)

        pcd2 = trimesh.PointCloud(dcn(place[B_i]))
        pcd2.colors = [255, 0, 0]
        draw.append(pcd2)

        trimesh.Scene(draw).show()


if __name__ == '__main__':
    main()
