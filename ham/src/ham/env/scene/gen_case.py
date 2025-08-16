#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch as th
import einops
from icecream import ic

from ham.env.scene.gen_util import (
    sample_cloud_from_triangles
)


def gen_wall_mask(
        shape: Tuple[int, ...],
        wall_prob: float = None,
        ceil_prob: float = None,

    no_ceil: bool = False,
    no_fore: bool = False,
    no_back: bool = False,
    device: Optional[str] = None
):
    if wall_prob is not None:
        wall_prob = th.full((*shape, 5),
                            wall_prob,
                            dtype=th.float32,
                            device=device)
        wall_mask = th.bernoulli(wall_prob).to(bool)
    else:
        wall_mask = th.randint(2,
                               size=(*shape, 5),
                               dtype=th.bool,
                               device=device)
    shuffle = True
    if no_ceil:
        wall_mask[..., 4].fill_(0)
    elif ceil_prob is not None:
        p = th.full_like(wall_mask[..., 4],
                         ceil_prob,
                         dtype=th.float32)
        wall_mask[..., 4] = th.bernoulli(p)

    if no_fore:
        wall_mask[..., 0].fill_(0)
    # if ceiling is closed then front needs to be open
    wall_mask[..., 0] &= (~wall_mask[..., 4])

    # Shuffle_along_dim
    if shuffle:
        # if ceiling is closed then
        # one of (back/left/right) walls needs to be closed.
        # Since we're going to shuffle wall_mask afterwards,
        # for now we just set the first(=back) wall.
        wall_mask[..., 1] |= wall_mask[..., 4]
        if no_back:
            noise = th.rand_like(wall_mask[..., 2:4], dtype=th.float)
            indices = th.argsort(noise, dim=-1)
            wall_mask[..., 2:4] = th.take_along_dim(
                wall_mask[..., 2:4], indices, dim=-1)
        else:
            noise = th.rand_like(wall_mask[..., 1:4], dtype=th.float)
            indices = th.argsort(noise, dim=-1)
            wall_mask[..., 1:4] = th.take_along_dim(
                wall_mask[..., 1:4], indices, dim=-1)
    return wall_mask


@dataclass
class CaseGenConfig:
    min_height: float = 0.3
    max_height: float = 0.5
    min_wall_height: float = 0.05
    max_wall_height: float = 0.1
    thickness: float = 0.05
    disable: bool = False
    no_ceil: bool = False
    no_back: bool = False
    z_max: float = 1.05
    no_front_wall: bool = False
    use_tight_ceil: bool = False
    p_tight: float = 0.25
    min_tight_gap: float = 0.03
    max_tight_gap: float = 0.05

    ceil_prob: Optional[float] = None
    wall_prob: Optional[float] = None


class CaseGen:
    Config = CaseGenConfig

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def geom(self, dim_table: th.Tensor,
             height_margin: float = 0.0):
        """
        list of geoms:
            * front wall (multipart?)
            * back wall (multipart?)
            * left wall (multipart?)
            * right wall (multipart?)
            * top ceiling (multipart?)
        """
        cfg = self.cfg
        device = dim_table.device
        B = dim_table.shape[:-1]

        geoms = []

        # front wall: (thickness, dim_y, max_height)
        for axis in [0, 1]:
            for loc in [-1, +1]:
                dims = th.zeros((*B, 3),
                                dtype=th.float,
                                device=device)
                dims[..., axis] = cfg.thickness
                dims[..., 1 - axis] = dim_table[..., 1 - axis]
                dims[..., 2] = cfg.max_height + height_margin
                geoms.append(dims)

        # ceiling
        dims = th.zeros((*B, 3),
                        dtype=th.float,
                        device=device)
        dims[..., 0] = dim_table[..., 0]
        dims[..., 1] = dim_table[..., 1]
        dims[..., 2] = cfg.thickness
        geoms.append(dims)

        # B X G x 3
        # True layout = B x (PLATE_DIR) x (BASE_NUM+EDGE_NUM) x 3
        geoms = th.stack(geoms, dim=-2)
        return geoms

    def pose(self,
             geoms,
             dim_table: th.Tensor,
             z_table: float = 0.0,
             offset: Optional[float] = 0.0,
             wall_mask: Optional[th.Tensor] = None,
             tight_ceil: Optional[th.Tensor] = None,
             object_height: Optional[th.Tensor] = None,
             open_base: Optional[th.Tensor] = None,
             **kwds):
        cfg = self.cfg

        # Parse configuration.
        B = dim_table.shape[:-1]
        device = dim_table.device

        if wall_mask is None:
            wall_mask = gen_wall_mask(B,
                                      cfg.wall_prob,
                                      cfg.ceil_prob,
                                      cfg.no_ceil,
                                      cfg.no_front_wall,
                                      cfg.no_Back,
                                      device)

        noise = th.rand((*B), device=device)

        # if cabinet-type (access from the front),
        # then {min,max} height = {min,max}_height
        # otherwise (wall/bin-type(, then
        # {min,max} height = {min,max}_wall_height
        min_height = th.where(wall_mask[..., 4],
                              cfg.min_height,
                              cfg.min_wall_height)

        max_height = th.where(wall_mask[..., 4],
                              cfg.max_height,
                              cfg.max_wall_height)

        height = min_height + (noise * (max_height - min_height))
        if cfg.use_tight_ceil:
            tight_offset = (noise * (cfg.max_tight_gap
                                     - cfg.min_tight_gap) + cfg.min_tight_gap)
            height = th.where(tight_ceil,
                              object_height.squeeze(-1) + tight_offset, height)
        height = th.where(wall_mask[..., 4],
                          height + offset, height)

        wall_height = wall_mask[..., :4] * height[..., None]

        # Front wall clamping
        wall_height[..., 0].clamp_max_(
            cfg.z_max - z_table).clamp_min_(0)
        # Revise wall_mask[...,0],
        # based on whether wall height is nonzero.
        wall_mask[..., 0] = th.logical_and(wall_mask[..., 0],
                                           wall_height[..., 0] > 0)

        # Apply `open_base`
        if open_base is not None:
            wall_height[..., :4] = th.where(
                wall_mask[..., :4],
                wall_height[..., :4],
                (-dim_table[..., 2] * open_base)[..., None]
            )

        if cfg.disable:
            wall_mask.fill_(0)
            wall_height.fill_(0)

        xfm = th.as_tensor([0, 0, 0, 0, 0, 0, 1],
                           dtype=th.float,
                           device=device).expand(*B, 5, 7).clone()
        xfm = xfm.clone()

        xfm[..., 0, 0] = -0.5 * (dim_table[..., 0] - cfg.thickness)
        xfm[..., 1, 0] = +0.5 * (dim_table[..., 0] - cfg.thickness)
        xfm[..., 2, 1] = -0.5 * (dim_table[..., 1] - cfg.thickness)
        xfm[..., 3, 1] = +0.5 * (dim_table[..., 1] - cfg.thickness)

        # walls
        xfm[..., :4, 2] = (z_table[..., None]
                           + wall_height
                           - 0.5 * geoms[..., :4, 2]
                           )
        # ceiling
        xfm[..., 4, 2] = (z_table
                          + wall_mask[..., 4] * height
                          - 0.5 * cfg.thickness
                          )

        out = {
            'pose': xfm,
            'geom': geoms,
            'has_wall': wall_mask[..., :4].reshape(
                *wall_mask.shape[:-1], 2, 2),
            'has_ceil': wall_mask[..., 4],
            'height': height
        }
        return out

    def face(self,
             workspace: th.Tensor,
             dim_table: th.Tensor,
             meta_base: Dict[str, th.Tensor],
             meta_case: Dict[str, th.Tensor]):
        cfg = self.cfg

        xfm = meta_base['pose']
        xfm_part = xfm[..., 1:, :].reshape(
            *xfm.shape[:-2], 2, 5, xfm.shape[-1])
        part_dims = meta_base['part_dim']
        xy_view = th.einsum('...dpd->...pd',
                            xfm_part[..., :2, :3, :2])  # 8232
        dd_view = th.einsum('...dpd->...pd',
                            part_dims[..., :2, :, :2]).abs()  # 8232

        # xy: <..., PLATE_NUM=3, SIGN=2, XY=2>
        xy = th.stack([
            xy_view - 0.5 * dd_view,
            xy_view + 0.5 * dd_view],
            dim=-2)
        xy = xy.reshape(*xy.shape[:-3], 6, 2)

        z = xfm_part[..., :3, 2] + 0.5 * part_dims[..., :3, 2]
        z = th.maximum(
            z[..., 0, :, None],
            z[..., 1, None, :])

        # FIXME(ycho): is this still (left-right-front-back) order?
        # Or has this been fixed to (front-back-left-right) order?
        z = th.stack([
            z[..., 0, :],  # front
            z[..., 2, :],  # back
            z[..., :, 0],  # left
            z[..., :, 2],  # right
        ], dim=-2)
        z = einops.repeat(z,
                          '... a p -> ... (p two) a',
                          two=2)

        # expand `xy` in front-back-right-left order.
        # NOTE(ycho): this can be a bit confusing,
        # but the coordinates for the _front_ wall (-x plate)
        # varies in the y-direction, so xy has to be reversed!
        xy = th.stack([
            # front (y-variation)
            xy[..., 1],
            # back (y-variation)
            xy[..., 1],
            # right (x-variation)
            xy[..., 0],
            # left (x-variation)
            xy[..., 0],
        ], dim=-1)
        xy_z = th.stack([xy, z], dim=-1)
        # 7,6,4,2
        #       ^ x_or_y, z
        #     ^   num sides
        #   ^     num tilings (pivots)
        # ^       batch dimensions

        xy_z_0 = xy_z[..., :-1, :, :]
        xy_z_1 = xy_z[..., 1:, :, :]

        # top triangles
        if True:
            ttri = top_tri(cfg,
                           dim_table, workspace,
                           meta_case, meta_base, xy_z_0,
                           xy_z_1)

        # ceiling triangles
        if True:
            ctri = ceil_tri(cfg,
                            dim_table, workspace,
                            meta_case, meta_base)

        if True:
            btri = bot_tri(cfg,
                           dim_table, workspace,
                           meta_case, meta_base, xy_z_0,
                           xy_z_1)

        # bot triangles
        ouv = th.cat([btri, ttri, ctri], dim=-3)  # << default
        return ouv

    def cloud(self,
              workspace: th.Tensor,
              dim_table: th.Tensor,
              meta_base: Dict[str, th.Tensor],
              meta_case: Dict[str, th.Tensor],
              num_samples: int):
        ouv = self.face(workspace,
                        dim_table,
                        meta_base,
                        meta_case)
        return sample_cloud_from_triangles(ouv, num_samples)


def bot_tri(cfg,
            dim_table,
            workspace,
            meta_case,
            meta_base,
            xy_z_0,
            xy_z_1):
    """
    bottom triangles (between top_tri and the base)
    """
    m = xy_z_1[..., 1] < xy_z_0[..., 1]
    o = th.where(m[..., None], xy_z_1, xy_z_0)
    uv = th.stack([xy_z_0, xy_z_1], dim=-2) - o[..., None, :]
    H = (
        meta_case['pose'][..., :4, 2]
        + 0.5 * meta_case['geom'][..., :4, 2]
        - 0.5 * cfg.thickness * meta_case['has_ceil'][..., None]
    )
    uv[..., 1] = th.maximum(uv[..., :1, 1],
                            uv[..., 1:, 1]).clamp_max_(
        H[..., None, :, None])
    u, v = th.unbind(uv, dim=-2)

    # o/u/v: <..., NUM_RANGE=5, XY=2, (X_OR_Y+Z)=2>
    ouv = th.stack([o, u, v], dim=-2)

    # front-back triangles
    y = ouv[..., :2, :, 0]
    x = th.zeros_like(y)
    z = ouv[..., :2, :, 1]
    ouv0 = th.stack([x, y, z], dim=-1)
    ouv0[..., 0, 0, 0] = (
        -0.5 * dim_table[..., 0, None]
        + cfg.thickness * meta_case['has_wall'][..., 0, 0, None])
    ouv0[..., 1, 0, 0] = (
        +0.5 * dim_table[..., 0, None]
        - cfg.thickness * meta_case['has_wall'][..., 0, 1, None])

    # right-left triangles
    x = ouv[..., 2:, :, 0]
    y = th.zeros_like(x)
    z = ouv[..., 2:, :, 1]
    ouv1 = th.stack([x, y, z], dim=-1)
    ouv1[..., 0, 0, 1] = (
        -0.5 * dim_table[..., 1, None]
        + cfg.thickness * meta_case['has_wall'][..., 1, 0, None]
    )
    ouv1[..., 1, 0, 1] = (
        +0.5 * dim_table[..., 1, None]
        - cfg.thickness * meta_case['has_wall'][..., 1, 1, None])

    # Presence of these triangles depends
    # on the presence of the walls...!
    # xyz: <..., NUM_RANGES=5,XY=2,SIGN=2,TRI=3,DIM=3>
    ouv = th.stack([ouv0, ouv1], dim=-4)
    ouv.mul_(meta_case['has_wall'][..., None, :, :, None, None])
    ouv = einops.rearrange(ouv, '... a b c d e -> ... (a b c) d e')
    return ouv


def ceil_tri(cfg,
             dim_table,
             workspace,
             meta_case,
             meta_base):
    H = th.ones_like(dim_table[..., 2]) * (
        # meta_case['height']
        meta_case['pose'][..., 4, 2]
        - 0.5 * cfg.thickness
    )
    center = meta_case['pose'][..., 4, :3]
    radius = 0.5 * meta_case['geom'][..., 4, :3]
    # corner = einops.repeat(center, '... d -> ... two1 two2 d',
    #                        two1=2,
    #                        two2=2)
    # center[... 0, 0, 0
    x0, x1 = center[..., 0] - radius[..., 0], center[..., 0] + radius[..., 0]
    y0, y1 = center[..., 0] - radius[..., 1], center[..., 0] + radius[..., 1]
    # ic(x0.shape, H.shape)

    points = th.stack([
        x0, y0, H,
        x0, y1, H,
        x1, y0, H,
        x1, y1, H
    ], dim=-1)
    # ic(points.shape)
    points = points.reshape(*points.shape[:-1], 4, 3)
    tri = th.stack([points[..., 0:3, :],
                    points[..., 1:4, :]
                    ], dim=-3)
    # ic(tri.shape)  # 14,3,3
    tri = tri.clip_(workspace[..., 0, None, None, :],
                    workspace[..., 1, None, None, :])

    tri = th.cat([tri[..., :1, :],
                  tri[..., 1:, :] - tri[..., :1, :]],
                 dim=-2)
    tri[..., 1:, :].mul_(meta_case['has_ceil'][..., None, None, None])
    tri = tri.reshape(*tri.shape[:-3], -1, 3, 3)
    return tri


def top_tri(cfg,
            dim_table,
            workspace,
            meta_case,
            meta_base,
            xy_z_0,
            xy_z_1
            ):

    # Max height i.e. ceiling
    # ic(meta_case['pose'].shape)  # 8,5,7
    H = (
        meta_case['pose'][..., :4, 2]
        + 0.5 * meta_case['geom'][..., :4, 2]
        - 0.5 * cfg.thickness * meta_case['has_ceil'][..., None]
    )[..., None, :]
    # ic(meta_case['height'])
    # ic(H)

    # Figure out the heights for each
    # of the four sides
    xfm = meta_base['pose']
    xfm_part = meta_base['pose'][..., 1:, :].reshape(
        *xfm.shape[:-2], 2, 5, xfm.shape[-1])

    part_dims = meta_base['part_dim']
    z = xfm_part[..., :3, 2] + 0.5 * part_dims[..., :3, 2]
    z = th.maximum(
        z[..., 0, :, None],
        z[..., 1, None, :]
    )
    z = th.stack([
        z[..., 0, :],  # front
        z[..., 2, :],  # back
        z[..., :, 0],  # right
        z[..., :, 2],  # left
    ], dim=-2)

    # repeat 3->6 for the two endpoint vertices per plate
    z = einops.repeat(z,
                      '... a p -> ... (p two) a',
                      two=2).clone()
    zm = th.maximum(z[..., :-1, :],
                    z[..., 1:, :]).clamp_max_(H)

    o = [
        xy_z_0[..., 0], H.expand_as(zm),
        xy_z_1[..., 0], H.expand_as(zm),
        xy_z_0[..., 0], zm,
        xy_z_1[..., 0], zm
    ]
    o = th.stack(o, dim=-2)
    o = einops.rearrange(o, '... t (four xy_z) s -> ... t s four xy_z',
                            four=4)

    # xyz0
    y = o[..., :2, :, 0]
    x = th.zeros_like(y)
    z = o[..., :2, :, 1]
    xyz0 = th.stack([x, y, z], dim=-1)
    xyz0[..., 0, :, 0] = (
        -0.5 * dim_table[..., None, None, 0]
        + cfg.thickness * meta_case['has_wall'][..., 0, 0, None, None]
    )
    xyz0[..., 1, :, 0] = (
        +0.5 * dim_table[..., None, None, 0]
        # - cfg.thickness * meta_case['has_wall'][..., 1, 0, None, None]
        - cfg.thickness * meta_case['has_wall'][..., 0, 1, None, None]
        # - cfg.thickness * meta_case['has_wall'][..., 1, 0, None, None]
        # -0.1
        # - cfg.thickness * meta_case['has_wall'][..., 1, 0, None, None]
        # - cfg.thickness * meta_case['has_wall'][..., 1, 0, None, None]
        # - cfg.thickness * meta_case['has_wall'][..., 1, 0, None, None]
        # - cfg.thickness * meta_case['has_wall'][..., 1, 0, None, None]
    )

    # xyz1
    x = o[..., 2:, :, 0]
    y = th.zeros_like(x)
    z = o[..., 2:, :, 1]
    xyz1 = th.stack([x, y, z], dim=-1)  # 75243
    xyz1[..., 0, :, 1] = (
        -0.5 * dim_table[..., None, None, 1]
        # + cfg.thickness * meta_case['has_wall'][..., 0, 1, None, None]
        + cfg.thickness * meta_case['has_wall'][..., 1, 0, None, None]
    )
    xyz1[..., 1, :, 1] = (
        +0.5 * dim_table[..., None, None, 1]
        - cfg.thickness * meta_case['has_wall'][..., 1, 1, None, None]
    )

    # HACK - for visualization
    o = th.cat([xyz0, xyz1], dim=-3)
    # o = th.cat([xyz0], dim=-3)

    o.reshape(*o.shape[:-3],
              2, 2,
              *o.shape[-2:]).mul_(
        meta_case['has_wall'][..., None, :, :, None, None]
    )
    # ic('ooo', o.shape)  # 7-5-2-4-3

    # HACK, for visualization
    tri = th.cat([o[..., :, :, 0:3, :],
                  o[..., :, :, 1:4, :]
                  ], dim=-3)
    # ic('tri', tri.shape)  # 7,5,8,3,2

    tri = tri.clip_(workspace[..., 0, None, None, None, :],
                    workspace[..., 1, None, None, None, :])

    # xyz -> ouv
    tri = th.cat([tri[..., :1, :],
                  tri[..., 1:, :] - tri[..., :1, :]],
                 dim=-2)
    tri = tri.reshape(*tri.shape[:-4], -1, 3, 3)
    return tri


def main():
    from ham.util.torch_util import dcn, set_seed
    from ham.util.math_util import matrix_from_pose
    import trimesh

    device: str = 'cpu'
    B: int = 4
    seed: int = 1
    set_seed(seed)

    dim_table = [0.4, 0.5, 0.4]
    dim_table = th.as_tensor(dim_table, device=device)
    dim_table = einops.repeat(dim_table, '... -> B ...',
                              B=B)
    z_table = th.as_tensor([0.2] * B, device=device)

    workspace = th.as_tensor(
        [[-0.2, -0.25, 0.2],
         [+0.2, +0.25, 0.2 + 0.9]],
        dtype=th.float32)
    workspace = einops.repeat(workspace,
                              '... -> B ...',
                              B=B)
    obj_radius = th.as_tensor(0.03)
    obj_radius = einops.repeat(obj_radius,
                               '... -> B ...',
                               B=B)
    cfg = CaseGenConfig(max_height=1.0,
                        no_ceil=1)
    gen = CaseGen(cfg)
    geom = gen.geom(dim_table)
    meta = gen.pose(geom, dim_table, z_table)
    for B_i in range(B):
        draw = []
        for G_i in range(geom.shape[1]):
            box = trimesh.creation.box(
                dcn(geom[B_i, G_i]),
                transform=dcn(matrix_from_pose(
                    meta['pose'][B_i, G_i, :3],
                    meta['pose'][B_i, G_i, 3:])))
            draw.append(box)

        trimesh.Scene(draw).show()


if __name__ == '__main__':
    main()
