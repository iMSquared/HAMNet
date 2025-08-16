#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict, Optional
import torch as th
import numpy as np
import einops
import io

from ham.util.torch_util import dcn

from ham.env.scene.gen_base import BaseGen
from ham.env.scene.gen_case import CaseGen, gen_wall_mask
from ham.env.scene.gen_util import sample_cloud_from_triangles
from ham.env.scene.util import (
    box_link,
    _list2str,
    multi_box_link
)
# from ham.util.math_util import euler
from cho_util.math import transform as tx


@dataclass
class BothGenConfig:
    base: BaseGen.Config = BaseGen.Config()
    case: CaseGen.Config = CaseGen.Config()


class BothGen:
    Config = BothGenConfig

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.base = BaseGen(cfg.base)
        self.case = CaseGen(cfg.case)

    def fused_urdf(self, dim_table: th.Tensor,
                   tmpdir: Optional[str] = None):
        geom = self.geom(dim_table)
        bg = dcn(geom).reshape(-1, *geom.shape[-2:])
        urdf = []

        pose = None
        pose = self.pose(geom, dim_table)
        _pose = dcn(pose['pose'])
        for i_scene, scene in enumerate(bg):

            # Build part meta
            xyzs = []
            rpys = []
            dims = []
            for i_part, part in enumerate(scene):
                xyz = _list2str(_pose[i_scene, i_part, 0:3])
                rpy = _list2str(tx.rotation.euler.from_quaternion(
                    _pose[i_scene, i_part, 3:7]))
                xyzs.append(xyz)
                rpys.append(rpy)
                dims.append(list(part))

            urdf_text: str = '''
                <robot name="robot">
                {box}
                </robot>
                '''.format(
                box=multi_box_link(
                    F'base_link',
                    dims,
                    xyzs,
                    rpys,
                    density=0.0
                )
            )
            urdf.append(urdf_text)
        urdf = np.reshape(urdf, geom.shape[:-2])

        return dict(urdf=urdf,
                    geom=geom,
                    pose=pose)

    def urdf(self,
             dim_table: th.Tensor,
             tmpdir: Optional[str] = None,
             fuse_pose: bool = False,
             geom: Optional[th.Tensor] = None
             ):
        if fuse_pose:
            return self.fused_urdf(dim_table, tmpdir)

        if geom is None:
            # geom: <..., G=16, 3>
            geom = self.geom(dim_table)

        bg = dcn(geom).reshape(-1, *geom.shape[-2:])
        urdf = []

        pose = None
        for i_scene, scene in enumerate(bg):
            urdf_i = []
            for i_part, part in enumerate(scene):
                xyz = '0 0 0'
                rpy = '0 0 0'

                urdf_text: str = '''
                <robot name="robot">
                {box}
                </robot>
                '''.format(
                    box=box_link(F'part_{i_part:02d}',
                                 list(part),
                                 density=0.0,
                                 xyz=xyz,
                                 rpy=rpy)
                )
                urdf_i.append(urdf_text)
            urdf.append(urdf_i)
        urdf = np.reshape(urdf, geom.shape[:-1])
        return dict(urdf=urdf, geom=geom)

    def geom(self, dim_table: th.Tensor):
        base_geom = self.base.geom(dim_table)
        case_geom = self.case.geom(
            dim_table,
            height_margin=self.base.cfg.max_height)
        return th.cat([base_geom, case_geom], dim=-2)

    def pose(self, geom, dim_table: th.Tensor, z_table: float = 0.0,
             **kwds):
        cfg = self.cfg

        if not isinstance(z_table, th.Tensor):
            z_table = z_table + th.zeros_like(dim_table[..., 2])

        if not cfg.case.use_tight_ceil:
            base = self.base.pose(geom[..., :11, :],
                                  dim_table, z_table, **kwds)
            base_offset = base['height'].amax(dim=(-1, -2))
            # FIXME(ycho): `kwds` (ex: wall_mask)
            # not passed to self.case.pose() !!
            _case = self.case.pose(geom[..., 11:, :],
                                   dim_table, z_table,
                                   offset=base_offset)
        else:
            B = dim_table.shape[:-1]
            device = dim_table.device
            # FIXME(ycho): code duplication...
            wall_mask = gen_wall_mask(B,
                                      cfg.case.wall_prob,
                                      cfg.case.ceil_prob,
                                      cfg.case.no_ceil,
                                      cfg.case.no_front_wall,
                                      cfg.case.no_back,
                                      device)
            tight_ceil = th.rand(*B,
                                 dtype=th.float,
                                 device=device) < cfg.case.p_tight
            # tight_ceil[:] = 1
            tight_ceil &= wall_mask[..., 4]
            base = self.base.pose(geom[..., :11, :],
                                  dim_table, z_table,
                                  tight_ceil=tight_ceil,
                                  **kwds)
            base_offset = base['height'].amax(dim=(-1, -2))

            open_base = None
            if cfg.base.p_open_base > 0:
                open_base = base['open_base']
            _case = self.case.pose(geom[..., 11:, :],
                                   dim_table, z_table,
                                   offset=base_offset,
                                   wall_mask=wall_mask,
                                   tight_ceil=tight_ceil,
                                   open_base=open_base,
                                   **kwds)

        return dict(
            base=base,
            case=_case,
            pose=th.cat([base['pose'], _case['pose']], dim=-2)
        )

    def face(self,
             workspace: th.Tensor,
             dim_table: th.Tensor,
             meta: Dict[str, th.Tensor]):
        base_face = self.base.face(workspace, dim_table,
                                   meta['base'])
        case_face = self.case.face(workspace, dim_table,
                                   meta['base'],
                                   meta['case'])
        return th.cat([base_face, case_face], dim=-3)

    def cloud(self, workspace, dim_table, meta, num_samples: int):
        ouv = self.face(workspace, dim_table, meta)
        return sample_cloud_from_triangles(ouv, num_samples)

    def place(self,
              workspace: th.Tensor,
              dim_table: th.Tensor,
              obj_radius: th.Tensor,
              meta: Dict[str, th.Tensor],
              num_samples: int = 1,
              eps: float = 1e-3, **kwds):
        # TODO(ycho): The current 'has_wall' logic is conservative,
        # in the sense that tit doesn't account for the
        # additional height due to the contributions
        # from the other axis.
        # TODO(ycho): now that case height is configured at offsets
        # from the base top, is it still necessary to apply the
        # `high_wall` logic? It seems that `high_wall` will
        # always be true.
        # has_wall = meta['case']['has_wall']
        # high_wall = (meta['case']['height'][..., None, None]
        #              > meta['base']['height'][..., :, [0, 2]])
        # has_wall = th.logical_and(high_wall, meta['case']['has_wall'])
        has_wall = meta['case']['has_wall']
        return self.base.place(workspace,
                               dim_table,
                               obj_radius,
                               meta['base'],
                               has_wall,
                               num_samples,
                               eps, **kwds)


def main():
    from ham.util.torch_util import dcn, set_seed
    from ham.util.math_util import matrix_from_pose
    import trimesh
    from icecream import ic

    device: str = 'cpu'
    B: int = 64
    seed: int = 5
    set_seed(seed)

    dim_table = [0.4, 0.5, 0.4]
    dim_table = th.as_tensor(dim_table, device=device)
    dim_table = einops.repeat(dim_table, '... -> B ...',
                              B=B)

    _z_table: float = 0.6
    z_table = th.as_tensor([_z_table] * B, device=device)

    workspace = th.as_tensor(
        [[-0.2, -0.25, _z_table],
         [+0.2, +0.25, _z_table + 0.9]],
        dtype=th.float32)
    workspace = einops.repeat(workspace,
                              '... -> B ...',
                              B=B)
    workspace = workspace.clone()
    workspace[..., 0, :2] = -0.5 * dim_table[..., :2]  # - 0.01
    workspace[..., 1, :2] = +0.5 * dim_table[..., :2]  # + 0.01
    obj_radius = th.as_tensor(0.1)
    obj_radius = einops.repeat(obj_radius,
                               '... -> B ...',
                               B=B)

    cfg = BothGenConfig()
    cfg.base.base_thickness = 0.02
    cfg.base.wall_thickness = 0.02
    cfg.base.min_base_length = 0.05
    cfg.base.min_height = 0.15
    cfg.base.max_height = 0.15
    cfg.base.min_ramp = 0.0
    cfg.base.max_ramp = 0.0
    cfg.case.thickness = 0.02
    cfg.case.max_wall_height = 0.35
    cfg.case.min_wall_height = 0.35
    cfg.case.no_ceil = False
    cfg.case.disable = False
    cfg.case.use_tight_ceil = True
    cfg.case.p_tight=1
    cfg.case.ceil_prob=1
    gen = BothGen(cfg)
    object_height = th.rand(B, device=device) * (0.15 - 0.08) + 0.08
    geom = gen.geom(dim_table)
    meta = gen.pose(geom, dim_table, z_table,
                    object_height=object_height)
    cloud = gen.cloud(workspace, dim_table,
                      meta, 4096)
    place = gen.place(workspace,
                      dim_table,
                      obj_radius[...,None],
                      meta,
                      256,
                      high_scale=1000.0).squeeze(dim=-3)
    print('place',place.shape) # 64,64,256,3

    if False:
        dd = gen.urdf(dim_table, fuse_pose=True)
        from yourdfpy import URDF

        print(dd['urdf'].shape)  # 64x16?
        for u in dd['urdf']:
            print(u.shape)
            with io.StringIO(u) as fp:
                scene = URDF.load(fp)
                scene.show()

    for B_i in range(B):
        draw = []
        for G_i in range(geom.shape[1]):
            box = trimesh.creation.box(
                dcn(geom[B_i, G_i]),
                transform=dcn(matrix_from_pose(
                    meta['pose'][B_i, G_i, :3],
                    meta['pose'][B_i, G_i, 3:])))
            draw.append(box)

        # pcd1 (blue): from `cloud`
        pcd1 = trimesh.PointCloud(dcn(cloud[B_i]))
        pcd1.colors = [0, 0, 255]
        draw.append(pcd1)

        # pcd2 (red ): from `place`
        pcd2 = trimesh.PointCloud(dcn(place[B_i]))
        pcd2.colors = [255, 0, 0]
        draw.append(pcd2)
        trimesh.Scene(draw).show()


if __name__ == '__main__':
    main()
