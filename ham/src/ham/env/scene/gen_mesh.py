#!/usr/bin/env python3

from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from tempfile import TemporaryDirectory
import io

from yourdfpy import URDF
import torch as th
import numpy as np
import trimesh
import einops
import open3d as o3d

import matplotlib
# matplotlib.rcParams['backend'] = 'TkAgg'
from matplotlib import pyplot as plt

# TODO(ycho): move to math_util
from ham.models.cloud.point_mae import subsample
from ham.util.torch_util import dcn
from ham.util.math_util import quat_rotate
from ham.util.o3d_util import th2o3d, np2o3d, o3d2th
from PIL import Image
import open3d as o3d

URDF_TEMPLATE: str = '''<robot name="robot">
    <link name="base_link">
        <inertial>
            <mass value="{mass}"/>
            <origin xyz="0 0 0"/>
            <inertia ixx="{ixx}" ixy="{ixy}" ixz="{ixz}"
            iyy="{iyy}" iyz="{iyz}" izz="{izz}"/>
        </inertial>
        <visual>
            <origin xyz="{origin}" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{vis_mesh}" scale="1.0 1.0 1.0"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="{origin}" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{col_mesh}" scale="1.0 1.0 1.0"/>
            </geometry>
        </collision>
    </link>
</robot>
'''


@dataclass
class MeshGenConfig:
    vis_file: str
    col_file: str
    init_cloud: Optional[str] = None
    env_cloud: Optional[str] = None
    goal_cloud: Optional[str] = None
    raycast_height: float = 0.1
    raycast_resolution: int = 1024
    divide_goal: bool = True
    has_ceil: bool = False
    hack_workspace_reduction: float = 0.9
    origin_offset: Tuple[float, float, float] = (0., 0., 0.)

def get_place_margin(
        mesh: trimesh.Trimesh,
        place: np.ndarray,
        height: float,
        resolution: int = 16):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.t.geometry.TriangleMesh.from_legacy(
            mesh.as_open3d))

    ray_src = place.copy()
    ray_src[..., 2] += height
    ray_src = einops.repeat(ray_src, 'n d -> n r d',
                            r=resolution)

    a = np.linspace(-np.pi, +np.pi, num=resolution)
    c, s = np.cos(a), np.sin(a)
    o = np.zeros_like(c)
    ray_dir = np.stack([c, s, o], axis=-1)
    ray_dir = einops.repeat(ray_dir, 'r d -> n r d',
                            n=ray_src.shape[0])
    ray = np.concatenate([ray_src, ray_dir], axis=-1).astype(
        np.float32)
    ans = scene.cast_rays(np2o3d(ray.reshape(-1, 6)))
    rad = np.min(ans['t_hit'].numpy().reshape(-1, resolution), axis=-1)
    return rad

class MeshGen:
    Config = MeshGenConfig

    def __init__(self, cfg: Config, device=None):
        self.cfg = cfg

        # annotated mesh
        ann_mesh = trimesh.load(cfg.vis_file,
                                force='mesh')

        # ann_mesh.show() 
        if ((cfg.env_cloud is None) or
            (cfg.init_cloud is None) or
            (cfg.goal_cloud is None)):
            cloud, face_id, colors = trimesh.sample.sample_surface(ann_mesh, 4096*20,
                                                                   sample_color=True)
            # colors = color_visual_from_texture_visual(
            #     ann_mesh.visual).face_colors[face_id]

            colors = colors[..., :3]

            place_mask = (colors == (255, 0, 0)).all(axis=-1)
            goal_mask = (colors == (0, 255, 0)).all(axis=-1)
            if not cfg.divide_goal:
                place_mask |= goal_mask
            cloud_mask = place_mask | (colors == (0, 0, 255)).all(axis=-1)
            cloud_mask |= goal_mask
            place, cloud, goal = (cloud[place_mask], cloud[cloud_mask],
                                cloud[goal_mask])
            
        if cfg.env_cloud is None:
            self.__cloud = th.as_tensor(cloud,
                                dtype=th.float32)[None]
        else:
            cloud = o3d.t.io.read_point_cloud(cfg.env_cloud)
            self.__cloud = o3d2th(cloud.point.positions).to(th.float32)[None]

        if cfg.init_cloud is None:
            place_margin = get_place_margin(trimesh.load(cfg.col_file,
                                                        force='mesh'),
                                            place,
                                            height=cfg.raycast_height,
                                            resolution=cfg.raycast_resolution)
            self.__radius = th.as_tensor(place_margin,
                                    dtype=th.float32)[None]
            self.__place = th.as_tensor(place,
                                dtype=th.float32)[None]
        else:
            self.__radius = None
            place = o3d.t.io.read_point_cloud(cfg.init_cloud)
            self.__place = o3d2th(place.point.positions).to(th.float32)[None]

        self.__goal = None
        self.__goal_margin = None  
        if cfg.divide_goal:
            if cfg.goal_cloud is None:
                self.__goal = th.as_tensor(goal,
                                        dtype=th.float32)[None]
                goal_margin = get_place_margin(trimesh.load(cfg.col_file,
                                                        force='mesh'),
                                            goal,
                                            height=cfg.raycast_height,
                                            resolution=cfg.raycast_resolution)
                self.__goal_margin = th.as_tensor(goal_margin,
                                                dtype=th.float32)[None]
            else:
                goal = o3d.t.io.read_point_cloud(cfg.goal_cloud)
                self.__goal = o3d2th(goal.point.positions).to(th.float32)[None]

        if cfg.origin_offset != (0., 0., 0.):
            offset = th.as_tensor(cfg.origin_offset,
                                  dtype=th.float32)
            self.__place += offset
            self.__cloud += offset
            if self.__goal is not None:
                self.__goal += offset
        

    def fused_urdf(self, dim_table: th.Tensor,
                   tmpdir: Optional[str] = None):
        cfg = self.cfg
        N: int = dim_table.shape[0]
        origin = ' '.join([str(v) for v in cfg.origin_offset])

        urdf = URDF_TEMPLATE.format(mass=0.0,
                                    ixx=0.0,
                                    ixy=0.0,
                                    ixz=0.0,
                                    iyy=0.0,
                                    iyz=0.0,
                                    izz=0.0,
                                    vis_mesh=cfg.vis_file,
                                    col_mesh=cfg.col_file,
                                    origin=origin)
        urdf = einops.repeat(np.asarray(urdf, dtype=str),
                             '... -> n ...', n=N)
        # Some sort of indicator as to which id this is
        # TODO(ycho): is this necessary...?
        pose = th.as_tensor(np.zeros(N, dtype=np.float32),
                            device=dim_table.device)
        return dict(urdf=urdf, pose=pose, geom=None)

    def urdf(self,
             dim_table: th.Tensor,
             tmpdir: Optional[str] = None,
             fuse_pose: bool = True):
        assert (fuse_pose)
        return self.fused_urdf(dim_table, tmpdir)

    def pose(self,
             geom,
             dim_table: th.Tensor,
             z_table: float = 0.0):
        # NOTE(ycho): should _not_ be
        # dynamically reconfigurable
        # (except for maybe table_height)
        return {}

    def cloud(self, workspace, dim_table, meta, num_samples: int,
              eps: float = 1e-6):
        ws_lo, ws_hi = workspace.unbind(dim=-2)
        cloud = self.__cloud.to(device=workspace.device)
        mask = ((ws_lo[..., None, :2] - eps <= cloud[..., :2]).all(dim=-1)
                & (cloud[..., :2] <= ws_hi[..., None, :2] + eps).all(dim=-1))
        i = th.multinomial(mask.float() + eps,
                           num_samples,
                           replacement=True)[..., None]
        cloud = th.take_along_dim(cloud, i, dim=-2)
        if meta is not None:
            cloud[..., 2] += meta[..., None]
        return cloud
    
    def __sample(self,
                 ws_lo: th.Tensor,
                 ws_hi: th.Tensor,
                 points: th.Tensor,
                 place_rad: Optional[th.Tensor],
                 obj_radius: th.Tensor,
                 num_samples: int,
                 eps: float,
                 pose_meta=None):
        mask = ((ws_lo[..., None, :2] - eps <= points[..., :2]).all(dim=-1)
                & (points[..., :2] <= ws_hi[..., None, :2] + eps).all(dim=-1))
        # print(points.mean(-2))
        # print(mask.sum(-1))
        if place_rad is not None:
            place = (place_rad[..., None, :] >= obj_radius[..., None])
            mask = mask[..., None, :].broadcast_to(*place.shape).clone()
            mask &= place
        else:
            mask = einops.repeat(mask, '... p -> ... n p',
                                 n=obj_radius.shape[-1])
        ss = mask.shape
        mask = mask.reshape(-1, *ss[2:])
        # print((place_rad >= obj_radius[..., None]).sum(-1))
        i = th.multinomial(mask.float() + eps,
                           num_samples,
                           replacement=True)[..., None]
        sampled_points = th.take_along_dim(points, i, dim=-2)
        sampled_points = sampled_points.reshape(*ss[:2],
                                                *sampled_points.shape[1:])
        if pose_meta is not None:
            sampled_points[..., 2] += pose_meta[..., None, None]
        return sampled_points

    def place(self,
              workspace: th.Tensor,
              dim_table: th.Tensor,
              obj_radius: th.Tensor,
              meta: Dict[str, th.Tensor],
              num_samples: int = 1,
              eps: float = 1e-6,
              **kwds):
        workspace *=self.cfg.hack_workspace_reduction
        ws_lo, ws_hi = workspace.unbind(dim=-2)

        place = self.__place.to(device=workspace.device)
        if self.__radius is not None:
            place_rad = self.__radius.to(device=workspace.device)
        else:
            place_rad = None

        place = self.__sample(ws_lo,
                              ws_hi,
                              place,
                              place_rad,
                              obj_radius,
                              num_samples,
                              eps,
                              meta)
        return place
    
    def goal_sample(self,
              workspace: th.Tensor,
              dim_table: th.Tensor,
              obj_radius: th.Tensor,
              meta: Dict[str, th.Tensor],
              num_samples: int = 1,
              eps: float = 1e-6,
              **kwds):
        cfg = self.cfg
        if not cfg.divide_goal:
            return self.place(workspace=workspace,
                            dim_table=dim_table,
                            obj_radius=obj_radius,
                            meta=meta,
                            num_samples=num_samples,
                            eps=eps,
                            **kwds)
        else:
            # workspace *=0.8
            ws_lo, ws_hi = workspace.unbind(dim=-2)
            goal = self.__goal.to(device=workspace.device)
            if self.__goal_margin is not None:
                goal_margin = self.__goal_margin.to(device=workspace.device)
            else:
                goal_margin = None
            place = self.__sample(ws_lo,
                                ws_hi,
                                goal,
                                goal_margin,
                                obj_radius,
                                num_samples,
                                eps,
                                pose_meta=meta)
            return place


def main():
    from gen_both import BothGen
    from ham.util.torch_util import set_seed
    set_seed(1)

    gen_both = BothGen(BothGen.Config())

    B: int = 4
    device: str = 'cpu'

    dim_table = th.as_tensor([0.51, 0.65, 0.4],
                             dtype=th.float32,
                             device=device)
    dim_table = einops.repeat(dim_table, '... -> b ...', b=B)
    _z_table: float = 0.0
    workspace = th.as_tensor(
        [[-0.255, -0.325, _z_table],
         [+0.255, +0.325, _z_table + 0.9]],
        dtype=th.float32)
    workspace = einops.repeat(workspace,
                              '... -> B ...',
                              B=B).clone()

    obj_radius = th.as_tensor(0.2)
    obj_radius = einops.repeat(obj_radius,
                               '... -> B ... t',
                               B=B, t=1)
    #[..., None]
    with TemporaryDirectory() as tmpdir:
        gen = MeshGen(MeshGen.Config(
            # vis_file='/tmp/docker/one-stand-u-like.obj',
            # col_file='/tmp/docker/one-stand-u-like-coacd.obj',
            # vis_file='/tmp/docker/bin.obj',
            # col_file='/tmp/docker/bin-coacd.obj',
            # vis_file='/tmp/docker/realistic/stand.obj',
            # col_file='/tmp/docker/realistic/stand_coacd.obj',
            # vis_file='/tmp/docker/step/step.obj',
            # col_file='/tmp/docker/step/step_coacd.obj',
            # vis_file='/tmp/docker/step/step-reduced3.obj',
            # col_file='/tmp/docker/step/step-reduced3_coacd.obj',
            vis_file= '/tmp/docker/domain/duktig/simple2.obj',
            col_file= '/tmp/docker/domain/duktig/coacd3.obj',
            env_cloud= '/tmp/docker/domain/duktig/env_cloud.ply',
            init_cloud= '/tmp/docker/domain/duktig/init_cloud.ply',
            goal_cloud= '/tmp/docker/domain/duktig/goal_cloud.ply',
            divide_goal=True
        ))
        urdf = gen.urdf(dim_table, tmpdir=tmpdir, fuse_pose=True)

        # TODO(ycho): meta?
        cloud_pcd = gen.cloud(workspace, dim_table, None, 1024)
        place_pcd = gen.place(workspace, dim_table,
                              obj_radius, None, 4096).squeeze(dim=1)
        goal_pcd = gen.goal_sample(workspace, dim_table,
                                   obj_radius, None,
                                   4096).squeeze(dim=1)

        for i in range(B):
            u = urdf['urdf'][i]
            with io.StringIO(u) as fp:
                scene = URDF.load(fp)

            # 1,1024,3
            # print('pcd', pcd.shape)
            pcd = trimesh.PointCloud(dcn(place_pcd[i]))
            pcd.colors = (0, 255, 255)

            pcd2 = trimesh.PointCloud(dcn(cloud_pcd[i]))
            pcd2.colors = (0, 255, 0)

            pcd3 = trimesh.PointCloud(dcn(goal_pcd[i]))
            pcd3.colors = (255, 255, 0) 

            # x = trimesh.creation.axis().apply_translation([0, 0, 0.25])
            trimesh.Scene(
                [scene.scene.dump(concatenate=True),
                 pcd, pcd2, pcd3, 
                 trimesh.creation.axis()]).show()


if __name__ == '__main__':
    main()
