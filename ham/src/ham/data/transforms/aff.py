#!/usr/bin/env/python3

import os
from typing import Optional, Tuple, Union, List
from pathlib import Path
from yourdfpy import URDF
import trimesh
import numpy as np

from yourdfpy.urdf import apply_visual_color
from ham.util.path import get_path
from ham.data.transforms.io_xfm import scene_to_mesh
from ham.util.mesh_util import scene_to_mesh_with_texture


def get_link_mesh(urdf, link, use_col: bool = True,
                  alt_path: Optional[Union[List[str], str]] = None,
                  cat: bool = True, scale: Optional[Tuple[float, ...]] = None):
    if isinstance(alt_path, str) or isinstance(alt_path, os.PathLike):
        alt_path = [alt_path]
    if alt_path is None:
        alt_path = []

    geometries = link.collisions if use_col else link.visuals
    visuals = link.visuals

    if len(geometries) == 0:
        return None
    meshes = []
    for g in geometries:
        if g.geometry.mesh is None:
            continue

        f = g.geometry.mesh.filename
        if 'package://' in f:
            f = f.replace('package://', '')

        for root in [''] + alt_path:
            f2 = F'{root}/{f}'
            # print(F'try =  {f2}')
            if Path(f2).is_file():
                f = f2
                break
        else:
            raise FileNotFoundError(
                F'{g.geometry.mesh.filename} not found from paths = {alt_path}')

        m = trimesh.load(f, skip_materials=False,
                         force="scene")

        # >> TRY TO LOAD A "DECENT" visual mesh.
        if not use_col:
            m = scene_to_mesh_with_texture(m)
        else:
            m = scene_to_mesh(m)

        # NOTE(ycho): also fill in potentially missing colors
        # from URDF material specifications, if available.
        if not use_col:
            apply_visual_color(m, g, urdf._material_map)
        else:
            if len(geometries) == 1 and len(visuals) == 1:
                apply_visual_color(m, visuals[0], urdf._material_map)

        pose = g.origin

        if pose is None:
            pose = np.eye(4)

        if scale is None:
            scale = g.geometry.mesh.scale
        if scale is not None:
            S = np.eye(4)
            S[:3, :3] = np.diag(scale)
            pose = pose @ S
        m.apply_transform(pose)
        meshes.append(m)

    if len(meshes) == 0:
        return None

    if cat:
        return trimesh.util.concatenate(meshes)
    return meshes


def get_gripper_mesh(cat: bool = False,
                     frame: str = 'panda_tool',
                     urdf_path: Optional[str] = None,
                     links: Tuple[str, ...] = ('panda_hand', 'panda_leftfinger', 'panda_rightfinger')
                     ):
    """ as chulls though """
    if urdf_path is None:
        urdf_path = get_path(
            'assets/franka_description/robots/franka_panda.urdf')
    urdf = URDF.load(urdf_path,
                     build_collision_scene_graph=True,
                     load_collision_meshes=True,
                     force_collision_mesh=False)

    hulls = []
    for link in links:
        xfm = urdf.get_transform(link, frame,
                                 collision_geometry=True)
        loc = urdf.link_map[link].collisions[0].origin
        if loc is not None:
            xfm = xfm @ loc

        hull_file = urdf.link_map[link].collisions[0].geometry.mesh.filename
        scene: trimesh.Scene = trimesh.load(
            Path(urdf_path).parent / hull_file,
            split_object=True,
            group_material=False,
            skip_texture=True,
            skip_materials=True,
            force='scene')
        scene.apply_transform(xfm)

        for node_name in scene.graph.nodes_geometry:
            (transform, geometry_name) = scene.graph[node_name]
            mesh = scene.geometry[geometry_name]
            mesh.apply_transform(transform)
            hulls.append(mesh)
    out = hulls

    if cat:
        out = trimesh.util.concatenate(hulls)
    return out
