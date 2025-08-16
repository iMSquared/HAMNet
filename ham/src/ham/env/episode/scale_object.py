#!/usr/bin/env python3

from typing import Tuple, Dict
import torch as th
import numpy as np
from xml.dom import minidom

from cho_util.math import transform as tx
from ham.env.episode.spec import DefaultSpec
from ham.models.common import map_tensor
from ham.util.torch_util import randu, dcn
from ham.util.math_util import quat_rotate
from ham.env.scene.util import (_is_stable, _foot_radius,
                                get_faces_from_box)


class ScaleObject(DefaultSpec):
    """
    Physics properties
    """
    @property
    def setup_keys(self) -> Tuple[str, ...]: return ('object_scale',
                                                     'rel_scale',
                                                     'obj_ctx',
                                                     )

    @property
    def setup_deps(self) -> Tuple[str, ...]: return ('table_dim',
                                                     'obj_ctx',
                                                     )

    def sample_setup(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        num_env = ctx['num_env']
        scale_lo, scale_hi = ctx['scale_bound']
        obj_ctx = data['obj_ctx']

        data['object_scale'] = randu(scale_lo,
                                     scale_hi,
                                     (num_env,),
                                     device=ctx['device'])
        obj_scale = data['object_scale']
        rel_scale = obj_scale / obj_ctx['obj_radius']
        data['rel_scale'] = rel_scale

        # == update all relevant data ==
        obj_ctx['obj_radius'] *= rel_scale
        obj_ctx['obj_hull'] *= rel_scale[..., None, None]
        obj_ctx['obj_bbox'] *= rel_scale[..., None, None]
        obj_ctx['obj_cloud'][..., :3] *= rel_scale[..., None, None]

        if 'obj_foot_radius' in data:
            obj_ctx['obj_foot_radius'] *= rel_scale[..., None]

        if 'obj_height' in data:
            obj_ctx['object_height'] = rel_scale[..., None]

        # NOTE(ycho): `stable_pose` is defined
        # w.r.t the tabletop, and the height
        # needs to be scaled with z_ref=tabletop.z
        # instead of z_ref=0.0.
        if 'obj_stable_pose' in data:
            poses = obj_ctx['obj_stable_pose']
            table_h = data['table_dim'][..., -1]
            poses[..., 2] -= table_h
            poses[..., 2] = poses[..., 2] * rel_scale[..., None]
            poses[..., 2] = poses[..., 2] + table_h
            # NOTE(ycho): this part is _not_ necessary
            # since we updated `poses` in-place
            # without clone()...!
            obj_ctx['obj_stable_pose'] = poses
        return data

    def apply_setup(self,
                    ctx,
                    data: Dict[str, th.Tensor]):
        # FIXME(ycho): object_scale
        # is __not__ applied here, but in
        # PhysProp...
        tmpdir = ctx['tmpdir']
        obj_set = ctx['obj_set']
        obj_ctx = data['obj_ctx']
        object_keys = obj_ctx['obj_name']

        # NOTE(ycho): overwrite urdf with
        # updated scale values
        obj_scale = data['object_scale']
        rel_scale = obj_scale / obj_ctx['obj_radius']
        rel_scale_np = dcn(obj_scale)
        for idx, scale in enumerate(rel_scale_np):
            key = object_keys[idx]
            urdf = obj_set.urdf(key)

            # <read and parse URDF>
            with open(urdf, 'r', encoding='utf-8') as f:
                str_urdf = f.read()
            dom = minidom.parseString(str_urdf)

            # <update URDF with new scale>
            meshes = dom.getElementsByTagName("mesh")
            for mesh in meshes:
                mesh_scales = mesh.attributes['scale'].value.split(' ')
                new_scale = [str(scale * float(mesh_scale))
                             for mesh_scale in mesh_scales]
                mesh.attributes['scale'].value = (
                    ' '.join(new_scale)
                )

                # Also offset the origin (position) by scale
                geom_node = mesh.parentNode.parentNode
                origin = geom_node.getElementsByTagName('origin')
                for o in origin:
                    old_xyz = o.attributes['xyz'].value.split(' ')
                    new_xyz = [str(scale * float(oo)) for oo in old_xyz]
                    o.attributes['xyz'].value = (
                        ' '.join(new_xyz)
                    )
            with open(f'{tmpdir}/{key}.urdf', "w") as f:
                dom.writexml(f)
        return data
