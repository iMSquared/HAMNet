#!/usr/bin/env python3

from typing import Tuple, Dict
import torch as th

from ham.env.episode.spec import DefaultSpec
from ham.env.episode.util import upsert
from ham.util.torch_util import dcn


class LookupProp(DefaultSpec):
    @property
    def reset_keys(self) -> Tuple[str, ...]: return (
        # name.
        # Currently, this is the only non-tensor
        # quantity (numpy array)
        # Consider fixed-sized uint8 array
        'obj_name',

        # New thing I guess?
        # 'obj_mass',

        # max point-to-point distance
        # for two points within the object
        'obj_radius',
        'obj_hull',
        'obj_bbox',
        'obj_stable_pose',
        'obj_stable_mask',

        # point cloud &
        # corresponding normal vectors
        # of the object.
        'obj_cloud',
        'obj_normal',

        # foot radius at a given pose.
        'obj_foot_radius',
        # height at a given pose.
        'obj_height',

        # only z-axis rotation
        # is afforded for this object.
        'obj_yaw_only',
    )

    @property
    def reset_deps(self) -> Tuple[str, ...]: return (
        # _DUMMY_
        'obj_ctx',
        'reset_ids',
        # results of the selection...
        # in terms of isaac gym indices.
        'obj_offset',
        'obj_id',
        'obj_body_id',

    )

    def sample_reset(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        reset_ids = data['reset_ids']
        num_reset: int = len(reset_ids)
        if num_reset <= 0:
            return data

        reset_ids = data['reset_ids']
        offsets = data['obj_offset'][reset_ids]
        nxt = {}
        obj_ctx = data['obj_ctx']

        lut_indices = (
            reset_ids * ctx['num_obj_per_env']
            + offsets
        )
        lut_indices_np = dcn(lut_indices)
        for k in self.reset_keys:
            if ((k in obj_ctx) and (k != 'obj_name')):
                nxt[k] = obj_ctx[k][lut_indices]
                pass

        # NOTE(ycho): we have to handle `obj_name`
        # in a very particular manner... because
        # it's a numpy array
        nxt['obj_name'] = (
            obj_ctx['obj_name'][lut_indices_np]
        )
        reset_ids_np = dcn(reset_ids)
        for k in nxt.keys():
            if k == 'obj_name':
                upsert(data, reset_ids_np, k, nxt[k])
            else:
                upsert(data, reset_ids, k, nxt[k])
        return data
