#!/usr/bin/env python3

from typing import Tuple, Dict
import torch as th

from ham.env.episode.spec import DefaultSpec
from ham.env.episode.util import upsert
from ham.util.torch_util import dcn


class SelectObject(DefaultSpec):
    """
    Select object, and load
    _current_ object properties
    from the global table, based on
    the selected object.
    """
    @property
    def reset_keys(self) -> Tuple[str, ...]: return (
        # "which object was selected?"
        'obj_offset',
        # results of the selection...
        # in terms of isaac gym indices.
        'obj_id',
        'obj_handle',
        'obj_body_id',
    )

    @property
    def reset_deps(self) -> Tuple[str, ...]: return (
        # _DUMMY_
        'obj_ctx',
        'reset_ids',
    )

    def sample_reset(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        reset_ids = data['reset_ids']
        num_reset: int = len(reset_ids)
        if num_reset <= 0:
            return data

        offsets = th.randint(ctx['num_obj_per_env'],
                             size=(num_reset,),
                             device=ctx['device'])
        upsert(data, reset_ids, 'obj_offset', offsets)

        all_obj_ids = ctx['obj_id']
        all_obj_handles = ctx['obj_handle']
        all_obj_body_ids = ctx['obj_body_id']

        reset_ids = data['reset_ids']
        prv_ids = data.get('obj_id', None)
        if prv_ids is not None:
            # Make a copy, to avoid reading
            # from `nxt_id` when trying to reference
            # `prv_id`.
            prv_ids = prv_ids.detach().clone()
        nxt = {}
        nxt['obj_id'] = all_obj_ids[reset_ids, offsets]
        nxt['obj_handle'] = all_obj_handles[reset_ids, offsets]
        nxt['obj_body_id'] = all_obj_body_ids[reset_ids, offsets]
        for k in nxt.keys():
            upsert(data, reset_ids, k, nxt[k])
        data['prv_id'] = prv_ids
        return data

    def apply_reset(self,
                    ctx,
                    data: Dict[str, th.Tensor]):
        root_tensor = ctx['root_tensor']
        # reset prv_object poses to somewhere far
        prv_ids = data.get('prv_id', None)
        cur_ids = data['obj_id'].long()
        reset_ids = data['reset_ids']

        num_reset: int = len(reset_ids)
        if num_reset <= 0:
            return data

        # If >1 object in the environment, then
        # also "clear" the object from the scene
        # by placing it to a distance location.
        if (ctx['num_obj_per_env'] > 1) and (prv_ids is not None):
            pi = None if (prv_ids is None) else prv_ids.long()
            # pos, orn, lin.vel/ang.vel --> 0
            root_tensor[pi] = 0
            # NOTE(ycho): "somewhere sufficiently far away"
            root_tensor[pi, 0] = (prv_ids + 1).float() * 100.0
            root_tensor[pi, 2] = 1.0
            # (0,1,2), (3,4,5,6)
            # Set orientation to unit quaternion
            root_tensor[pi, 6] = 1

        # reset nxt_object poses/vels to zeros by default.
        root_tensor.index_fill_(0, cur_ids[reset_ids], 0.0)

        return data
