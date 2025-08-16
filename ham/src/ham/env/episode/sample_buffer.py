#!/usr/bin/env python3

from typing import Iterable, Tuple, Dict
from collections import defaultdict

import networkx as nx
import torch as th
import numpy as np

from ham.env.episode.spec import (Spec, DefaultSpec, WrapSpec)
from ham.env.episode.util import upsert
from ham.models.common import map_tensor
from ham.util.torch_util import dcn


def map_array(src, dst, op, path=None):
    if isinstance(src, dict):
        if dst is None:
            dst = {k: None for k in src.keys()}
        for k, v in src.items():
            subpath = None
            if path is not None:
                subpath = F'{path}.{k}'
            dst[k] = map_array(v, dst.get(k, None),
                               op, path=subpath)
        return dst

    if isinstance(src, (list, tuple)):
        if dst is None:
            dst = [None for _ in src]
        for i, v in enumerate(src):
            subpath = None
            if path is not None:
                subpath = F'{path}[{i}]'
            dst[i] = map_array(v, dst[i],
                               op, path=subpath)
        return dst

    try:
        return op(src, dst, path=path)
    except Exception as e:
        print(F'Problem in {path} = {e}')
        raise


class SampleBuffer(WrapSpec):
    def __init__(self, base: Spec, file: str,
                 tight_prior: float,
                 qrand_prior: float):
        super().__init__([base])
        self.base = self.specs[0]

        with open(file, 'rb') as fp:
            data = th.load(fp, map_location='cpu')
        self.data = data

        self._k_setup = tuple(set(self.data.keys()).intersection(
            self.base.setup_keys))
        self._k_reset = tuple(set(self.data.keys()).intersection(
            self.base.reset_keys)) + ('buf_idx',)

        # Initialize selection probability to ones
        # self._p_sel = self.data['has_dof'].float()
        has_ceil = self.data['pose_meta']['case']['has_ceil']
        ceil_height = self.data['pose_meta']['case']['height']
        bump_height = self.data['pose_meta']['base']['height'].amax(
            dim=(-2, -1))
        is_tight = th.logical_and(has_ceil, (ceil_height - bump_height) < 0.3)
        self.data['is_tight'] = is_tight
        self._setup_p_sel(tight_prior, qrand_prior)

    @property
    def setup_keys(self) -> Tuple[str, ...]: return self._k_setup

    @property
    def setup_deps(self) -> Tuple[str, ...]: return self.base.setup_deps

    @property
    def reset_keys(self) -> Tuple[str, ...]: return self._k_reset

    @property
    def reset_deps(self) -> Tuple[str, ...]: return self.base.reset_deps

    def _setup_p_sel(self,
                     tight_prior,
                     qrand_prior,
                     device=None):
        is_tight = self.data['is_tight']
        if 'is_near' in self.data:
            is_near = self.data['is_near']
            self._p_sel = (
                self.data['has_dof'].float()
                * th.where(is_tight, tight_prior, 1.0)
                * th.where(is_near, 1.0, qrand_prior)
            )
        else:
            self._p_sel = (
                self.data['has_dof'].float()
                * th.where(is_tight, tight_prior, 1.0)
            )
        if device is not None:
            self._p_sel = self._p_sel.to(device=device)
        print(self._p_sel.min(), self._p_sel.max())

    def _sample(self, ctx, data, keys, setup: bool):
        device = ctx['device']
        reset_ids = data['reset_ids']
        num_reset: int = len(reset_ids)

        # Move to device
        # somewhat recursively.
        if setup:
            def _to_device(src, dst, path=''):
                if isinstance(src, th.Tensor):
                    src = src.to(device=device)
                    if dst is None:
                        return src
                    dst[...] = src
                    return dst
                return src
            self.data = map_array(self.data, None,
                                  op=_to_device,
                                  path='')
            self._p_sel = self._p_sel.to(device=device)
        if num_reset <= 0:
            return data

        # Select subset (and make a new
        # modifiable dict as a copy)
        cur_data = {k: self.data[k]
                    for k in keys
                    if (k in self.data)}

        # FIXME(ycho): hardcoded `has_dof` as the
        # indicator variable for having data...!
        # has_data = self.data['has_dof'].swapaxes(0, 1)[reset_ids].float()
        sel_prob = self._p_sel.swapaxes(0, 1)[reset_ids]
        sample_index = th.multinomial(
            sel_prob, 1).squeeze(dim=-1)

        def _update_data(src, dst, path=''):
            # == skip updates for consts ==

            if isinstance(src, th.Tensor):
                src = src.detach().clone()

            I = reset_ids
            I0 = sample_index
            if isinstance(src, np.ndarray):
                I = dcn(I)
                I0 = dcn(I0)

            if path in ['.object_files',
                        '.obj_ctx']:
                return None

            if dst is None:
                return src[I0, I]

            dst[I] = src[I0, I]
            return dst

        cur_data.pop('curr_pose_meta', None)
        object_files = cur_data.pop('object_files', None)
        obj_ctx = cur_data.pop('obj_ctx', None)
        old_buf_idx = cur_data.pop('buf_idx', None)
        new_data = map_array(src=cur_data,
                             dst=data,
                             op=_update_data,
                             path='')

        # Add the skipped fields here...
        if object_files is not None:
            new_data['object_files'] = object_files
        if obj_ctx is not None:
            new_data['obj_ctx'] = obj_ctx

        # FIXME(ycho): hardcoded cache update...
        if not setup:
            def _select_reset(src, dst):
                return src[reset_ids]
            new_data['curr_pose_meta'] = map_tensor(
                src=new_data['pose_meta'],
                op=_select_reset)

        # NOTE(ycho): below code is not _strictly_ necessary
        # update (out-of-place-ish)
        for k in self.reset_keys:
            if k not in new_data:
                continue
            data[k] = new_data[k]
        upsert(data, reset_ids, 'buf_idx', sample_index)

        return data

    def sample_setup(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        return self._sample(ctx, data,
                            self.setup_keys, setup=True)

    def sample_reset(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        return self._sample(ctx, data,
                            self.reset_keys, setup=False)

    def apply_setup(self,
                    ctx,
                    data: Dict[str, th.Tensor]):
        return self.base.apply_setup(ctx, data)

    def apply_reset(self,
                    ctx,
                    data: Dict[str, th.Tensor]):
        return self.base.apply_reset(ctx, data)
