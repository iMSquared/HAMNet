#!/usr/bin/env python3
from typing import Tuple, Dict
import torch as th
from icecream import ic

from ham.util.torch_util import randu, dcn
from ham.env.episode.spec import DefaultSpec
from ham.env.common import apply_domain_randomization
from ham.env.episode.util import upsert


class PhysProp(DefaultSpec):
    """
    Physics properties
    """

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.__counter = 0

    @property
    def setup_keys(self) -> Tuple[str, ...]: return ('obj_mass',
                                                     'obj_friction',
                                                     'table_friction',
                                                     'obj_restitution')

    @property
    def reset_keys(self) -> Tuple[str, ...]: return ('obj_friction',
                                                     'table_friction',
                                                     'obj_restitution')

    @property
    def setup_deps(self) -> Tuple[str, ...]: return ('obj_ctx',
                                                     'object_scale',
                                                     'rel_scale',
                                                     )

    @property
    def reset_deps(self) -> Tuple[str, ...]: return ('reset_ids',)

    def _sample(self, keys, ctx,
                data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        reset_ids = data['reset_ids']
        mass_lo, mass_hi = ctx['obj_mass_bound']
        ofrc_lo, ofrc_hi = ctx['obj_friction_bound']
        ores_lo, ores_hi = ctx['obj_restitution_bound']
        tfrc_lo, tfrc_hi = ctx['table_friction_bound']

        for key in keys:
            lo, hi = ctx[F'{key}_bound']
            if 'obj' in key:
                shape = (len(reset_ids), ctx['num_obj_per_env'])
            else:
                shape = (len(reset_ids),)

            upsert(data, reset_ids, key,
                   randu(lo, hi, shape,
                         device=ctx['device']))
        return data

    def _apply(self, ctx, data,
               setup: bool = False):
        reset_ids = data['reset_ids']
        obj_ctx = data['obj_ctx']
        if 'barrier_handle' not in ctx:
            return False
        barrier_handless = ctx['barrier_handle']

        obj_mass_np = dcn(data['obj_mass'])
        # obj_scale_np = dcn(data['object_scale'])
        obj_scale_np = dcn(data['rel_scale'])
        obj_friction_np = dcn(data['obj_friction'])
        obj_restitution_np = dcn(data['obj_restitution'])
        table_friction_np = dcn(data['table_friction'])

        for _i in range(len(reset_ids)):
            i = reset_ids[_i]

            # == object ==
            for j in range(ctx['num_obj_per_env']):
                # print('_change_scale_(object)')
                apply_domain_randomization(
                    ctx['gym'], ctx['envs'][i],
                    ctx['obj_handle'][i, j],

                    enable_mass=(setup and ctx['use_mass_dr']),
                    min_mass=obj_mass_np[i, j],
                    max_mass=obj_mass_np[i, j],
                    use_mass_set=False,
                    mass_set=ctx.get('mass_set', None),

                    change_scale=(setup and ctx['use_scale_dr']),
                    min_scale=obj_scale_np[i],
                    max_scale=obj_scale_np[i],

                    enable_friction=True,
                    min_friction=obj_friction_np[i, j],
                    max_friction=obj_friction_np[i, j],

                    enable_restitution=True,
                    min_restitution=obj_restitution_np[i, j],
                    max_restitution=obj_restitution_np[i, j],

                    enable_com=(setup and ctx['use_com_dr']),
                    com_dr_scale=ctx.get('com_dr_scale', None)
                )

            # == barrier ==
            barrier_handles = barrier_handless[i]
            for j in range(len(barrier_handles)):
                apply_domain_randomization(
                    ctx['gym'], ctx['envs'][i],
                    barrier_handles[j],

                    change_scale=(setup),
                    min_scale=1.0,
                    max_scale=1.0,

                    enable_friction=True,
                    min_friction=table_friction_np[i],
                    max_friction=table_friction_np[i]
                )
        return True

    def sample_setup(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        self._sample(self.setup_keys, ctx, data)
        self.__counter = 0
        return data

    def apply_setup(self,
                    ctx,
                    data: Dict[str, th.Tensor]):
        applied = False
        applied = self._apply(ctx, data, setup=True)
        if applied:
            self.__counter = ctx['dr_period']
        return data

    def sample_reset(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        self.__counter -= 1
        if self.__counter <= 0:
            return self._sample(self.reset_keys, ctx, data)
        return data

    def apply_reset(self,
                    ctx,
                    data: Dict[str, th.Tensor]):
        applied = False
        if self.__counter <= 0:
            applied = self._apply(ctx, data, setup=False)
        if applied:
            self.__counter = ctx['dr_period']
        return data
