#!/usr/bin/env python3

from typing import Tuple, Dict
import torch as th
from functools import partial

from ham.env.episode.spec import DefaultSpec
from ham.env.episode.util import upsert
from ham.models.common import map_tensor
from ham.env.scene.sample_pose import (
    SampleRandomOrientation,
    SampleCuboidOrientation,
    SampleStableOrientation,
    SampleMixtureOrientation,
    RandomizeYaw,
    z_from_q_and_hull,
)


class InitObjectOrn(DefaultSpec):

    @property
    def reset_keys(self) -> Tuple[str, ...]: return (
        # sampled xyz + orientation
        'obj_orn',

        # FIXME(ycho): rename which_pose --> which_quat
        'which_pose'
    )

    @property
    def reset_deps(self) -> Tuple[str, ...]: return (
        'reset_ids',
        'obj_stable_pose',
    )

    def sample_reset(
            self, ctx, data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        gen = ctx['gen']
        reset_ids = data['reset_ids']
        num_reset: int = len(reset_ids)
        if len(reset_ids) <= 0:
            return data
        # NOTE(ycho): we cannot have multiple samples
        # for object orientation, since collision free Robot DoF
        # computation depends on a unique object orientation per env.
        # num_samples = ctx['samples_per_rejection']
        num_samples = 1

        # == Sample orientation ==
        stable_pose = data['obj_stable_pose']  # [reset_ids]
        # sample_q = SampleStableOrientation(lambda: stable_pose)
        sample_q = SampleStableOrientation(lambda: stable_pose)
        if ctx.get('randomize_yaw', True):
            sample_q = RandomizeYaw(sample_q,
                                    device=ctx['device'])
        q_aux = {}
        qs = sample_q(reset_ids,
                      (num_reset, num_samples),
                      aux=q_aux)
        which_pose = q_aux['pose_index']

        upsert(data, reset_ids, 'obj_orn', qs)
        upsert(data, reset_ids, 'which_pose', which_pose)
        return data
