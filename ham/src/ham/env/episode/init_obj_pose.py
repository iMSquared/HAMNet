#!/usr/bin/env python3

from typing import Tuple, Dict
import torch as th

from ham.env.episode.spec import DefaultSpec
from ham.env.episode.util import upsert
from ham.env.scene.sample_pose import (
    z_from_q_and_hull,
)


class InitObjectPose(DefaultSpec):

    @property
    def reset_keys(self) -> Tuple[str, ...]: return (
        # sampled xyz + orientation
        'obj_poses',
    )

    @property
    def reset_deps(self) -> Tuple[str, ...]: return (
        'reset_ids',
        'curr_pose_meta',
        'table_dim',
        'table_pos',
        'obj_radius',
        'obj_stable_pose',
        'obj_hull',

        'which_pose',
        'obj_orn'
    )

    def sample_reset(
            self, ctx, data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        gen = ctx['gen']
        ws_height = ctx['max_ws_height']

        reset_ids = data['reset_ids']
        num_reset: int = len(reset_ids)
        if num_reset <= 0:
            return data
        pose_meta = data['curr_pose_meta']

        table_dim = data['table_dim'][reset_ids]
        table_pos = data['table_pos'][reset_ids]
        obj_radius = data['obj_radius'][reset_ids]
        foot_radius = data['obj_foot_radius'][reset_ids]
        # workspace = data['workspace'][reset_ids]

        base_height = (table_pos[..., 2] + 0.5 * table_dim[..., 2])
        # NOTE(ycho): `clear_xy` here is necessary
        # since `pose_meta` does _not_ include (and
        # currently _cannot_ include) table_pos[...,:2]
        # information (i.e. result of table pos. DR)
        clear_xy = th.as_tensor([0, 0, 1],
                                dtype=th.float32,
                                device=ctx['device'])
        ws_lo = table_pos * clear_xy[None] - 0.5 * table_dim
        ws_lo[..., 2] = base_height - 0.01
        ws_hi = table_pos * clear_xy[None] + 0.5 * table_dim
        ws_hi[..., 2] = (base_height + ws_height)
        workspace = th.stack([ws_lo, ws_hi], dim=-2)

        which_pose = data['which_pose'][reset_ids]
        quat = data['obj_orn'][reset_ids]

        foot_radius_samples = th.take_along_dim(foot_radius,
                                                which_pose,
                                                dim=-1)
        num_samples = ctx['samples_per_rejection']
        foot_radius_samples = foot_radius_samples.expand(
            *foot_radius_samples.shape[:-1],
            num_samples)
        quat = quat.expand(*quat.shape[:-2], num_samples, -1)
        xyz = gen.place(workspace,
                        table_dim[..., :3],
                        foot_radius_samples,
                        meta=pose_meta,
                        high_scale=(1.0 / ctx['high_bias']),
                        init=True)
        xyz = xyz.squeeze(dim=-2)

        # As of now, xyz[...,2] is sampled from the table surface.
        # We need to offset the z value by the object's origin.z offset, like this:
        # NOTE(ycho): can't use object_height since it is p2p
        # FIXME(ycho): doesn't this need _which_pose_?
        xyz[..., 2] = z_from_q_and_hull(
            quat,
            data['obj_hull'][reset_ids, ..., None, :, :],
            xyz[..., 2]
        )
        # offset by table pose DR...
        xyz[..., :2] += table_pos[..., None, :2]
        # additionally lift up a little bit based on z_eps.
        xyz[..., 2] += ctx['z_eps']

        pose = th.cat([xyz, quat], dim=-1)

        upsert(data, reset_ids, 'obj_poses', pose)
        upsert(data, reset_ids, 'which_pose', which_pose)

        return data

    def apply_reset(self,
                    ctx,
                    data: Dict[str, th.Tensor]):
       reset_ids = data['reset_ids']
       if len(reset_ids) <= 0:
           return data
       set_ids = data['obj_id'][reset_ids]
       root_tensor = ctx['root_tensor']
       root_tensor[set_ids, :7] = data['obj_pose'][reset_ids]
       return data
