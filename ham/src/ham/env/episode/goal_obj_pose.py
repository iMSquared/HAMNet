#!/usr/bin/env python3

from typing import Tuple, Dict
import torch as th
from functools import partial
from icecream import ic

from ham.models.common import map_tensor
from ham.env.episode.spec import DefaultSpec
from ham.env.episode.util import upsert
from ham.util.math_util import quat_multiply
from ham.util.torch_util import randu
from ham.env.task.reward_util import (
    pose_error
)
from ham.env.scene.sample_pose import (
    SampleInitialOrientation,
    SampleRandomOrientation,
    SampleCuboidOrientation,
    SampleStableOrientation,
    SampleMixtureOrientation,
    RandomizeYaw,
    z_from_q_and_hull,
    sample_yaw,
    rejection_sample
)


class GoalObjectPose(DefaultSpec):

    @property
    def reset_keys(self) -> Tuple[str, ...]: return (
        # These _used_ to be included in `push_task.py`.
        'goal_radius',
        'goal_angle',
        'goal_poses',
    )

    @property
    def reset_deps(self) -> Tuple[str, ...]: return ('reset_ids',
                                                     'curr_pose_meta',
                                                     'table_dim',
                                                     'table_pos',
                                                     'obj_radius',
                                                     'obj_stable_pose',
                                                     'obj_hull',

                                                     # :thinking_face:
                                                     'obj_poses',
                                                     'which_pose'
                                                     )

    def sample_reset(
            self, ctx, data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        gen = ctx['gen']
        device = ctx['device']
        ws_height = ctx['max_ws_height']
        num_samples = ctx['samples_per_rejection']
        rand_yaw = ctx.get('randomize_yaw', True)
        reset_ids = data['reset_ids']

        num_reset: int = len(reset_ids)
        if num_reset <= 0:
            return data

        pose_meta = data['curr_pose_meta']
        table_dim = data['table_dim'][reset_ids]
        table_pos = data['table_pos'][reset_ids]
        obj_radius = data['obj_radius'][reset_ids]

        # Generate new goal threshold configuration...
        r_lo, r_hi = ctx['goal_radius_bound']
        a_lo, a_hi = ctx['goal_angle_bound']
        new_goal_radius = randu(r_lo, r_hi,
                                (num_reset,),
                                device=ctx['device'])
        new_goal_angle = randu(a_lo, a_hi,
                               (num_reset,),
                               device=ctx['device'])
        upsert(data, reset_ids,
               'goal_radius',
               new_goal_radius)
        upsert(data, reset_ids,
               'goal_angle',
               new_goal_angle)

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

        # == sample orientation ==
        num_reset: int = len(reset_ids)
        stable_pose = data['obj_stable_pose']  # [reset_ids]
        sample_q = SampleStableOrientation(lambda: stable_pose)
        if rand_yaw:
            sample_q = RandomizeYaw(sample_q,
                                    device=ctx['device'])

        q_aux = {}
        yaw_only_mask = data.get('obj_yaw_only', None)

        quat = sample_q(reset_ids, (num_reset, num_samples), aux=q_aux)
        which_pose = q_aux['pose_index']

        if yaw_only_mask is not None:
            q_z = sample_yaw((num_reset * num_samples), device=device).reshape(
                num_reset, num_samples, 4)
            src_quat = data['obj_poses'][reset_ids, ...,
                                         3:7].expand(quat.shape)
            quat = th.where(
                yaw_only_mask[reset_ids, None, None],
                # perturb-yaw
                quat_multiply(q_z, src_quat),
                quat
            )
            which_pose = th.where(
                yaw_only_mask[reset_ids, None],
                data['which_pose'][reset_ids],
                which_pose
            )
        foot_radius = th.take_along_dim(
            data['obj_foot_radius'][reset_ids],
            which_pose,
            -1)

        # -- rejection sampling block --
        def sample_fn():
            # == sample xyz ==
            xyz = gen.place(workspace,
                            table_dim[..., :3],
                            foot_radius,
                            meta=pose_meta,
                            high_scale=ctx['high_bias'],
                            init=False)
            xyz = xyz.squeeze(dim=-2)

            # == query height ==
            obj_height = th.take_along_dim(
                data['obj_height'][reset_ids],
                which_pose,
                -1)
            # == return a bunch of crap ==
            return th.cat([xyz, quat, obj_height[..., None]],
                          dim=-1).swapaxes(0, 1)

        def accept_fn(pose):
            src_pose = data['obj_poses'][reset_ids].swapaxes(
                0, -2).expand((*pose.shape[:-1], 7))
            pos_err, orn_err = pose_error(
                pose[..., 0:3], src_pose[..., 0:3],
                pose[..., 3:7], src_pose[..., 3:7]
            )

            is_far = th.logical_or(
                pos_err > data['goal_radius'][reset_ids],
                orn_err > data['goal_angle'][reset_ids],
            )
            obj_height = pose[..., 7]  # pose[..., 2] + pose[..., 7]
            has_ceil = data['pose_meta']['case']['has_ceil'][reset_ids]
            ceil_height = data['pose_meta']['case']['height'][reset_ids]
            # ic(obj_height.amin(),
            #   obj_height.amax(),
            #   ceil_height.amin(),
            #   ceil_height.amax())
            is_below_ceil = th.logical_or(
                ~has_ceil, th.logical_and(
                    has_ceil,
                    obj_height < ceil_height
                )
            )

            return th.logical_and(is_far, is_below_ceil)

        pose_h = rejection_sample(sample_fn,
                                  accept_fn,
                                  batched=True,
                                  sample=True,
                                  n_per_batch=num_samples
                                  )
        # ------------------------ --

        xyz = pose_h[..., 0:3].swapaxes(0, 1)
        quat = pose_h[..., 3:7].swapaxes(0, 1)

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

        pose = th.cat([xyz, quat], dim=-1)
        upsert(data, reset_ids, 'goal_poses', pose)
        return data
