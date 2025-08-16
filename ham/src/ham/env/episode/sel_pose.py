#!/usr/bin/env python3

from typing import Tuple, Dict

import torch as th

from ham.util.torch_util import dcn
from ham.util.path import get_path
from ham.util.math_util import invert_pose_tq, xyzw2wxyz
from ham.env.episode.spec import DefaultSpec
from ham.env.episode.util import upsert


class SelPose(DefaultSpec):
    """
    Drop-in replacement for `CollisionFree` that does _not_
    override the franka poses given by RobotDof.
    """

    @property
    def reset_deps(self) -> Tuple[str, ...]: return ('reset_ids',
                                                     'obj_poses',
                                                     'goal_poses',
                                                     )

    @property
    def reset_keys(self) -> Tuple[str, ...]: return (
        'obj_pose',
        'goal_pose',
    )

    def sample_reset(self,
                     ctx, data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        reset_ids = data['reset_ids']
        num_reset: int = len(reset_ids)
        if num_reset <= 0:
            return data

        # Compute target positions
        # (also, relative to the robot)
        init_poses = data['obj_poses'][reset_ids]
        goal_poses = data['goal_poses'][reset_ids]

        init_pose = init_poses[..., 0, :]
        goal_pose = goal_poses[..., 0, :]

        # And then update data.
        upsert(data, reset_ids, 'obj_pose', init_pose)
        upsert(data, reset_ids, 'goal_pose', goal_pose)

        return data
