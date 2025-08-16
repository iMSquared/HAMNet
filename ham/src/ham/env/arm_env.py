#!/usr/bin/env python3

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil

from typing import Dict, Optional, Iterable, Tuple, Any
from dataclasses import dataclass
from gym import spaces
from functools import reduce

import torch as th
import numpy as np
import einops

from ham.env.env.wrap.base import (
    ObservationWrapper, add_obs_field)
from ham.env.push_env import PushEnv
from ham.env.task.push_with_arm_task import PushWithArmTask
from ham.util.config import ConfigBase
from ham.util.math_util import apply_pose_tq, matrix_from_quaternion

""" Pre-configured obs-space types """
Point = spaces.Box(-np.inf, +np.inf, (3,))
Pose = spaces.Box(-np.inf, +np.inf, (7,))
Pose6d = spaces.Box(-np.inf, +np.inf, (9,))
PoseVel = spaces.Box(-np.inf, +np.inf, (13,))
Pose6dVel = spaces.Box(-np.inf, +np.inf, (15,))
Wrench = spaces.Box(-np.inf, +np.inf, (6,))
Keypoint = spaces.Box(-np.inf, +np.inf, (24,))
Cloud512 = spaces.Box(-np.inf, +np.inf, (512, 3))
FrankaDofPosNoGripper = spaces.Box(-2 * np.pi, +2 * np.pi, (7,))
FrankaDofPosVelNoGripper = spaces.Box(-2 * np.pi, +2 * np.pi, (14,))
# in case we implement mimic joints, should be (9,) -> (8,)
FrankaDofPosWithGripper = spaces.Box(-2 * np.pi, +2 * np.pi, (9,))
# in case we implement mimic joints, should be (16,) -> (15,)
FrankaDofPosVelWithGripper = spaces.Box(-2 * np.pi, +2 * np.pi, (16,))
Mass = spaces.Box(-np.inf, +np.inf, (1,))
PhysParams = spaces.Box(-np.inf, +np.inf, (5,))

OBS_SPACE_MAP = {
    'point': Point,
    'pose': Pose,
    'pose6d': Pose6d,
    # FIXME(ycho): not entirely correct
    'relpose': Pose,
    # FIXME(ycho): not entirely correct
    'relpose6d': Pose6d,
    'pose_vel': PoseVel,
    'pose6d_vel': Pose6dVel,
    'keypoint': Keypoint,
    'cloud': Cloud512,
    'pos7': FrankaDofPosNoGripper,
    'pos9': FrankaDofPosWithGripper,
    'pos_vel7': FrankaDofPosVelNoGripper,
    'pos_vel9': FrankaDofPosVelWithGripper,
    'wrench': Wrench,
    'mass': Mass,
    'phys_params': PhysParams,
    'none': None
}


def _sanitize_bounds(b) -> Tuple[Any, Any]:
    if b is None:
        return None
    try:
        assert (len(b) == 2)
    except AssertionError as e:
        print(F"sanitization failed : {e}")
        raise

    # First try explicit ~scalar conversion
    # silently ignore exceptions
    try:
        b0 = float(b[0])
        b1 = float(b[1])
        return (b0, b1)
    except TypeError as e:
        pass

    # Then try iterable (list) convention
    # silently ignore exceptions
    try:
        b0 = [float(x) for x in b[0]]
        b1 = [float(x) for x in b[1]]
    except TypeError as e:
        pass

    # Ensure any iterables are given as tuples
    if isinstance(b0, Iterable):
        b0 = tuple(b0)
    if isinstance(b1, Iterable):
        b1 = tuple(b1)

    return (b0, b1)


def _merge_bounds(*args):
    s = _sanitize_bounds
    args = [s(x) for x in args]
    return reduce(lambda a, b: (tuple(a[0]) + tuple(b[0]),
                                tuple(a[1]) + tuple(b[1])),
                  args)


def _identity_bound(shape):
    bound = np.zeros(shape), np.ones(shape)
    return _sanitize_bounds(bound)


def _get_obs_bound_map():
    """
    Default observation bounds.

    NOTE(ycho): we independently maintain obs_bound_map
    instead of just using the bounds from `spaces.Box`
    to avoid assuming the symmetry of the input-spaces
    during normalization: x' = (x-c)/s, c != (lo+hi)/2
    """
    point = ((0.0, 0.0, 0.55), (0.4, 0.4, 0.4))
    vector = ((0.0, 0.0, 0.0), (0.4, 0.4, 0.4))
    color = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    quat = ([0] * 4, [1] * 4)
    sixd = ([0] * 6, [1] * 6)
    rot_6d = ([0] * 6, [1] * 6)
    lin_vel = ([0] * 3, [1] * 3)
    ang_vel = ([0] * 3, [1] * 3)
    pose = _merge_bounds(point, quat)
    pose6d = _merge_bounds(point, sixd)
    relpose = _merge_bounds(vector, quat)
    relpose6d = _merge_bounds(vector, sixd)
    pose_vel = _merge_bounds(pose, lin_vel, ang_vel)
    pose6d_vel = _merge_bounds(pose6d, lin_vel, ang_vel)
    keypoint = (point[0] * 8, point[1] * 8)  # flattened for some reason...
    cloud = point  # exploits broadcasting
    color_cloud = _merge_bounds(point, color)
    pos7 = ([0] * 7, [1] * 7)
    pos9 = ([0] * 9, [1] * 9)
    vel7 = ([0] * 7, [1] * 7)
    vel9 = ([0] * 9, [1] * 9)
    pos_vel7 = _merge_bounds(pos7, vel7)
    pos_vel9 = _merge_bounds(pos9, vel9)
    force = ([0] * 3, [30] * 3)
    torque = ([0] * 3, [1] * 3)

    wrench = _merge_bounds(force, torque)
    # FIXME(ycho): the ranges here are set arbitrarily !!
    mass = ([0], [1])
    friction = ([0], [2])
    phys_params = ([0.5, 1.0, 1.0, 1.0, 0.5], [0.5, 1.0, 1.0, 1.0, 0.5])
    phys_params_w_com = _merge_bounds(phys_params, point)
    env_type = ([0.5] * 4, [1] * 4)  # one-hot
    env_type_and_param = ([0.0] * 11, [1] * 11)  # id(4) + params(7)
    env_code = (
        # mean
        []
        + [0.0] * 4  # -table_dim/2 ~ +table_dim/2
        + [np.deg2rad(22.5)] * 4  # 0 ~ 45
        + [0.075] * 6  # 0 ~ 0.15
        + [0.175] * 4  # 0.05 ~ 0.3?? doesn't work when `has_ceil`=True
        + [0.25] * 1
        # approximately table_dim * 0.75, assuming DR from (0.5 ~ 1.0)
        + [0.51 * 0.75, 0.65 * 0.75, 0.4]
        + [0, 0, 0.2]  # table pos mean
        ,
        # std?
        []
        + [0.25] * 4  # approximately table_dim / 2
        + [np.deg2rad(22.5)] * 4
        + [0.075] * 6
        + [0.125] * 4
        + [0.25] * 1
        # approximately table_dim * 0.25, assuming DR from (0.5 ~ 1.0)
        + [0.51 * 0.25, 0.65 * 0.25, 0.4 * 0.25]
        + [0.1, 0.1, 0.3]  # table pos DR (placeholder) + table height DR
    )
    cloud_count = [[0.0], [1.0]]

    out = dict(
        point=point,
        quat=quat,
        rot_6d=rot_6d,

        lin_vel=lin_vel,
        ang_vel=ang_vel,
        pose=pose,
        pose6d=pose6d,
        relpose=relpose,
        relpose6d=relpose6d,
        pose_vel=pose_vel,
        pose6d_vel=pose6d_vel,
        keypoint=keypoint,
        cloud=cloud,
        pos7=pos7,
        pos9=pos9,
        vel7=vel7,
        vel9=vel9,
        pos_vel7=pos_vel7,
        pos_vel9=pos_vel9,
        force=force,
        torque=torque,
        wrench=wrench,
        mass=mass,
        friction=friction,
        phys_params=phys_params,
        phys_params_w_com=phys_params_w_com,
        env_type=env_type,
        env_type_and_param=env_type_and_param,
        env_code=env_code,
        cloud_count=cloud_count,
        color_cloud=color_cloud
    )
    out = {k: _sanitize_bounds(v)
           for (k, v) in out.items()}
    return out


OBS_BOUND_MAP = _get_obs_bound_map()


class ArmEnvWrapper(ObservationWrapper):

    @dataclass
    class Config(ConfigBase):
        # options: (point,pose,cloud,keypoint,none)
        # add_goal: bool = True
        goal_type: str = 'pose'
        # options: (pose,pose_vel,cloud,keypoint,none)
        object_state_type: str = 'pose_vel'
        # options: (pose,pose_vel,none)
        hand_state_type: str = 'pose_vel'
        # options: (pos,pos_vel,none)
        robot_state_type: str = 'pos_vel7'

    def __init__(self, cfg: Config, env):
        super().__init__(env, self._wrap_obs)

        new_obs = {
            'goal': cfg.goal_type,
            'object_state': cfg.object_state_type,
            'hand_state': cfg.hand_state_type,
            'robot_state': cfg.robot_state_type,
        }

        obs_space = env.observation_space
        update_fn = {}
        for k, t in new_obs.items():
            if isinstance(t, spaces.Space):
                s = t
            else:
                s = OBS_SPACE_MAP.get(t)
            if s is None:
                continue
            obs_space, update_fn[k] = add_obs_field(obs_space, k, s)
        self._obs_space = obs_space
        self._update_fn = update_fn

        self._get_fn = {
            'goal': self.__goal,
            'object_state': self.__object_state,
            'robot_state': self.__robot_state,
            'hand_state': self.__hand_state,
        }

    @property
    def observation_space(self):
        return self._obs_space

    def __goal(self):
        """ Get goal observation. """
        cfg = self.cfg
        # FIXME(ycho): padding `cur_cloud` is only valid
        # if using full-cloud inputs...!
        return self.__rigid_body_state(cfg.goal_type,
                                       self.task.goal,
                                       self.scene.data['obj_cloud'],
                                       self.scene.data['obj_bbox'])

    def __object_state(self):
        """ Get object-state observation. """
        cfg = self.cfg
        obj_ids = self.scene.data['obj_id'].long()  # cur_props.ids.long()
        obj_state = self.tensors['root'][obj_ids, :]
        # FIXME(ycho): padding `cur_cloud` is only valid
        # if using full-cloud inputs...!
        return self.__rigid_body_state(cfg.object_state_type,
                                       obj_state,
                                       self.scene.data['obj_cloud'],
                                       self.scene.data['obj_bbox'])

    def __robot_state(self):
        """ Get robot-state observation. """
        cfg = self.cfg
        if cfg.robot_state_type.startswith('pos_vel'):
            return self.tensors['dof'].reshape(
                self.tensors['dof'].shape[0], -1)
        elif cfg.robot_state_type.startswith('pos'):
            return self.tensors['dof'][..., :, 0]
        else:
            raise ValueError(
                F'Unknown robot_state_type={cfg.robot_state_type}')

    def __hand_state(self):
        """ Get hand-state observation. """
        cfg = self.cfg
        body_tensors = self.tensors['body']
        body_indices = self.robot.tip_body_indices.long()
        eef_pose = body_tensors[body_indices, :]
        return self.__rigid_body_state(cfg.hand_state_type,
                                       eef_pose,
                                       None,
                                       None)

    def __rigid_body_state(self,
                           obs_type: str,
                           ref_pose_vel: Optional[th.Tensor] = None,
                           ref_cloud: Optional[th.Tensor] = None,
                           ref_bbox: Optional[th.Tensor] = None):
        """ Map rigid-body states to the given observation type. """
        if obs_type in ['pose', 'relpose']:
            return ref_pose_vel[..., 0:7]
        elif obs_type == 'pose_vel':
            return ref_pose_vel[..., 0:13]
        elif obs_type in ['pose6d', 'relpose6d']:
            rot_mat = matrix_from_quaternion(ref_pose_vel[..., 3:7])
            qd = rot_mat[..., :, :2].reshape(*rot_mat.shape[:-2], -1)
            return th.cat([ref_pose_vel[..., :3], qd], dim=-1)
        elif obs_type == 'pose6d_vel':
            rot_mat = matrix_from_quaternion(ref_pose_vel[..., 3:7])
            qd = rot_mat[..., :, :2].reshape(*rot_mat.shape[:-2], -1)
            return th.cat([ref_pose_vel[..., :3],
                           qd, ref_pose_vel[..., 7:13]], dim=-1)
        elif obs_type == 'cloud':
            # ref_bbox = object-frame point cloud given as (P, 3) array
            assert (ref_cloud is not None)
            return apply_pose_tq(ref_pose_vel[..., None, 0:7], ref_cloud)
        elif obs_type == 'keypoint':
            # ref_bbox = canonical keypoints given as (K, 3) array
            # For keypoints, we also flatten the result to 1D afterwards.
            assert (ref_bbox is not None)
            out = apply_pose_tq(ref_pose_vel[..., None, 0:7], ref_bbox)
            out = einops.rearrange(out, '... k d -> ... (k d)')
            return out
        else:
            raise ValueError(F'unknown type = {obs_type}')

    def _wrap_obs(self, obs: Dict[str, th.Tensor]):
        for k, u in self._update_fn.items():
            try:
                obs = u(obs, self._get_fn[k]())
            except Exception:
                print(F'Failed to wrap {k}')
                raise
        return obs


@dataclass
class ArmEnvConfig(PushEnv.Config, ArmEnvWrapper.Config):
    task: PushWithArmTask.Config = PushWithArmTask.Config()


def make_arm_env(cfg: ArmEnvConfig, **kwds):
    # TODO(ycho): rename ArmEnv -> get_arm_env since
    # it's a function, not a class.
    task_cls = kwds.pop('task_cls', PushWithArmTask)
    env = PushEnv(cfg, task_cls=task_cls)
    env = ArmEnvWrapper(cfg, env)
    return env
