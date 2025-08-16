#!/usr/bin/env python3

from isaacgym import gymtorch

import inspect
import logging
import pickle
from typing import Tuple, Dict, Iterable, Any, Optional
from dataclasses import dataclass, replace
from ham.util.config import ConfigBase
from isaacgym.torch_utils import (
    quat_mul,
    quat_conjugate, quat_from_euler_xyz)
from functools import partial


from ham.env.task.base import TaskBase
from ham.env.env.base import EnvBase

import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch.utils.tensorboard import SummaryWriter
from ham.models.common import merge_shapes

from ham.util.torch_util import randu, randu_like
from ham.util.math_util import (quat_diff_rad,
                                quat_rotate,
                                quat_multiply,
                                quat_conjugate
                                )
from ham.env.scene.sample_pose import (
    SampleRandomOrientation,
    SampleCuboidOrientation,
    SampleStableOrientation,
    SampleMixtureOrientation,
    SampleWeightedStableOrientation,
    SampleInitialOrientation,
    RandomizeYaw,
    CuboidRoll,
    RotateOverPrimaryAxis,
    z_from_q_and_hull,
    sample_wall_xy,
    sample_bump_xy,
    sample_flat_xy,
    rejection_sample
)
from ham.env.task.reward_util import (
    pose_error,
    keypoint_error,
    geodesic_step_keypoint_error,
    PoseDistance,
    KeypointDistance
)
from ham.env.task.rewards import (
    SuccessReward,
    ReachReward,
    AddReward,
    CatReward,
)

import nvtx
from icecream import ic

LEGACY: bool = False


def potential(goal_pos: th.Tensor, obj_pos: th.Tensor) -> th.Tensor:
    """
    negative distance between two positions.
    This means, high potential = good
    """
    return -th.linalg.norm(
        obj_pos[..., :3] - goal_pos[..., :3],
        dim=-1)


class Feedback(nn.Module):
    @dataclass
    class Config:
        succ: SuccessReward.Config = SuccessReward.Config()
        reach: ReachReward.Config = ReachReward.Config()
        pose_dist: PoseDistance.Config = PoseDistance.Config()
        kpt_dist: KeypointDistance.Config = KeypointDistance.Config()
        # if you don't want `pose_goal`, then
        # just set `goal_angle` to infinity.
        # pose_goal: bool = True
        add: AddReward.Config = AddReward.Config(coef={'succ': 1.0,
                                                       'reach': 0.15})
        cat: CatReward.Config = CatReward.Config(keys=['succ', 'reach'])

        rewd_dist_type: str = 'kpt'
        rewd_reduce_type: str = ('add' if LEGACY else 'cat')
        timeout: int = 0
        joint_torque_limit: Optional[str] = None
        # Hacky config for sysid
        only_timeout: bool = False

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.rewds = {}
        self.rewds['succ'] = SuccessReward()
        self.rewds['reach'] = ReachReward(cfg.reach)
        if cfg.rewd_reduce_type == 'add':
            self.rewd = AddReward(cfg.add, self.rewds)
        elif cfg.rewd_reduce_type == 'cat':
            self.rewd = CatReward(cfg.cat, self.rewds)
        else:
            raise ValueError(F'Unknown reduce={cfg.rewd_reduce_type}')
        self.pose_dist = PoseDistance(cfg.pose_dist)
        self.kpt_dist = KeypointDistance(cfg.kpt_dist)

    # @th.compile
    def forward(self, *,
                step: th.Tensor,
                # pose-vel states
                prev_obj_state: th.Tensor,
                curr_obj_state: th.Tensor,
                goal: th.Tensor,
                goal_radius: th.Tensor,
                goal_angle: th.Tensor,
                bound_lo: th.Tensor,
                bound_hi: th.Tensor,
                kpts: Optional[th.Tensor],
                joint_torques: Optional[th.Tensor],
                max_joint_torques: Optional[th.Tensor]
                ):
        cfg = self.cfg
        path = th.stack([prev_obj_state,
                         curr_obj_state], dim=0)
        goal = goal[None].expand(path.shape)
        pose_dist = self.pose_dist(path=path,
                                   goal=goal)
        succ = th.logical_and(
            pose_dist[1, ..., 0] <= goal_radius,
            pose_dist[1, ..., 1] <= goal_angle
        )
        pos = curr_obj_state[..., :3]
        oob = th.logical_or((pos < bound_lo).any(dim=-1),
                            (pos >= bound_hi).any(dim=-1))
        timeout = (step >= cfg.timeout)
        if not cfg.only_timeout:
            done = (succ | oob | timeout)
        else:
            done = timeout
        if cfg.joint_torque_limit == 'kill':
            exceed = (joint_torques >= max_joint_torques).any(-1)
            done |= exceed
        kpt_dist = self.kpt_dist(kpts=kpts,
                                 path=path,
                                 goal=goal)
        dist = (kpt_dist if cfg.rewd_dist_type == 'kpt'
                else pose_dist)
        rewd, info = self.rewd(path=path,
                               goal=goal,
                               dist=dist,
                               succ=succ,
                               done=done)
        return (rewd, done, succ, timeout, info)


def compute_workspace(pos, dim,
                      max_height: float = 0.3,
                      margin: float = 0.1):
    # max bound
    max_bound = th.add(
        pos,
        th.multiply(0.5, dim))
    max_bound[..., :2] += margin
    max_bound[..., 2] += max_height

    # min bound
    min_bound = th.subtract(
        pos,
        th.multiply(0.5, dim))
    min_bound[..., :2] -= margin
    min_bound[..., 2] = (
        max_bound[..., 2] - max_height
        - 0.5 * dim[..., 2]
    )

    # workspace
    ws_lo = min_bound
    ws_hi = max_bound
    return (ws_lo, ws_hi)


class PushTask(TaskBase):

    @dataclass
    class Config(ConfigBase):
        goal_radius: float = 1e-1
        # goal_angle: float = float(np.deg2rad(30))
        use_pose_goal: bool = False
        goal_angle: float = float('inf')
        timeout: int = 1024
        sparse_reward: bool = False
        contact_thresh: float = 1e-2
        randomize_goal: bool = False
        goal_type: str = 'stable'
        # Only used if goal_type == stable
        randomize_yaw: bool = False

        # Use keypoint based reward or not
        use_keypoint: bool = False
        planar: bool = True
        # Use potential based reward or inverse
        use_potential: bool = True
        epsilon: float = 0.02

        # "potential reward" coefficient
        # between the object and the goal.
        pot_coef: float = 1.0

        # `fail_coef` is negated in practice :)
        fail_coef: float = 1.0
        succ_coef: float = 1.0
        time_coef: float = 0.001

        # Check whether or not the object has
        # stopped at the goal, based on
        # max-velocity threshold.
        check_stable: bool = False
        max_speed: float = 1e-1

        # Workspace height
        ws_height: float = 0.5
        ws_margin: float = 0.3

        # Minimum separation distance
        # between the object CoM and the goal.
        min_separation_scale: float = 1.0
        max_separation_scale: float = float('inf')
        eps: float = 1e-6

        sample_thresh: Tuple[float, float] = (1.0, 1.0)
        margin_scale: float = 0.95

        use_log_reward: bool = False
        use_exp_reward: bool = False
        k_1: float = 0.37
        k_2: float = 27.2
        gamma: float = 0.995
        mode: str = 'train'

        filter_stable_goal: bool = True
        calc_cross: bool = False
        calc_task_type: bool = False
        lift_thresh: float = 0.05
        high_thresh: float = 0.45
        hack_climb_only: bool = False

        hack_pos_only: bool = False

        avoid_direct_suc: bool = False
        samples_per_rejection: int = 64

        feedback: Feedback.Config = Feedback.Config()

        import_task: Optional[str] = None
        export_task: Optional[str] = None

        joint_torque_limit: Optional[str] = None  # or kill or penalty
        target_joints: Tuple[int, ...] = (4, 5, 6)
        override_joint_limit: Optional[float] = None

        def __post_init__(self):
            self.feedback = replace(self.feedback,
                                    timeout=self.timeout)
            self.feedback = replace(self.feedback,
                                    joint_torque_limit=self.joint_torque_limit)

    def __init__(self, cfg: Config, writer: Optional[SummaryWriter] = None):
        super().__init__()
        self.cfg = cfg
        self.goal: th.Tensor = None
        self.cross: th.Tensor = None
        # FIXME(ycho): fix intrusive `task_type` logic.
        # It might be better if we can do something more like
        # task_type = cat(scene_type, _task_type)
        self.task_type: th.Tensor = None

        # Cache previous object positions for calculating potential-based
        # rewards
        self.__prev_obj_pose: th.Tensor = None
        self.__has_prev: th.Tensor = None

        # Adaptive goal thresholds
        self.goal_radius: float = cfg.goal_radius
        self.goal_angle: float = cfg.goal_angle
        self.max_speed: float = cfg.max_speed
        self.gamma: float = cfg.gamma

        # self.goal_radius_samples: th.Tensor = None
        # self.goal_angle_samples: th.Tensor = None
        # self.max_speed_samples: th.Tensor = None

        self.min_separation_scale = cfg.min_separation_scale
        self.max_separation_scale = cfg.max_separation_scale

        self._writer = writer
        self.feedback = Feedback(cfg.feedback)

        self.__episode = {}
        if cfg.import_task is not None:
            with open(cfg.import_task, 'rb') as fp:
                self.__episode = pickle.load(fp)

    def export(self):
        cfg = self.cfg
        if cfg.export_task is None:
            return
        with open(cfg.export_task, 'wb') as fp:
            pickle.dump(self.__episode, fp)

    @property
    def timeout(self) -> int:
        return self.cfg.timeout

    def create_assets(self, *args, **kwds):
        return {}

    def create_actors(self, *args, **kwds):
        return {}

    def create_sensors(self, *args, **kwds):
        return {}

    def setup(self, env: 'EnvBase'):
        cfg = self.cfg
        if cfg.filter_stable_goal:
            assert (env.scene.cfg.load_stable_mask)
        device: th.device = th.device(env.cfg.th_device)
        num_env: int = env.num_env
        self.env = env

        if cfg.use_pose_goal:
            # pose goal
            self.goal = th.empty(
                (num_env, 7),
                dtype=th.float,
                device=device)
        else:
            # position-only goal
            self.goal = th.empty(
                (num_env, 3),
                dtype=th.float,
                device=device)
        self.__prev_obj_pose = th.zeros(
            (num_env, 7),
            dtype=th.float,
            device=device)
        self.__has_prev = th.zeros(
            (num_env,),
            dtype=th.bool,
            device=device)

        if cfg.calc_cross:
            self.cross = th.zeros(
                (num_env,),
                dtype=th.bool,
                device=device)

        if cfg.calc_task_type:
            self.task_type = th.zeros(
                (num_env,),
                dtype=th.long,
                device=device)

        if False:
            # p = env.scene.table_pos
            # d = env.scene.table_dims
            p = env.scene.data['table_pos']
            d = env.scene.data['table_dim']
            min_bound = p - 0.5 * cfg.margin_scale * d
            max_bound = p + 0.5 * cfg.margin_scale * d
            # set min_bound also to tabletop.
            min_bound[..., 2] = max_bound[..., 2]
            self.goal_bound = th.zeros((num_env, 2, 3),
                                       dtype=th.float,
                                       device=device)
            self.goal_lo = self.goal_bound[..., 0, :]
            self.goal_hi = self.goal_bound[..., 1, :]
            self.goal_lo[...] = th.as_tensor(min_bound,
                                             dtype=th.float,
                                             device=device)
            self.goal_hi[...] = th.as_tensor(max_bound,
                                             dtype=th.float,
                                             device=device)

        self.env_lo = th.as_tensor(env.cfg.env_bound_lower,
                                   dtype=th.float,
                                   device=device)
        self.env_hi = th.as_tensor(env.cfg.env_bound_upper,
                                   dtype=th.float,
                                   device=device)

        self.ws_bound = th.zeros((num_env, 2, 3),
                                 dtype=th.float,
                                 device=device)
        self.ws_lo = self.ws_bound[:, 0]
        self.ws_hi = self.ws_bound[:, 1]

        # self.table_body_ids = th.as_tensor(
        #     env.scene.table_body_ids,
        #     dtype=th.int32,
        #     device=device)

        # self.goal_radius_samples = th.full((num_env,),
        #                                self.goal_radius,
        #                                dtype=th.float,
        #                                device=device)
        # self.goal_angle_samples = th.full((num_env,),
        #                                self.goal_angle,
        #                                dtype=th.float,
        #                                device=device)
        # self.max_speed_samples = th.full((num_env,),
        #                                self.max_speed,
        #                                dtype=th.float,
        #                                device=device)

        if cfg.joint_torque_limit:
            self.__target_joints = th.as_tensor(cfg.target_joints,
                                                dtype=th.long,
                                                device=device)

    @property
    def num_rew(self) -> int:
        # FIXME(ycho): assumes each catted-reward is of len 1!!
        cfg = self.cfg
        return self.feedback.rewd.dim

    @property
    def num_type(self) -> int:
        # FIXME(ycho): for now, return
        # based on env.scene.type
        return 1

    def compute_feedback(self,
                         env: 'EnvBase',
                         obs: th.Tensor,
                         action: th.Tensor
                         ) -> Tuple[th.Tensor, th.Tensor, Dict]:
        cfg = self.cfg
        # max_speed: float = self.max_speed
        # y_step = z_step = None
        # curr_obj = env.tensors['root'][env.scene.data['obj_id']][..., : 7]
        curr_obj = env.tensors['root'][env.scene.data['obj_id']][..., : 7]
        prev_obj = th.where(self.__has_prev[..., None],
                            self.__prev_obj_pose,
                            curr_obj)

        joint_torques = None
        max_joint_torques = None
        if cfg.joint_torque_limit:
            joint_torques = env.tensors['joint_torque'][...,
                                                        self.__target_joints]
            max_joint_torques = env.robot.eff_limits[self.__target_joints][
                None]
            if cfg.override_joint_limit:
                max_joint_torques = max_joint_torques.clamp_max(
                    cfg.override_joint_limit)
        out = self.feedback(
            step=env.buffers['step'],
            prev_obj_state=prev_obj,
            curr_obj_state=curr_obj,
            goal=self.goal,
            # self.goal_radius_samples,
            goal_radius=env.scene.data['goal_radius'],
            # self.goal_angle_samples,
            goal_angle=env.scene.data['goal_angle'],
            bound_lo=self.ws_lo,
            # cur_props.bboxes,
            bound_hi=self.ws_hi, kpts=env.scene.data['obj_bbox'],
            joint_torques=joint_torques, max_joint_torques=max_joint_torques)
        rewd, done, succ, timeout, rewards_info = out

        if LEGACY:
            rewards_info = {k: v.squeeze(dim=-1)
                            for (k, v) in rewards_info.items()}
            rewd = rewd.squeeze(dim=-1)

        # rew, done,
        # rew2, done2, suc2, timeout2 = out2
        # rew2 = rew2.squeeze(dim=-1)
        # i = (rew - rew2).abs().argmax()
        # print(rew.shape, rew2.shape, i)
        # print(rew[i], rew2[i])
        # print(env.buffers['step'][i],
        #       prev_obj_state[i],
        #       curr_obj_state[i],
        #       goal[i],
        #       goal_radius[i],
        #       bound_lo[i],
        #       bound_hi[i])
        # print((done ^ done2).any())

        info: Dict[str, Any] = {}
        info['success'] = succ
        info['timeout'] = timeout
        info['reward'] = rewards_info

        # FIXME(ycho): restore proper `env_type`
        if cfg.calc_task_type:
            # info['env_type'] = self.task_type
            info['env_type'] = th.zeros((env.num_env,),
                                        dtype=th.long,
                                        device=env.device)
        else:
            info['env_type'] = th.zeros((env.num_env,),
                                        dtype=th.long,
                                        device=env.device)

        self.__prev_obj_pose[...] = env.tensors['root'][
            # env.scene.data['obj_id'].long(),
            env.scene.data['obj_id'].long(),
            :7]
        self.__has_prev.copy_(~done)
        return (rewd, done, info)

    def calc_task_spec(self, env, indices, goal_xyz):
        cfg = self.cfg

        table_pos = env.scene.data['table_pos']
        table_dim = env.scene.data['table_dim']

        obj_pose = env.tensors['root'][
            # env.scene.data['obj_id'].long(),
            env.scene.data['obj_id'].long(),
            :7]
        rob_pose = env.tensors['root'][
            env.robot.indices.long(),
            :7]
        obpi = obj_pose[indices]
        rbpi = rob_pose[indices]

        if cfg.calc_cross:
            # self.cross[indices] = cross
            base = (
                table_pos[..., 2]
                + 0.5 * table_dim[..., 2]
            )
            self.cross[indices] = (goal_xyz[..., 2] > (base + 1e-3))
            # ^^ _not_ strictly correct I think,
            # should account for object etc etc
            # but whatever... for now

        if cfg.calc_task_type:
            # see "tabletop_with_object_scene_v2.py
            # for the meaning of `env_code`.
            # env_code = env.scene.cur_props.env_code[indices]
            # .env_code[indices]
            env_code = env.scene.data['env_code'][indices]

            height_from_robot_base = (
                # = tabletop
                table_pos[indices, ..., 2] +
                0.5 * table_dim[indices, ..., 2]
                # = subtract franka pos
                - rbpi[..., 2]
            )

            # FIXME(ycho): brittle (and hardcoded) indexing logic
            has_ceil = (env_code[..., 18] > 0)
            has_front = (env_code[..., 14] > 0)
            has_side = (env_code[..., 15:18] > 0).any(dim=-1)
            # access_type :
            # [0:free] = ~has_ceil, ~has_front
            # [1:ceil+side] = has_front, ~has_ceil, ~has_side
            # [2:ceil] = has_front, ~has_ceil, has_side
            # [3:front] = ~has_front, has_ceil, has_side
            # All other configurations are invalid, by construction.
            # * when `has_front` is `True`, `has_ceil` is always False.
            # * when `has_ceil` is True, `has_front` is always False
            #   and `has_side` is always True.
            access_type = (
                # 0: completely open (do nothing)
                # 0 * (~has_ceil & ~has_front)
                # 1: front-wall only
                + 1 * (has_front & ~has_side).long()
                # 2: front+side
                + 2 * (has_front & has_side).long()
                # 3: ceil
                + 3 * has_ceil.long()
            )

            # motion type;
            # 0 : down; 1: flat; 2: up
            motion_type = (
                # default=1 (flat)
                1
                # add 1 if going up (=>2)
                + (goal_xyz[..., 2] > (obpi[..., 2] + cfg.lift_thresh)).long()
                # subtract 1 if going down (=>0)
                - (obpi[..., 2] > (goal_xyz[..., 2] + cfg.lift_thresh)).long()
            )
            is_high = (height_from_robot_base > cfg.high_thresh).long()
            task_type = (
                # stride=:2x3=6 | 4 options for access_type
                6 * access_type
                # stride:2 | 3 options for motion_type
                + 2 * motion_type
                # stride:1 | 2 options for is_high
                + is_high
            )
            self.task_type[indices] = task_type

    def load_goal(self,
                  env: 'EnvBase',
                  indices: Iterable[int]):
        xyz = self.__episode['xyz'][indices]
        q = self.__episode['q'][indices]
        self.goal[indices, 0:3] = xyz
        self.goal[indices, 3:7] = q
        return (xyz, q)

    def sample_goal(self, env: 'EnvBase', indices: Iterable[int]):
        num_reset: int = len(indices)
        cfg = self.cfg

        if cfg.goal_type == 'sampled':
            raise ValueError('no longer supported')
            self.goal[indices, :] = (
                env.scene.cur_props.predefined_goal[indices, 0, :]
            )
            return

        self.goal[indices, :] = env.scene.data['goal_pose'][indices]
        goal_xyz = self.goal[..., :3]
        goal_quat = self.goal[..., 3:7]
        return goal_xyz, goal_quat

        if cfg.use_pose_goal:
            # == sample orientation ==
            if cfg.goal_type == 'random':
                sample_q = SampleRandomOrientation(env.device)
            elif cfg.goal_type == 'cuboid':
                sample_q = SampleCuboidOrientation(env.device)
            elif cfg.goal_type == 'stable':
                if cfg.filter_stable_goal:
                    def weights_fn():
                        EPS: float = 1e-6
                        prob = env.scene.cur_props.stable_masks.float().add(EPS)
                        prob = prob.div_(prob.sum(keepdim=True, dim=-1))
                        return prob
                    sample_q = SampleWeightedStableOrientation(
                        partial(getattr, env.scene.cur_props, 'stable_poses'),
                        weights_fn
                    )
                else:
                    sample_q = SampleStableOrientation(
                        partial(getattr, env.scene.cur_props, 'stable_poses'))
            elif cfg.goal_type == 'random+cuboid':
                sample_q = SampleMixtureOrientation(
                    [SampleRandomOrientation(env.device),
                        SampleCuboidOrientation(env.device)],
                    [1.0 - cfg.canonical_pose_prob, cfg.canonical_pose_prob])
            elif cfg.goal_type == 'yaw_only':
                sample_q = SampleInitialOrientation(env)
                if not cfg.randomize_yaw:
                    sample_q = RandomizeYaw(sample_q, env.device)
            elif cfg.goal_type == 'roll+cuboid':
                sample_q = SampleInitialOrientation(env)
                sample_q = CuboidRoll(sample_q, env.device)
            elif cfg.goal_type == 'init':
                sample_q = SampleInitialOrientation(env)
            else:
                raise ValueError(F'Unknown init_type={cfg.goal_type}')

            # == Wrap with randomization ==
            if cfg.randomize_yaw:
                sample_q = RandomizeYaw(sample_q, device=env.device)

            # == Account for yaw-only objects ==
            q_aux = {}
            if (env.scene.cur_props.yaw_only is not None and
                    env.scene.cur_props.yaw_only[indices].sum() > 0):
                is_yaw_only = env.scene.cur_props.yaw_only[indices]
                sample_q_planar = SampleInitialOrientation(env)
                if cfg.randomize_yaw:
                    sample_q_planar = RandomizeYaw(
                        sample_q_planar, device=env.device)
                q1 = sample_q(indices, num_reset, aux=q_aux)
                q2 = sample_q_planar(indices, num_reset)
                q = th.where(is_yaw_only[..., None], q2, q1)
            else:
                q = sample_q(indices, num_reset, aux=q_aux)

            obj_pose = env.tensors['root'][
                env.scene.data['obj_id'].long(),
                :7]
            obpi = obj_pose[indices]
            which_pose = q_aux['pose_index']
            foot_radius = th.take_along_dim(
                env.scene.data['obj_foot_radius'][indices],
                which_pose,
                -1)

            # == sample XY ==
            if not cfg.avoid_direct_suc:
                # _not_ avoid_direct_suc (??)
                rob_pose = env.tensors['root'][
                    env.robot.indices.long(),
                    :7]
                rbpi = rob_pose[indices]
                keepout_center = obpi[..., :2]
                keepout_radius_sq = th.square(
                    self.goal_radius_samples[indices] *
                    self.min_separation_scale)
                keepin_radius_sq = th.square(
                    self.goal_radius_samples[indices] *
                    self.max_separation_scale)

            if (self.min_separation_scale > 0) or (
                    self.max_separation_scale < float('inf')):

                def sample_fn():
                    num_samples = cfg.samples_per_rejection * num_reset
                    return env.scene._sample_pos(obj_radius=foot_radius,
                                                 num_samples=num_samples).reshape(
                        num_reset,
                        num_samples * which_pose.shape[-1],
                        -1
                    ).swapaxes(0, 1)

                def accept_keepout(xyz):
                    delta = xyz[..., :2] - keepout_center[None]
                    sqd = th.einsum('...i,...i->...', delta, delta)
                    if self.max_separation_scale < float('inf'):
                        # Also return a "preference"
                        # to select closer samples, in case
                        # `max_separation_scale` is active.
                        # In case no solution satisfies sqd < keepin_radius_sq,
                        # this will return the placement with smallest sqd.
                        # ex: keepin=1.3, sqd=[0.9, 1.5, 1.7, 3.9]
                        # near_bonus = [0.4, -0.2, -0.4, -2.6]
                        # near_bonus = keepin_radius_sq - sqd
                        # keep it positive: [3.0, 2.4, 2.2, 0.0]
                        # near_bonus += near_bonus.amin(dim=0, keepdim=True)
                        # keep it positive & normalize: [1.0, 0.8, 22/30, 0]
                        # near_bonus /= (near_bonus.amax(dim=0,
                        #               keepdim=True) + 1e-6)
                        # keep it above 0 and below 1 (cannot exceed 1)
                        # also do not add bonus if it's already below sqd < keepin
                        # [0.99, 0.8, 22/30, 0] => [0, 0.8, 22/30, 0]
                        # near_bonus.clamp_(0, 0.99).mul_(sqd > keepin_radius_sq)
                        # + near_bonus
                        return th.logical_and(keepout_radius_sq < sqd,
                                              sqd < keepin_radius_sq)
                    else:
                        return (keepout_radius_sq < sqd)

                xyz = rejection_sample(sample_fn,
                                       accept_keepout,
                                       batched=True,
                                       sample=False)

            elif cfg.avoid_direct_suc:
                num_samples = cfg.samples_per_rejection
                obj_pi = obpi[None].expand(num_samples,
                                           *obpi.shape)

                if env.scene.cfg.use_tight_ceil:
                    h = env.scene.cur_props.hulls[indices]
                    h = h[None].expand(num_samples,
                                       *h.shape)
                    c = env.scene.cur_props.env_code[indices]
                    c = c[None].expand(num_samples,
                                       *c.shape)

                def sample_fn():
                    qq = {}
                    if (env.scene.cur_props.yaw_only is not None and
                            env.scene.cur_props.yaw_only[indices].sum() > 0):
                        q1 = sample_q(
                            indices, num_reset * num_samples, aux=qq)
                        q2 = sample_q_planar(
                            indices, num_reset * num_samples)
                        is_yaw_only = env.scene.cur_props.yaw_only[indices]
                        pi = env.scene.cur_props.pose_index[indices]
                        yaw_only = einops.repeat(is_yaw_only,
                                                 'B ... -> (B n) ...',
                                                 n=num_samples)
                        q = th.where(yaw_only[..., None], q2, q1)
                        which_pose = qq['pose_index']
                        which_pose[is_yaw_only] = pi[is_yaw_only].expand(
                            -1, num_samples)
                    else:
                        q = sample_q(
                            indices, num_reset * num_samples, aux=qq)
                        which_pose = qq['pose_index']

                    foot_radius = th.take_along_dim(
                        env.scene.data['obj_foot_radius'][indices],
                        which_pose,
                        -1)
                    xyz = env.scene._sample_pos(obj_radius=foot_radius,
                                                num_samples=1).squeeze(-2)
                    pp = th.cat([xyz,
                                 q.reshape(*xyz.shape[:2], 4)],
                                dim=-1)
                    obj_height = th.take_along_dim(
                        env.scene.cur_props.object_height[indices], which_pose, -1)
                    return th.cat([pp, obj_height[..., None]],
                                  dim=-1).swapaxes(0, 1)

                def accept(pose):
                    pos_err, orn_err = pose_error(
                        pose, obj_pi[..., : 3],
                        pose[..., 3: 7],
                        obj_pi[..., 3:7])
                    # at least far from init or rotated right amount
                    non_overlap = th.logical_or(
                        pos_err > self.goal_radius_samples[indices],
                        orn_err > self.goal_angle_samples[indices],
                    )
                    if env.scene.cfg.use_tight_ceil:
                        z = z_from_q_and_hull(
                            pose[..., 3:7],
                            h,
                            pose[..., 2]
                        )
                        u = pose[..., 2] + z
                        # no ceil or object is below ceil when it is at the
                        # goal
                        is_below_ceil = (c[..., -1] == 0) | ((c[..., -1] > 0)
                                                             & (u < c[..., -1]))
                        non_overlap &= is_below_ceil

                    return non_overlap
                p = rejection_sample(sample_fn,
                                     accept,
                                     batched=True,
                                     sample=False)
                xyz = p[..., :3]
                q = p[..., 3:7]
            else:
                xyz = env.scene._sample_pos(
                    obj_radius=foot_radius,
                    num_samples=1).squeeze(dim=-2)

            xyz[..., 2] = z_from_q_and_hull(
                q,
                env.scene.cur_props.hulls[indices],
                xyz[..., 2]
            )

            self.goal[indices, 0:3] = xyz
            self.goal[indices, 3:7] = q
        return (xyz, q)

    @nvtx.annotate('PushTask.reset()', color="blue")
    def reset(self, env: 'EnvBase', indices: Iterable[int]):
        cfg = self.cfg

        # Figure out which envs need resets.
        if indices is None:
            indices = th.arange(env.cfg.num_env,
                                dtype=th.long,
                                device=env.cfg.th_device)
            num_reset: int = env.cfg.num_env
        else:
            indices = th.as_tensor(indices,
                                   dtype=th.long,
                                   device=env.cfg.th_device)
            num_reset: int = len(indices)

        if len(indices) <= 0:
            return

        table_pos = env.scene.data['table_pos'][indices]
        table_dims = env.scene.data['table_dim'][indices]
        obj_ids = env.scene.data['obj_id'][indices]

        if True:
            # Revise workspace dimensions.
            # table_pos = env.scene.table_pos[indices]
            # table_dims = env.scene.table_dims[indices]

            # Update workspace bounds.
            ws_lo, ws_hi = compute_workspace(
                table_pos,
                table_dims,
                cfg.ws_height,
                cfg.ws_margin
            )
            self.ws_lo[indices] = ws_lo
            self.ws_hi[indices] = ws_hi

            # Compute goal bounds.
            # goal_lo, goal_hi = compute_workspace(
            #     table_pos,
            #     table_dims * cfg.margin_scale,
            #     0.0, 0.0)
            # self.goal_lo[indices] = goal_lo
            # self.goal_hi[indices] = goal_hi

        # Update (sampled) goal thresholds.
        # r0, r1 = cfg.sample_thresh
        # self.goal_radius_samples[indices] = self.goal_radius * randu(
        #     r0, r1, size=(num_reset,), device=self.goal_radius_samples.device)
        # self.goal_angle_samples[indices] = self.goal_angle * randu(
        #     r0, r1, size=(num_reset,), device=self.goal_angle_samples.device)
        # self.max_speed_samples[indices] = self.max_speed * randu(
        #     r0, r1, size=(num_reset,), device=self.max_speed_samples.device)

        # == Sample Z ==
        try:
            self.goal[indices] = env.scene.data['goal_pose'][indices]
            goal_xyz = self.goal[indices][..., :3]
            # if cfg.import_task:
            #    goal_xyz, goal_q = self.load_goal(env, indices)
            # else:
            #    goal_xyz, goal_q = self.sample_goal(env, indices)
            #    if cfg.export_task:
            #        if 'xyz' not in self.__episode:
            #            self.__episode['xyz'] = goal_xyz
            #            self.__episode['q'] = goal_q
            self.calc_task_spec(env, indices, goal_xyz)
        finally:
            # __always__ update __prev
            self.__prev_obj_pose[indices] = env.tensors['root'][
                obj_ids.long(), :7]
            self.__has_prev[indices] = False


def main():
    cfg = Feedback.Config()
    feedback = Feedback(cfg)
    print(feedback.rewd.dim)

    kwds = dict(device='cpu')
    B: int = 1024

    data = dict(
        step=th.randint(0, 128, (B,), **kwds),
        prev_obj_state=th.randn((B, 7), **kwds),
        curr_obj_state=th.randn((B, 7), **kwds),
        goal=th.randn((B, 7), **kwds),
        goal_radius=th.randn((B,), **kwds),
        goal_angle=th.randn((B,), **kwds),
        bound_lo=th.randn((B, 3), **kwds),
        bound_hi=th.randn((B, 3), **kwds),
        kpts=th.randn((B, 16, 3), **kwds)
    )
    rewd, done, succ, timeout, info = feedback(**data)
    print(rewd.shape)
    print(done.shape)
    print(succ.shape)
    print(timeout.shape)
    # compute_feedback_legacy(


if __name__ == '__main__':
    main()
