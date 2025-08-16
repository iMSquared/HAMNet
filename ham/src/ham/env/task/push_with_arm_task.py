#!/usr/bin/env python3

import isaacgym

from typing import Tuple, Iterable, Optional, Union, Dict
from dataclasses import dataclass, field
from ham.util.config import ConfigBase
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from cho_util.math import transform as tx
import copy


from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import einops

from ham.env.common import quat_rotate
from ham.util.math_util import apply_pose_tq

from ham.env.scene.base import SceneBase
from ham.env.scene.tabletop_scene import TableTopScene
from ham.env.robot.base import RobotBase
from ham.env.robot.franka import Franka

# from ham.env.task.null_task import NullTask
from ham.env.task.push_task import (
    PushTask,
    ReachReward,
    CatReward,
    AddReward,
)
from ham.env.task.reward_util import (
    keypoint_error,
    geodesic_step_keypoint_error,
    PointDistance,
    penetration_depth
)
from ham.env.task.rewards import (
    TaskReward,
    EnergyRegularizingReward,
    SoftConstraintReward,
    HardConstraintReward
)
from ham.env.env.base import (EnvBase)
from ham.env.common import (
    get_default_sim_params,
    create_camera
)

from ham.env.util import get_mass_properties
from ham.util.vis.flow import flow_image
from ham.util.torch_util import dcn
from ham.util.config import recursive_replace_map
from ham.util.torch_util import dot

from icecream import ic
import nvtx


def copy_obs(obs):
    if isinstance(obs, th.Tensor):
        return obs.detach().clone()
    elif isinstance(obs, dict):
        return {k: copy_obs(v) for k, v in obs.items()}
    else:
        raise ValueError('no idea.')


class PushWithArmTask(PushTask):
    @dataclass
    class Config(PushTask.Config):
        hand_obj_pot_coef: float = 1.0  # deprecated
        # NOTE(ycho): unused, only here for compatibility
        oob_fail_coef: Optional[float] = None  # deprecated
        max_pot_rew: float = 0.5  # deprecated
        # Potential reward magnitude, relative to
        # the object<->goal potential.
        rel_pot_rew_scale: float = 0.5  # deprecated
        regularize_coef: float = 0.  # deprecated
        crm_override: bool = False  # deprecated
        nearest_induce: bool = False  # deprecated
        opposite: bool = False  # deprecated

        soft_constraint: bool = False
        contact_force_penalty: Optional[Tuple[str, ...]] = None
        col_penalty_coef: float = 0.

        dist: PointDistance.Config = PointDistance.Config()
        task_reward: TaskReward.Config = TaskReward.Config(dim=2)
        contact_reward: ReachReward.Config = ReachReward.Config()
        energy_reward: EnergyRegularizingReward.Config = EnergyRegularizingReward.Config()
        sc_reward: SoftConstraintReward.Config = SoftConstraintReward.Config()

        rewd_reduce_type: str = 'cat'
        rewd_coef: Dict[str, float] = field(default_factory=lambda: dict(
            succ=1.0,
            reach=0.15,
            contact=0.15 * 0.2,
            col_penalty=1.0,
            energy=0.0,
        ))
        add: AddReward.Config = AddReward.Config(
            coef={
                # NOTE(ycho): `task` reward (succ/reach) is internally
                # normalized with the proper coefficients
                'task': 1.0,
                'contact': 0.15 * 0.2,
                # "regularize_coef"
                # currently set to 0.0
                'energy': 0.0,
                # NOTE(ycho):
                # `col_penalty` coef will be
                # dynamically adjusted ,
                # so we pass in `1.0` instead
                'col_penalty': 1.0
            })
        cat: CatReward.Config = CatReward.Config(keys=[
            'task',
            'contact',
            'col_penalty',
            'energy',
        ])

    def __init__(self, cfg: Config, writer=None):
        super().__init__(cfg, writer=writer)
        self.cfg = cfg
        if cfg.crm_override:
            raise ValueError('crm_override has been deprecated')
        self._has_prev: th.Tensor = None
        self._prev_state: th.Tensor = None
        self.is_col: th.Tensor = None
        self.col_depth: th.Tensor = None
        self.col_penalty_coef = cfg.col_penalty_coef

        self.dist = PointDistance(cfg.dist)
        col_rew = (SoftConstraintReward(cfg.sc_reward)
                   if (cfg.soft_constraint or cfg.contact_force_penalty)
                   else HardConstraintReward(HardConstraintReward.Config(
                       split=cfg.sc_reward.split)))
        if cfg.rewd_reduce_type == 'cat':
            self.__reward = CatReward(cfg.cat, {
                'task': TaskReward(cfg.task_reward),
                'contact': ReachReward(cfg.contact_reward),
                'col_penalty': col_rew,
                'energy': EnergyRegularizingReward(cfg.energy_reward)
            })
            self.rewd_coef = cfg.rewd_coef
        elif cfg.rewd_reduce_type == 'add':
            self.__reward = AddReward(cfg.add, {
                'task': TaskReward(cfg.task_reward),
                'contact': ReachReward(cfg.contact_reward),
                'col_penalty': col_rew,
                'energy': EnergyRegularizingReward(cfg.energy_reward)
            })
        else:
            raise ValueError(
                F'Unknown rewd_reduce_type={cfg.rewd_reduce_type}')

    def setup(self, env: 'EnvBase'):
        super().setup(env)

        cfg = self.cfg
        device: th.device = th.device(env.device)
        self.device = device

        # Previous values cache
        self._has_prev: th.Tensor = th.zeros(
            (env.num_env,), dtype=th.bool,
            device=device)
        self._prev_state: th.Tensor = None
        self.regularize = env.robot.cfg.regularize

        if cfg.soft_constraint:
            link_keys = list(self.env.robot.link_ids.keys())
            num_body = len(link_keys)
            num_points = sum([self.env.robot.cloud[k].shape[-2]
                              for k in link_keys])
            self.is_col = th.zeros(env.num_env,
                                   num_points,
                                   dtype=th.bool,
                                   device=env.device)
            self.depth = th.zeros(env.num_env,
                                  num_points,
                                  dtype=th.float,
                                  device=env.device)
        if cfg.contact_force_penalty:
            self.link_ids = {}
            for k in cfg.contact_force_penalty:
                link_indices = [env.gym.find_actor_rigid_body_index(
                    env.envs[i],
                    env.robot.handles[i],
                    k,
                    gymapi.DOMAIN_SIM
                ) for i in range(env.num_env)]
                self.link_ids[k] = th.as_tensor(
                    link_indices,
                    dtype=th.long,
                    device=self.device
                )

    def reset(self, env: 'EnvBase', indices: Iterable[int]):
        if indices is None:
            indices = th.arange(env.cfg.num_env,
                                dtype=th.int32,
                                device=env.cfg.th_device)
        else:
            indices = th.as_tensor(indices,
                                   dtype=th.long,
                                   device=env.cfg.th_device)
            self._has_prev[indices] = 0
        return super().reset(env, indices)

    def update_reward(self,
                      env: 'EnvBase',
                      obs: th.Tensor,
                      action: th.Tensor,
                      task_reward: th.Tensor,
                      info: Dict[str, th.Tensor]
                      ):
        cfg = self.cfg

        obj_ids = env.scene.data['obj_id'].long()
        obj_state = env.tensors['root'][obj_ids, :]
        tip_indices = env.robot.tip_body_indices.long()
        tip_state = env.tensors['body'][tip_indices]

        # Compute robot<->env
        # penetration depth, for SC.
        # Note that we do not compute this internally,
        # to avoid computing it twice (ex: when using pot-based reward)
        if cfg.soft_constraint:
            has_ceil = env.scene.cur_props.env_code[..., 18:19]
            box_faces = env.scene.cur_props.env_faces
            body_tensors = self.env.tensors['body']
            cur_hand_depth, cur_else_depth = (
                penetration_depth(has_ceil,
                                  box_faces,
                                  body_tensors,
                                  env.robot.cloud,
                                  env.robot.link_ids,
                                  reduce=False))
            cur_hand_count, cur_else_count = (
                (cur_hand_depth < 0).float().sum(-1),
                (cur_else_depth < 0).float().sum(-1)
            )
            cur_hand_depth = cur_hand_depth.sum(-1)
            cur_else_depth = cur_else_depth.sum(-1)

        if self.regularize == 'action':
            energy = env.robot.energy
        elif self.regularize in ('energy', 'torque'):
            energy = env.robot.energy / env.cfg.action_period
        else:
            energy = th.zeros(env.num_env,
                              dtype=th.float32,
                              device=env.device)

        if self._prev_state is not None:
            # FIXME(ycho): very ugly code
            prv = self._prev_state
            prv_tip = th.where(self._has_prev[..., None],
                               prv['tip_state'],
                               tip_state)
            prv_obj = th.where(self._has_prev[..., None],
                               prv['obj_state'],
                               obj_state)

            if cfg.soft_constraint:
                prv_hand_depth = th.where(self._has_prev,
                                          prv['hand_depth'],
                                          cur_hand_depth)
                prv_else_depth = th.where(self._has_prev,
                                          prv['else_depth'],
                                          cur_else_depth)
                prv_hand_count = th.where(self._has_prev,
                                          prv['hand_count'],
                                          cur_hand_count)
                prv_else_count = th.where(self._has_prev,
                                          prv['else_count'],
                                          cur_else_count)

            src_point = th.stack([prv_tip, tip_state],
                                 dim=0)
            dst_point = th.stack([prv_obj, obj_state],
                                 dim=0)
            if cfg.soft_constraint:
                hand_depth = th.stack([prv_hand_depth, cur_hand_depth],
                                      dim=0)
                else_depth = th.stack([prv_else_depth, cur_else_depth],
                                      dim=0)
                hand_count = th.stack([prv_hand_count, cur_hand_count],
                                      dim=0)
                else_count = th.stack([prv_else_count, cur_else_count],
                                      dim=0)
        else:
            src_point = einops.repeat(tip_state,
                                      '... -> t ...', t=2)
            dst_point = einops.repeat(obj_state,
                                      '... -> t ...', t=2)
            if cfg.soft_constraint:
                hand_depth = einops.repeat(cur_hand_depth,
                                           '... -> t ...', t=2)
                else_depth = einops.repeat(cur_else_depth,
                                           '... -> t ...', t=2)
                hand_count = einops.repeat(cur_hand_count,
                                           '... -> t ...', t=2)
                else_count = einops.repeat(cur_else_count,
                                           '... -> t ...', t=2)

        if cfg.soft_constraint:
            hand_depth = hand_depth[..., None]
            else_depth = else_depth[..., None]
            hand_count = hand_count[..., None]
            else_count = else_count[..., None]
        else:
            hand_depth = None
            else_depth = None
            hand_count = None
            else_count = None

        if cfg.contact_force_penalty:
            forces = []
            net_force = env.tensors['net_contact']
            for k in cfg.contact_force_penalty:
                forces.append(net_force[self.link_ids[k]])
            finger_force = (th.norm(th.cat(forces, -1),
                            dim=-1, keepdim=True)
                            / np.sqrt(len(cfg.contact_force_penalty)))
            # FIXME (JH) We are currently using sc as a placeholder
            # Sc reauires previous info even if it is not potential reward
            # thus repeat the current value
            hand_depth = einops.repeat(finger_force,
                                       '... -> t ...', t=2)
            else_depth = th.zeros_like(hand_depth)
            hand_count = th.zeros_like(hand_depth)
            else_count = th.zeros_like(hand_depth)

        reward, reward_info = self.__reward(
            # needed for `task`
            task=task_reward,
            # needed for hand-object ("contact") reward
            dist=self.dist(src_point=src_point[..., :3],
                           dst_point=dst_point[..., :3]),
            # needed for SC
            # NOTE(ycho): to _not_ use [...,None] here,
            # another option is to toggle `reduce=False`
            # in the underlying PotLogRew() config.
            add_rew_coef_col_penalty=self.col_penalty_coef,
            hand_depth=hand_depth,
            else_depth=else_depth,
            hand_count=hand_count,
            else_count=else_count,
            # needed for `energy`
            energy=energy
        )
        # Do not include the internal virtual reward ("task")
        # in the logs.
        reward_info.pop('task', None)
        if 'reward' not in info:
            info['reward'] = {}
        info['reward'].update(reward_info)

        state = {}
        state['obj_state'] = obj_state
        state['tip_state'] = tip_state
        if cfg.soft_constraint:
            state['hand_depth'] = cur_hand_depth
            state['else_depth'] = cur_else_depth
            state['hand_count'] = cur_hand_count
            state['else_count'] = cur_else_count
        return reward, info, state

    @property
    def num_rew(self):
        return self.__reward.dim

    def compute_feedback(self,
                         env: 'EnvBase',
                         obs: th.Tensor,
                         action: th.Tensor):
        cfg = self.cfg

        # Compute the basics from PushTask.
        reward, done, info = super().compute_feedback(env, obs, action)
        reward, info, state = self.update_reward(
            env, obs, action, reward, info)

        # FIXME(ycho): HACK!!!
        if cfg.rewd_reduce_type == 'cat':
            # NOTE(ycho): remember not to apply
            # the coefficients _twice_( i.e.
            # once in push_task and once in push_with_arm_task)
            # especially to reach_reward (0.15x0.15)
            # FIXME(ycho): hardcoded known order of rewards...!
            REWD_WEIGHT: th.Tensor = th.as_tensor(
                [self.rewd_coef['succ'],
                 # NOTE(ycho): since we don't apply `0.15`
                 # coef in `push_task.py` when rewd_reduce_type=='cat',
                 # we apply the coefficients here.
                 self.rewd_coef['reach'],
                 self.rewd_coef['contact'],
                 self.col_penalty_coef,
                 self.col_penalty_coef,
                 self.rewd_coef['energy']],
                dtype=th.float, device=reward.device)
            reward = reward * REWD_WEIGHT

            # FIXME(ycho): apply coefficients to logged rewards...
            info['reward']['succ'] *= self.rewd_coef['succ']
            info['reward']['reach'] *= self.rewd_coef['reach']
            info['reward']['contact'] *= self.rewd_coef['contact']
            info['reward']['col_penalty'] *= self.col_penalty_coef
            info['reward']['energy'] *= self.rewd_coef['energy']
        elif cfg.rewd_reduce_type == 'add':
            # NOTE(ycho): we flip the sign of the
            # logged reward, for the purpose of
            # consistent logging...
            # FIXME(ycho): probably best not to do this
            # in the long run!
            info['reward']['col_penalty'] *= -1

        # Update `prev` values for computing potentials.
        self._has_prev.copy_(~done)
        self._prev_state = copy_obs(state)

        return (reward, done, info)


def main():
    cfg = PushWithArmTask.Config()
    reward = CatReward(cfg.cat, {
        'task': TaskReward(cfg.task_reward),
        'contact': ReachReward(cfg.contact_reward),
        # 'col_penalty': SoftConstraintReward(cfg.sc_reward),
        'col_penalty': HardConstraintReward(cfg.sc_reward),
        'energy': EnergyRegularizingReward(cfg.energy_reward)
    })
    print(reward, reward.dim)

    B: int = 13
    P: int = 511
    S: int = 16

    kwds = dict(device='cpu')
    data = dict(
        task=th.randn((B, 2), **kwds),
        succ=th.randn((B, ), **kwds),
        dist=th.randn((2, B, 1), **kwds).abs(),
        hand_depth=th.randn((2, B, 1), **kwds),
        else_depth=th.randn((2, B, 1), **kwds),
        # robot_pcd=th.randn((B, P, 3), **kwds),
        # has_ceil=(th.randn((B, 1), **kwds) > 0.5),
        # 6 faces X (3+3) for normal/origin
        # box_faces=th.randn((B, S, 6, 6), **kwds),
        energy=th.randn((B,), **kwds),
        # col_penalty_coef=1.0,
        hand_range=(511 - 88, 511)
    )
    print(reward(**data)[0].shape)
    print(reward(**data)[0][0])


if __name__ == '__main__':
    main()
