#!/usr/bin/env python3

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import torch_utils

from typing import Tuple, Iterable, List, Optional, Dict, Union
import math
import nvtx
import pkg_resources
from dataclasses import dataclass, replace
from ham.util.config import ConfigBase
import numpy as np
import torch as th
import torch.nn.functional as F
import einops
from gym import spaces
from icecream import ic
import pickle

from ham.env.env.base import EnvBase
from ham.env.robot.base import RobotBase
from ham.env.robot.franka_util import (
    CartesianControlError,
    JointControlError,
    IKController,
    find_actor_indices,
    find_actor_handles,
    CartesianImpedanceController)
from ham.env.common import apply_domain_randomization
from ham.util.math_util import matrix_from_quaternion
from ham.util.torch_util import dcn, tile_data


def get_action_space(cfg: 'Franka.Config',
                     vel_lo,
                     vel_hi,
                     dof_limits,
                     use_fr3: bool = False):
    if cfg.target_type == 'rel':
        if cfg.ctrl_mode == 'jpos':
            if cfg.gain == 'variable':
                # NOTE(ycho): since target_type==`rel`,
                # we use the _relative_ joint residuals
                # which require us to multiply the
                # velocity by the timestep size(=0.1s)
                # which is unfortunately hardcoded here...
                q_lo = np.multiply(vel_lo, 0.1)
                q_hi = np.multiply(vel_hi, 0.1)
                if use_fr3:
                    # FIXME (JH): hardcord for now
                    q_lo[-3:] = -0.26
                    q_hi[-3:] = 0.26

                # FIXME(ycho): hardcoded min bound for
                # KP_pos!!
                min_bound = np.concatenate([q_lo,
                                            [cfg.KP_min] * 7,
                                            [cfg.KD_min] * 7]
                                           )
                max_bound = np.concatenate([q_hi,
                                            [cfg.KP_pos] * 7,
                                            [cfg.KD_max] * 7])
                action_space = spaces.Box(
                    np.asarray(min_bound),
                    np.asarray(max_bound),
                )
            else:
                action_space = spaces.Box(*dof_limits)
        elif cfg.ctrl_mode == 'jvel':
            action_space = spaces.Box(
                np.asarray(vel_lo), np.asarray(vel_hi))
        elif cfg.ctrl_mode in ('cpos_n', 'cpos_a', 'CI', 'osc'):
            if cfg.rot_type == 'axis_angle':
                min_bound = [-cfg.max_pos, -cfg.max_pos, -cfg.max_pos,
                             -cfg.max_ori, -cfg.max_ori, -cfg.max_ori]
                max_bound = [cfg.max_pos, cfg.max_pos, cfg.max_pos,
                             cfg.max_ori, cfg.max_ori, cfg.max_ori]
                if cfg.lock_orn:
                    min_bound = min_bound[:3]
                    max_bound = max_bound[:3]
                if cfg.gain == 'variable':
                    if cfg.ctrl_mode in ('CI', 'osc'):
                        min_bound += [10.] * 6
                        min_bound += [0.] * 6
                        max_bound += [cfg.KP_pos] * 3
                        max_bound += [cfg.KP_ori] * 3
                        max_bound += [2.] * 6
                    else:  # cpos_*
                        min_bound += [10.] * 7
                        min_bound += [0.1] * 7
                        max_bound += [cfg.KP_pos] * 7
                        max_bound += [2.] * 7
                action_space = spaces.Box(
                    np.asarray(min_bound),
                    np.asarray(max_bound),
                )
            else:
                min_bound = [-0.2, -0.2, -0.2, -1, -1, -1, -1]
                max_bound = [+0.2, +0.2, +0.2, +1, +1, +1, +1]
                if cfg.lock_orn:
                    min_bound = min_bound[:3]
                    max_bound = max_bound[:3]
                action_space = spaces.Box(
                    np.asarray(min_bound),
                    np.asarray(max_bound),
                )
        elif cfg.ctrl_mode == 'jpos+cpos_n':
            # jpos+cpos_n
            j_cfg = replace(cfg, ctrl_mode='jpos')
            j_space = get_action_space(j_cfg, vel_lo, vel_hi, dof_limits,
                                       cfg.use_fr3)
            c_cfg = replace(cfg, ctrl_mode='cpos_n')
            c_space = get_action_space(c_cfg, vel_lo, vel_hi, dof_limits)
            # FIXME(ycho): for now we share the gains
            return spaces.Box(
                np.r_[j_space.low[:7], c_space.low],
                np.r_[j_space.high[:7], c_space.high]
            )
    else:
        if cfg.ctrl_mode == 'jpos':
            if cfg.gain == 'variable':
                # NOTE(ycho): since target_type==`rel`,
                # we use the _relative_ joint residuals
                # which require us to multiply the
                # velocity by the timestep size(=0.1s)
                # which is unfortunately hardcoded here...
                q_lo = dof_limits[0]
                q_hi = dof_limits[1]
                # FIXME(ycho): hardcoded min bound for
                # KP_pos!!
                min_bound = np.concatenate([q_lo,
                                            [cfg.KP_min] * 7,
                                            [cfg.KD_min] * 7]
                                           )
                max_bound = np.concatenate([q_hi,
                                            [cfg.KP_pos] * 7,
                                            [cfg.KD_max] * 7])
                action_space = spaces.Box(
                    np.asarray(min_bound),
                    np.asarray(max_bound),
                )
            else:
                action_space = spaces.Box(*dof_limits)
        else:
            raise ValueError(F'invalid target type = {cfg.target_type}')
    return action_space


class Franka(RobotBase):
    @dataclass
    class Config(ConfigBase):
        # cube_dims: Tuple[float, float, float] = (0.08, 0.08, 0.08)
        # cube_dims: Tuple[float, float, float] = (0.045, 0.045, 0.045)
        # apply_mask: bool = False

        # What are reasonably 'random' joint initializations?
        # I can think of four:
        # Option#0 - home position
        # Option#1 - uniformly sample joint cfgs
        # Option#2 - discretely sample from "valid" cfgs
        # Option#3 - kinematics-based euclidean-ish sampling
        asset_root: str = pkg_resources.resource_filename('ham.data', 'assets')
        robot_file: str = 'franka_description/robots/franka_panda.urdf'
        open_hand_robot_file: str = 'franka_description/robots/franka_panda_open_hand.urdf'
        half_open_hand_robot_file: str = 'franka_description/robots/franka_panda_half_open_hand.urdf'
        custom_hand_robot_file: str = 'franka_description/robots/franka_panda_custom_v3.urdf'
        use_rg6: bool = False
        rg6_robot_file: str = 'franka_description/robots/franka_panda_rg6.urdf'
        use_fr3: bool = True
        fr3_file: str = 'franka_description/robots/fr3_custom.urdf'

        # randomize_init_joints: bool = False
        # if `sample`, use one of the pre-sampled configurations
        init_type: str = 'home'

        # 1. joint position control
        # 2. joint velocity control
        # 3. cartesian position control; numerical IK
        # 4. cartesian position control; analytic IK (unsupported)
        # 5. cartesian impedance;
        # (jpos, jvel, cpos_n, cpos_a, CI)
        ee_frame: str = 'tool'
        ctrl_mode: str = 'CI'

        gain: str = 'variable'  # or fixed
        # or action magnitude or torque or None
        regularize: Optional[str] = 'energy'

        # (use isaacgym built in pos or vel controller or effort controller)
        use_effort: bool = True
        target_type: str = 'rel'  # or 'abs'
        # Numerical IK damping factor.
        damping: float = 0.05
        rot_type: str = 'axis_angle'
        KP_pos: float = 10.0  # 1.0
        KP_ori: float = 100.0  # 1.0#0.3
        KD_pos: float = math.sqrt(KP_pos)  # * 2
        KD_ori: float = math.sqrt(KP_ori)

        # JPOS GAIN BOUNDS
        KP_min: float = 10.0
        KD_min: float = 1.0
        KD_max: float = 2.0

        # KD_pos: float = 0.0
        # KD_ori: float = 0.0
        VISCOUS_FRICTION: float = 0.0
        keepout_radius: float = 0.3

        max_pos: float = 0.1  # 0.1m / timestep(=0.04s)
        max_ori: float = 0.5  # 0.5rad (~30deg) / timestep(=0.04s)

        lin_vel_damping: float = 1.0
        ang_vel_damping: float = 5.0
        max_lin_vel: float = 2.0
        max_ang_vel: float = 6.28

        accumulate: bool = True
        lock_orn: bool = False

        ws_bound: Optional[List[List[float]]] = (
            [-0.3, -0.4, 0.4],  # min
            [+0.3, +0.4, 0.8]  # max
        )

        track_object: bool = False
        obj_margin: float = max_pos
        use_open_hand: bool = False
        use_half_open_hand: bool = False
        use_custom_hand: bool = False

        crm_override: bool = False
        hack_open_hand: bool = False
        base_height: str = 'ground'  # or 'table'
        clip_bound: bool = False

        init_ik_prob: float = 0.5
        disable_table_collision: bool = False

        add_tip_sensor: bool = False
        add_control_noise: bool = False
        control_noise_mag: float = 0.03

        # HACK for heigh offset for handling some custom env with non zero
        # center
        height_offset: float = 0.0

        box_min: Tuple[float, ...] = (-0.3, -0.4636, -0.2,
                                      -2.7432, -0.3335, 1.5269, -np.pi / 2)  # 0.3816)
        box_max: Tuple[float, ...] = (
            0.3, 0.5432, 0.2, -1.5237, 0.3335, 2.5744, np.pi / 2)  # 1.3914)

        # NOTE(ycho): friction parameters here are
        # applied to left-right fingers of panda FE gripper.
        default_hand_friction: float = 1.5
        default_body_friction: float = 0.1
        restitution: float = 0.5
        randomize_hand_friction: bool = True
        min_hand_friction: float = 1.0
        max_hand_friction: float = 1.2
        max_control_delay: Optional[int] = None
        # clamp joint target relative to the max
        # e.g. [-0.9, 0.9]
        rel_clamp_joint_target: Optional[Tuple[float, float]] = None

        load_pcd: bool = False
        cloud_file: str = '/input/robot/cloud.pkl'
        half_open_cloud_file: str = '/input/robot/half_open_cloud.pkl'
        custom_cloud_file: str = '/input/robot/custom_v3_cloud.pkl'

        # cabinet_type_ik_file: str = '/input/robot/cabinet.pkl'
        # wall_type_ik_file: str = '/input/robot/wall.pkl'

        import_robot: Optional[str] = None
        export_robot: Optional[str] = None

        measure_joint_torque: bool = False
        ema_action: Optional[float] = None
        disable_all_collsion: bool = False
        joint_param_file: Optional[str] = None

    def __init__(self, cfg: Config):
        self.cfg = cfg
        ic(cfg)
        self.n_bodies: int = None
        self.n_dofs: int = None
        self.dof_limits: Tuple[np.ndarray, np.ndarray] = None
        self.assets = {}
        self.valid_qs: Optional[th.Tensor] = None
        self.q_lo: th.Tensor = None
        self.q_hi: th.Tensor = None
        self._first = True
        self.q_home: th.Tensor = None

        # NOTE(ycho): decided to ALWAYS
        # assume that `robot_radius` references
        # `panda_hand`.
        # if cfg.ee_frame == 'tool':
        #     # round up from 0.1562...
        #     self.robot_radius: float = 0.16
        # elif cfg.ee_frame == 'hand':
        #     # round up from 0.1162...
        #     self.robot_radius: float = 0.12
        # else:
        #     raise ValueError(F'Unknown ee_frame = panda_{cfg.ee_frame}')
        self.robot_radius: float = 0.12

        self.pose_error: Union[CartesianControlError, JointControlError] = None
        if (not cfg.clip_bound) and cfg.track_object:
            raise ValueError(
                'clip_bound should be true to enable track_object')
        self.delay_counter = None  # counter for delay
        self.__episode = {}

        if cfg.import_robot:
            with open(cfg.import_robot, 'rb') as fp:
                self.__episode = pickle.load(fp)

    def export(self):
        cfg = self.cfg
        if cfg.export_robot is None:
            return
        with open(cfg.export_robot, 'wb') as fp:
            pickle.dump(self.__episode, fp)

    def setup(self, env: 'EnvBase'):
        # FIXME(ycho): introspection!
        cfg = self.cfg

        self.num_env = env.cfg.num_env
        self.device = th.device(env.cfg.th_device)

        accumulate = (
            self.cfg.accumulate and
            self.cfg.target_type == 'rel'
        )

        # Workspace boundary.
        # Ignored in case `track_object` is true.
        self.ws_bound = None
        # cfg.ws_bound = None
        if cfg.ws_bound is not None:
            self.ws_bound = th.as_tensor(cfg.ws_bound,
                                         dtype=th.float,
                                         device=self.device)
        if cfg.ctrl_mode in ['osc', 'CI']:
           self.pose_error = CartesianControlError(
               self.cfg.accumulate,
               target_shape=(self.num_env, 7),
               dtype=th.float,
               device=self.device,
               pos_bound=self.ws_bound if cfg.clip_bound else None)
        else:
            # FIXME(ycho): pose_error -> joint_error
            self.pose_error = JointControlError(
                self.cfg.accumulate,
                target_shape=(self.num_env, 7),
                dtype=th.float,
                device=self.device,
                pos_bound=self.ws_bound if cfg.clip_bound else None)

        self.eff_limits = th.as_tensor(self.eff_limits,
                                       dtype=th.float,
                                       device=self.device)

        if cfg.ctrl_mode in ['osc', 'CI']:
            self.controller = CartesianImpedanceController(
                cfg.KP_pos,
                cfg.KD_pos,
                cfg.KP_ori,
                cfg.KD_ori,
                self.eff_limits,
                device=self.device,
                OSC=(cfg.ctrl_mode == 'osc')
            )
        else:
            self.controller = IKController(
                cfg.KP_pos,
                self.eff_limits,
                num_env=env.num_env,
                device=self.device
            )

        if cfg.regularize is not None:
            self.energy = th.zeros(self.num_env, dtype=th.float,
                                   device=self.device)

        self.indices = th.as_tensor(
            find_actor_indices(env.gym, env.envs, 'robot'),
            dtype=th.int32, device=self.device)
        self.handles = find_actor_handles(env.gym, env.envs, 'robot')

        self.hand_ids = [
            env.gym.find_actor_rigid_body_index(
                env.envs[i],
                self.handles[i],
                f'panda_{cfg.ee_frame}',
                gymapi.IndexDomain.DOMAIN_SIM
            ) for i in range(self.num_env)
        ]
        self.hand_ids = th.as_tensor(
            self.hand_ids,
            dtype=th.long,
            device=self.device
        )

        self._control = th.zeros((self.num_env, self.n_dofs),
                                 dtype=th.float, device=self.device)
        self._target = th.zeros((self.num_env, 7),
                                dtype=th.float, device=self.device)
        self.ee_wrench = th.zeros((self.num_env, 6),
                                  dtype=th.float, device=self.device)
        self.q_lo = th.as_tensor(self.dof_limits[0],
                                 dtype=th.float, device=self.device)
        self.q_hi = th.as_tensor(self.dof_limits[1],
                                 dtype=th.float, device=self.device)
        if cfg.rel_clamp_joint_target is not None:
            dof_center = th.as_tensor(
                0.5 * (self.dof_limits[0] + self.dof_limits[1]),
                dtype=th.float,
                device=self.device)
            dof_scale = th.as_tensor(
                (self.dof_limits[1] - self.dof_limits[0]),
                dtype=th.float,
                device=env.device)
            self.__reduced_q_lo = (dof_center
                                   + cfg.rel_clamp_joint_target[0]
                                   * dof_scale / 2)
            self.__reduced_q_hi = (dof_center
                                   + cfg.rel_clamp_joint_target[1]
                                   * dof_scale / 2)
            ic(dof_center, dof_scale, self.__reduced_q_lo, self.__reduced_q_hi)
        if self.cfg.init_type == 'sample':
            valid_qs = np.load('/tmp/qs.npy')
            self.valid_qs = th.as_tensor(valid_qs,
                                         dtype=th.float32, device=self.device)
            self.valid_qs = (self.valid_qs + np.pi) % (2 * np.pi) - np.pi
        elif self.cfg.init_type == 'home':
            if cfg.base_height == 'ground':
                self.q_home = th.tensor(
                    [0.0, 0.0, 0.0, -0.9425, 0.0, 1.1205, 0.0],
                    device=self.device)
            elif cfg.base_height == 'table':
                self.q_home = 0.5 * (self.q_lo + self.q_hi)
                # self.q_home =th.tensor(
                #     [-0.0122, -0.1095,  0.0562, -2.5737,\
                #      -0.0196,  2.4479,  0.7756],
                #      device=self.device
                # )
            else:
                raise KeyError(F'Unknown base_height = {cfg.base_height}')
        elif self.cfg.init_type == 'ik-test':
            with open('/input/pre_contacts.pkl', 'rb') as fp:
                contacts = pickle.load(fp)
            self.ik_configs = th.as_tensor([np.random.permutation(
                contacts[k])
                for k in env.scene.keys],
                device=env.device)
            self.cursor = th.zeros(self.ik_configs.shape[0], dtype=th.long,
                                   device=env.device)
            if cfg.base_height == 'ground':
                self.q_home = th.tensor(
                    [0.0, 0.0, 0.0, -0.9425, 0.0, 1.1205, 0.0],
                    device=self.device)
            elif cfg.base_height == 'table':
                self.q_home = 0.5 * (self.q_lo + self.q_hi)
        elif self.cfg.init_type == 'ik' or self.cfg.init_type == 'pre-oracle':
            if cfg.base_height == 'ground':
                self.q_home = th.tensor(
                    [0.0, 0.0, 0.0, -0.9425, 0.0, 1.1205, 0.0],
                    device=self.device)
            elif cfg.base_height == 'table':
                self.q_home = 0.5 * (self.q_lo + self.q_hi)
        elif self.cfg.init_type == 'easy':
            # TODO(ycho): make
            # self.q_easy = th.as_tensor(
            #     [0.6634681, 0.42946462, 0.19089655, -2.15512631, -0.1472046,
            #      2.57276871, 2.53247449, 0.012, 0.012],
            #     dtype=th.float, device=self.device)
            # self.q_easy = th.as_tensor([0.3664, -0.6752, -0.2700, -1.8482, -0.1418,  1.7056,  1.0519],
            #                             dtype=th.float,
            #                             device=self.device)
            self.q_easy = th.as_tensor([-0.1765, -0.5463, -0.0490,
                                        -2.7035, -0.5, 3.6892, 2.0986],
                                       dtype=th.float,
                                       device=self.device)
        elif self.cfg.init_type == 'mvp0':
            self.q_mvp0 = th.as_tensor(
                [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035],
                dtype=th.float, device=self.device)
        elif self.cfg.init_type == 'box_sample':
            self.box_min = th.as_tensor(cfg.box_min,
                                        dtype=th.float, device=self.device)
            self.box_max = th.as_tensor(cfg.box_max,
                                        dtype=th.float, device=self.device)
        elif self.cfg.init_type == 'ik-presample':
            pass
        self._first = True

        # Acquire jacobian
        # FIXME(ycho): why here? well, it's because it's kinda
        # hard for env to know a prior what the name for the `robot`
        # should be...right? Hmmm....

        self._jacobian = gymtorch.wrap_tensor(
            env.gym.acquire_jacobian_tensor(
                env.sim, 'robot'))
        _mm = env.gym.acquire_mass_matrix_tensor(env.sim, "robot")

        EE_INDEX = self.franka_link_dict[f'panda_{cfg.ee_frame}']
        self.j_eef = self._jacobian[:, EE_INDEX - 1, :, :7]
        self.lmbda = th.eye(6, dtype=th.float,
                            device=self.device) * (self.cfg.damping**2)
        self.mm = gymtorch.wrap_tensor(_mm)
        self.mm = self.mm[:, :(EE_INDEX - 1), :(EE_INDEX - 1)]

        base_body_indices = []
        ee_body_indices = []
        tip_body_indices = []
        if env.task.cfg.nearest_induce:
            right_finger_tool_indices = []
            left_finger_tool_indices = []
            right_finger_indices = []
            left_finger_indices = []
        for i in range(self.num_env):
            base_idx = env.gym.find_actor_rigid_body_index(
                env.envs[i],
                self.handles[i],
                'panda_link0',
                gymapi.DOMAIN_SIM
            )
            base_body_indices.append(base_idx)

            ee_idx = env.gym.find_actor_rigid_body_index(
                env.envs[i],
                self.handles[i],
                # 'tool_tip',
                # 'wrist_3_link',
                'panda_hand',
                gymapi.DOMAIN_SIM
            )
            ee_body_indices.append(ee_idx)

            tip_idx = env.gym.find_actor_rigid_body_index(
                env.envs[i],
                self.handles[i],
                'panda_tool',
                gymapi.DOMAIN_SIM
            )
            tip_body_indices.append(tip_idx)
            if env.task.cfg.nearest_induce:
                left_finger_tool_idx = env.gym.find_actor_rigid_body_index(
                    env.envs[i],
                    self.handles[i],
                    'panda_left_tool',
                    gymapi.DOMAIN_SIM
                )
                left_finger_tool_indices.append(left_finger_tool_idx)
                right_finger_tool_idx = env.gym.find_actor_rigid_body_index(
                    env.envs[i],
                    self.handles[i],
                    'panda_right_tool',
                    gymapi.DOMAIN_SIM
                )
                right_finger_tool_indices.append(right_finger_tool_idx)

                left_finger_idx = env.gym.find_actor_rigid_body_index(
                    env.envs[i],
                    self.handles[i],
                    'panda_leftfinger',
                    gymapi.DOMAIN_SIM
                )
                left_finger_indices.append(left_finger_idx)
                right_finger_idx = env.gym.find_actor_rigid_body_index(
                    env.envs[i],
                    self.handles[i],
                    'panda_rightfinger',
                    gymapi.DOMAIN_SIM
                )
                right_finger_indices.append(right_finger_idx)

        self.base_body_indices = th.as_tensor(
            base_body_indices,
            dtype=th.int32,
            device=self.device)

        self.ee_body_indices = th.as_tensor(
            ee_body_indices,
            dtype=th.int32,
            device=self.device)

        self.tip_body_indices = th.as_tensor(
            tip_body_indices,
            dtype=th.int32,
            device=self.device)

        self._hack_base_offset = th.as_tensor(
            [0.4, 0, -0.4],
            dtype=th.float32,
            device=self.device)
        if env.task.cfg.nearest_induce:
            self.left_finger_tool_indices = th.as_tensor(
                left_finger_tool_indices,
                dtype=th.int32,
                device=self.device
            )
            self.right_finger_tool_indices = th.as_tensor(
                right_finger_tool_indices,
                dtype=th.int32,
                device=self.device
            )
            self.left_finger_indices = th.as_tensor(
                left_finger_indices,
                dtype=th.int32,
                device=self.device
            )
            self.right_finger_indices = th.as_tensor(
                right_finger_indices,
                dtype=th.int32,
                device=self.device
            )
        self.cur_hand_friction = th.full((self.num_env,),
                                         cfg.default_hand_friction,
                                         dtype=th.float,
                                         device=self.device)
        if cfg.max_control_delay is not None:
            self.delay_counter = th.zeros((self.num_env,),
                                          dtype=th.long,
                                          device=self.device)

        if cfg.load_pcd:
            if cfg.use_half_open_hand:
                cloud_file = cfg.half_open_cloud_file
            elif cfg.use_custom_hand:
                cloud_file = cfg.custom_cloud_file
            else:
                cloud_file = cfg.cloud_file
            with open(cloud_file, 'rb') as fp:
                cloud = pickle.load(fp)
            if True:
                self.cloud = {
                    k: (th.as_tensor(v, dtype=th.float,
                                     device=self.device)
                        .expand(self.num_env, -1, -1))
                    for k, v in cloud.items()
                }
                self.link_ids = {}
                for k in cloud.keys():
                    link_indices = [env.gym.find_actor_rigid_body_index(
                        env.envs[i],
                        self.handles[i],
                        f'panda_{k}',
                        gymapi.DOMAIN_SIM
                    ) for i in range(self.num_env)]
                    self.link_ids[k] = th.as_tensor(
                        link_indices,
                        dtype=th.long,
                        device=self.device
                    )
        if cfg.ema_action:
            n_subgoal: int = 6 if cfg.ctrl_mode == 'cpos_n' else 7
            self._ema_action = th.zeros(env.num_env,
                                        n_subgoal,
                                        dtype=th.float,
                                        device=self.device)

        if cfg.import_robot:
            # make N copy of imported robot
            l = len(self.__episode['robot_dof'])
            if l != env.num_env:
                d = self.__episode['robot_dof']
                self.__episode['robot_dof'] = tile_data(d, env.num_env)
                ic(d[..., 0], self.__episode['robot_dof'][..., 0])

    def create_assets(self, gym, sim, counts: Optional[Dict[str, int]] = None):
        cfg = self.cfg
        asset_options = gymapi.AssetOptions()
        # asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        asset_options.enable_gyroscopic_forces = True

        asset_options.vhacd_enabled = False
        asset_options.convex_decomposition_from_submeshes = True

        # asset_options.override_inertia = True
        # asset_options.override_com = True
        # asset_options.linear_damping = cfg.lin_vel_damping
        # asset_options.angular_damping = cfg.ang_vel_damping
        # asset_options.max_linear_velocity = cfg.max_lin_vel
        # asset_options.max_angular_velocity = cfg.max_ang_vel
        if cfg.use_fr3:
            robot_asset = gym.load_urdf(sim,
                                        cfg.asset_root,
                                        cfg.fr3_file,
                                        asset_options)
        elif cfg.use_open_hand:
            robot_asset = gym.load_urdf(sim,
                                        cfg.asset_root,
                                        cfg.open_hand_robot_file,
                                        asset_options)
        elif cfg.use_half_open_hand:
            robot_asset = gym.load_urdf(sim,
                                        cfg.asset_root,
                                        cfg.half_open_hand_robot_file,
                                        asset_options)
        elif cfg.use_custom_hand:
            robot_asset = gym.load_urdf(sim,
                                        cfg.asset_root,
                                        cfg.custom_hand_robot_file,
                                        asset_options)
        elif cfg.use_rg6:
            robot_asset = gym.load_urdf(sim,
                                        cfg.asset_root,
                                        cfg.rg6_robot_file,
                                        asset_options)
        elif cfg.crm_override:
            robot_asset_options = gymapi.AssetOptions()
            robot_asset_options.flip_visual_attachments = True
            robot_asset_options.fix_base_link = True
            robot_asset_options.collapse_fixed_joints = False
            robot_asset_options.disable_gravity = True
            robot_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
            robot_asset_options.thickness = 0.001
            if cfg.hack_open_hand:
                robot_file = "crm-panda/robots/franka_panda_fixed_finger_open.urdf"
            else:
                robot_file = "crm-panda/robots/franka_panda_fixed_finger.urdf"
            # if cfg.init_type =='ik':
            #     robot_file = cfg.robot_file.replace('franka_panda', 'franka_panda_no_coll')
            print(F'load {cfg.asset_root} - {robot_file}')
            robot_asset = gym.load_urdf(
                sim, cfg.asset_root,
                robot_file,
                robot_asset_options)
        else:
            if cfg.init_type == 'ik':
                # robot_file = cfg.robot_file.replace('franka_panda', 'franka_panda_no_coll')
                robot_file = cfg.robot_file
            else:
                robot_file = cfg.robot_file
            print(F'load {cfg.asset_root} - {cfg.robot_file}')
            robot_asset = gym.load_urdf(sim,
                                        cfg.asset_root,
                                        robot_file,
                                        asset_options)

        robot_props = gym.get_asset_rigid_shape_properties(robot_asset)
        left_finger_handle = gym.find_asset_rigid_body_index(
            robot_asset, "panda_leftfinger")
        right_finger_handle = gym.find_asset_rigid_body_index(
            robot_asset, "panda_rightfinger")
        hand_handle = gym.find_asset_rigid_body_index(
            robot_asset, "panda_hand")

        shape_indices = gym.get_asset_rigid_body_shape_indices(robot_asset)
        sil = shape_indices[left_finger_handle]
        sir = shape_indices[right_finger_handle]

        finger_shape_indices = (
            list(range(sil.start, sil.start + sil.count))
            + list(range(sir.start, sir.start + sir.count))
        )
        self.__finger_shape_indices = finger_shape_indices

        cnt = robot_asset
        for i, p in enumerate(robot_props):
            if i in finger_shape_indices:
                p.friction = cfg.default_hand_friction
            else:
                p.friction = cfg.default_body_friction
            p.restitution = cfg.restitution

            if i in [left_finger_handle, right_finger_handle,
                     hand_handle]:
                if i == hand_handle:
                    # Pass through the object
                    print('pass through the object')
                    p.filter |= 0b0110
                    #              ^----[arm]
                    #               ^---[object]
                    #                ^--[table]
                else:
                    # Hits the table and object
                    p.filter |= 0b0100
            else:
                # ARM
                if cfg.disable_table_collision:
                    # Pass through the table and the object
                    print('arm is 0b0111')
                    p.filter |= 0b0111
                else:
                    # Hits the table
                    p.filter |= 0b0100

        gym.set_asset_rigid_shape_properties(robot_asset, robot_props)
        # Cache some properties.
        self.n_bodies = gym.get_asset_rigid_body_count(robot_asset)
        self.n_dofs = gym.get_asset_dof_count(robot_asset)
        dof_props = gym.get_asset_dof_properties(robot_asset)
        dof_lo = []
        dof_hi = []
        vel_lo = []
        vel_hi = []
        eff_hi = []
        for i in range(self.n_dofs):
            dof_lo.append(dof_props['lower'][i])
            dof_hi.append(dof_props['upper'][i])
            vel_lo.append(-dof_props['velocity'][i])
            vel_hi.append(dof_props['velocity'][i])
            eff_hi.append(dof_props['effort'][i])
        self.dof_limits = (
            np.asarray(dof_lo), np.asarray(dof_hi)
        )
        self.eff_limits = np.asarray(eff_hi)

        self.action_space = get_action_space(cfg,
                                             vel_lo,
                                             vel_hi,
                                             self.dof_limits,
                                             cfg.use_fr3)
        ic(self.action_space)
        # ic(gym.get_asset_joint_names(robot_asset))
        # ic('body count', self.n_bodies)
        # ic('body names', gym.get_asset_rigid_body_names(robot_asset))
        # for i in range(self.n_dofs):
        #     ic(gym.get_asset_actuator_name(robot_asset, i))
        self.franka_link_dict = gym.get_asset_rigid_body_dict(robot_asset)
        self.hand_idx = gym.find_asset_rigid_body_index(
            robot_asset, "panda_hand")

        if False:
            # sensor with constraint
            sensor_props = gymapi.ForceSensorProperties()
            sensor_props.enable_forward_dynamics_forces = True
            sensor_props.enable_constraint_solver_forces = False
            sensor_props.use_world_frame = True
            sensor_pose = gymapi.Transform(gymapi.Vec3(0., 0.0, 0.0))
            sensor_idx = gym.create_asset_force_sensor(
                robot_asset, self.hand_idx, sensor_pose, sensor_props)

            # sensor with forward_dynamics_forces
            sensor_props.enable_forward_dynamics_forces = False
            sensor_props.enable_constraint_solver_forces = True
            sensor_idx = gym.create_asset_force_sensor(
                robot_asset, self.hand_idx, sensor_pose, sensor_props)

        if cfg.add_tip_sensor:
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_forward_dynamics_forces = False  # for example gravity
            sensor_options.enable_constraint_solver_forces = True  # for example contacts
            # report forces in world frame (easier to get vertical components)
            sensor_options.use_world_frame = True

            _hand_index = gym.find_asset_rigid_body_index(robot_asset,
                                                          'panda_hand')
            _lf_index = gym.find_asset_rigid_body_index(robot_asset,
                                                        'panda_leftfinger')
            _rf_index = gym.find_asset_rigid_body_index(robot_asset,
                                                        'panda_rightfinger')
            gym.create_asset_force_sensor(
                robot_asset,
                _hand_index,
                gymapi.Transform(),
                sensor_options)
            gym.create_asset_force_sensor(
                robot_asset, _lf_index, gymapi.Transform(), sensor_options)
            gym.create_asset_force_sensor(
                robot_asset, _rf_index, gymapi.Transform(), sensor_options)

        if counts is not None:
            body_count = gym.get_asset_rigid_body_count(robot_asset)
            shape_count = gym.get_asset_rigid_shape_count(robot_asset)
            counts['body'] = body_count
            counts['shape'] = shape_count

        self.assets = {'robot': robot_asset}
        return dict(self.assets)

    def create_actors(self, gym, sim, env, env_id: int):
        cfg = self.cfg

        robot = gym.create_actor(env,
                                 self.assets['robot'],
                                 gymapi.Transform(),
                                 'robot',
                                 env_id,
                                 0b0100
                                 )

        if cfg.disable_table_collision or cfg.disable_all_collsion:
            left_finger_handle = gym.find_actor_rigid_body_index(
                env,
                robot, "panda_leftfinger",
                gymapi.DOMAIN_ACTOR)
            right_finger_handle = gym.find_actor_rigid_body_index(
                env,
                robot, "panda_rightfinger",
                gymapi.DOMAIN_ACTOR)
            hand_handle = gym.find_actor_rigid_body_index(
                env,
                robot, "panda_hand",
                gymapi.DOMAIN_ACTOR)

            hand_part_index = [left_finger_handle, right_finger_handle,
                               hand_handle]

            index_range = gym.get_actor_rigid_body_shape_indices(
                env, robot)
            shape_props = gym.get_actor_rigid_shape_properties(
                env, robot)
            for i, p in enumerate(shape_props):
                # Disable table collision
                if cfg.disable_all_collsion:
                    filter_mask = 0b1111
                elif cfg.disable_table_collision:
                    filter_mask = 0b0101
                else:
                    filter_mask = 0b0100

                p.filter = filter_mask
            gym.set_actor_rigid_shape_properties(
                env, robot, shape_props)

        # Configure the controller.
        robot_dof_props = gym.get_asset_dof_properties(
            self.assets['robot'])

        CTRL_MODES = {
            'jpos': gymapi.DOF_MODE_POS,
            'jvel': gymapi.DOF_MODE_VEL,
            'cpos_n': gymapi.DOF_MODE_POS,
            'jpos+cpos_n': gymapi.DOF_MODE_POS,
        }
        if cfg.joint_param_file is not None:
            with open(cfg.joint_param_file, "rb") as fp:
                joint_params = pickle.load(fp)
            sysid_friction = joint_params['friction']
            sysid_damping = joint_params['damping']
            sysid_armature = joint_params['armature']
        else:
            sysid_friction = [
                0.00174,
                0.01,
                7.5e-09,
                2.72e-07,
                0.39 * 0.2,
                0.12,
                0.9]
            sysid_damping = [2.12, 2.3, 1.29, 2.8, 0.194 * 1.5, 0.3, 0.46]
            sysid_armature = [0.192, 0.54, 0.128, 0.172, 0.15, 0.08, 0.06]
        if self.cfg.use_effort:
            ctrl_mode = gymapi.DOF_MODE_EFFORT
        else:
            ctrl_mode = CTRL_MODES[self.cfg.ctrl_mode]
        # sysid_damping = [2.12, 2.3, 1.29, 2.8, 0.194, 0.3, 0.46]
        # sysid_armature = [0.192, 0.54, 0.128, 0.172, 5.26e-09, 0.08, 0.06]

        if (self.cfg.ctrl_mode in ['cpos_n', 'j_pos', 'jpos+cpos_n'] and
                not self.cfg.use_effort
                and self.cfg.gain != 'variable'
                ):
            # == fixed-gain IG-internal controller ==
            robot_dof_props['driveMode'][...] = ctrl_mode
            robot_dof_props['stiffness'][...] = 200.0  # KP
            robot_dof_props['damping'][...] = 14.0 * 10.0  # KD
            robot_dof_props['armature'][...] = sysid_armature
        else:
            # == variable gain ==
            for i in range(self.n_dofs):
                # robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
                # Apparently this is what ACRONYM expects!
                robot_dof_props['driveMode'][i] = ctrl_mode
                if i < 7:
                    if ctrl_mode == gymapi.DOF_MODE_POS:
                        robot_dof_props['stiffness'][i] = 300.0
                    else:
                        robot_dof_props['stiffness'][i] = 0.0  # 5000
                        # robot_dof_props['damping'][i] = self.cfg.VISCOUS_FRICTION
                        robot_dof_props['friction'][i] = sysid_friction[i]
                        robot_dof_props['damping'][i] = sysid_damping[i]
                        robot_dof_props['armature'][i] = sysid_armature[i]
                else:
                    robot_dof_props['damping'][i] = 1e2
                    robot_dof_props['friction'][i] = 1e3
                    robot_dof_props['armature'][i] = 1e2

        gym.set_actor_dof_properties(env,
                                     robot, robot_dof_props)

        if cfg.measure_joint_torque:
            gym.enable_actor_dof_force_sensors(env, robot)

        return {'robot': robot}

    def reset(self, gym, sim, env, env_id) -> th.Tensor:
        """ Reset the _intrinsic states_ of the robot. """
        # if not self._first:
        #     return [], None
        cfg = self.cfg
        qpos = None
        qvel = None
        if env_id is None:
            env_id = th.arange(self.num_env,
                               dtype=th.int32,
                               device=self.device)

        indices = self.indices[env_id.long()]
        # indices = env_id
        # I = indices.long()
        I = env_id.long()

        if self._first:
            iii = indices.long()
            root_tensor = env.tensors['root']
            # zero-out random stuff
            root_tensor[iii] = 0
            # robot pos
            root_tensor[iii, :3] = env.scene.data['robot_pos']
            # unit quaternion
            root_tensor[iii, 6] = 1
        self._first = False

        # if self._first:
        #    iii = indices.long()
        #    root_tensor = env.tensors['root']
        #    # zero out first
        #    root_tensor[iii, 0] = (
        #        env.scene.data['table_pos'][..., 0]
        #        - 0.5 * env.scene.data['table_dim'][..., 0]
        #        - cfg.keepout_radius
        #    )
        #    if cfg.base_height == 'ground':
        #        root_tensor[iii, 2] = 0.0
        #    elif cfg.base_height == 'table':
        #        root_tensor[iii, 2] = env.scene.data['table_dim'][..., 2]
        #    else:
        #        raise KeyError(F'Unknown base_height = {cfg.base_height}')

        #    # unit quaternion
        #    root_tensor[iii, 6] = 1
        # self._first = False

        if cfg.randomize_hand_friction:
            num_reset: int = len(I)
            self.cur_hand_friction[I] = (th.empty((num_reset,),
                                                  dtype=th.float,
                                                  device=self.device)
                                         .uniform_(
                                             cfg.min_hand_friction,
                                             cfg.max_hand_friction)
                                         )
            hand_friction = dcn(self.cur_hand_friction)
            for i in dcn(env_id):
                # NOTE(ycho): we randomize & the friction _first_,
                # commit to `cur_hand_friction`,
                # then apply that value during apply_domain_randomization.
                dr_params = apply_domain_randomization(
                    gym,
                    env.envs[i],
                    self.handles[i],
                    enable_friction=True,
                    min_friction=hand_friction[i],
                    max_friction=hand_friction[i],
                    target_shape_indices=self.__finger_shape_indices)

        dof_tensor = env.tensors['dof']
        dof_tensor[I, ..., 0] = env.scene.data['robot_dof'][I]
        dof_tensor[I, ..., 1] = 0

        # if not (cfg.import_robot):
        #    # Initialize the joint positions.
        #    if cfg.init_type == 'zero':
        #        dof_tensor[I, ..., 0] = 0.0  # pos (zero?)
        #    elif cfg.init_type == 'sample':
        #        sample_indices = th.randint(self.valid_qs.shape[0],
        #                                    size=(len(I),))
        #        dof_tensor[I, ..., 0] = self.valid_qs[sample_indices.long()]
        #    elif cfg.init_type == 'home':
        #        dof_tensor[I, :, 0] = self.q_home
        #    elif cfg.init_type == 'easy':
        #        dof_tensor[I, :, 0] = self.q_easy
        #    elif cfg.init_type == 'ik-test':
        #        index = th.randperm(len(I), device=self.device)
        #        num_reset_ik = int(len(I) * cfg.init_ik_prob)
        #        reset_w_ik, _ = I[index[:num_reset_ik]].sort()
        #        reset_wo_ik, _ = I[index[num_reset_ik:]].sort()
        #        obj_ids = env.scene.cur_props.ids.long()[reset_w_ik]
        #        n_obj_pos = self.ik_configs[reset_w_ik,
        #                                    self.cursor[reset_w_ik], :7]
        #        n_robot_pos = self.ik_configs[reset_w_ik,
        #                                      self.cursor[reset_w_ik], 7:]
        #        env.tensors['root'][obj_ids, :7] = n_obj_pos

        #        dof_tensor[reset_wo_ik, :, 0] = self.q_home
        #        dof_tensor[reset_w_ik, :, 0] = n_robot_pos

        #        self.cursor[reset_w_ik] = (self.cursor[reset_w_ik] + 1) % 1000

        #    elif cfg.init_type == 'pre-oracle':
        #        if len(I) > 0:
        #            iii = indices.long()
        #            root_tensor = env.tensors['root']
        #            T_b = einops.repeat(
        #                th.eye(4, device=env.device),
        #                '... -> n ...', n=len(I)).contiguous()

        #            T_b[..., : 3, : 3] = matrix_from_quaternion(
        #                root_tensor[iii, 3: 7])
        #            T_b[..., :3, 3] = root_tensor[iii, :3]

        #            obj_ids = env.scene.cur_props.ids.long()[I]
        #            obj_pose = env.tensors['root'][obj_ids, :]
        #            obj_rad = env.scene.cur_props.radii[I]

        #            if True:
        #                contact_point = env.scene.table_pos[I].clone()
        #                contact_point[..., 0] -= env.scene.table_dims[I, 0] / 2
        #                contact_point[..., 2] = 0.5
        #                # (obj_pose[..., 2] +obj_rad[:]*1.1)

        #            else:
        #                # Get contact point from point set
        #                point_sets = env.scene.cur_cloud[I]
        #                point_sets = apply_pose_tq(obj_pose[..., None, 0:7],
        #                                           point_sets)
        #                right_most_cloud = th.argmin(point_sets[..., 1], -1,
        #                                             keepdim=True)

        #                contact_point = point_sets[right_most_cloud]
        #                contact_point = th.gather(
        #                    point_sets, -1,
        #                    right_most_cloud[..., None].repeat(1, 1, 3))
        #                contact_point = th.take_along_dim(
        #                    point_sets, right_most_cloud[..., None], dim=-2)

        #                contact_point[..., 1] -= 0.01
        #                contact_point[..., 2] = obj_pose[..., None, 2] + 0.05

        #            # EE_ori = th.tensor([1/math.sqrt(2),
        #            #                     1/math.sqrt(2), 0, 0], dtype=th.float,
        #            # device=env.device)[None].repeat(len(I), 1)

        #            EE_ori = th.tensor(
        #                [1, 0, 0, 0],
        #                dtype=th.float, device=env.device)[None].repeat(
        #                len(I),
        #                1)

        #            EE_pos = th.cat([contact_point, EE_ori], -1)
        #            q_ik, suc = solve_ik_from_contact(
        #                env, I,
        #                self.q_home, T_b,
        #                self._check_col,
        #                len(I),
        #                # offset=0.023,
        #                offset=0.02,
        #                yaw_random=True,
        #                # cone_max_theta=0.0,
        #                cone_max_theta=math.radians(30.0),
        #                # Oversample contacts
        #                # and IK solutions by 32x.
        #                query_multiplier=1,
        #                EE_pos=EE_pos[:, None]
        #            )
        #            dof_tensor[I, :, 0] = q_ik

        #    elif cfg.init_type == 'ik':
        #        if True:
        #            # I = env_id
        #            # iii = robot indices
        #            assert (env.scene.cfg.load_normal)
        #            index = th.randperm(len(I), device=self.device)
        #            num_reset_ik = int(len(I) * cfg.init_ik_prob)
        #            reset_w_ik, _ = I[index[:num_reset_ik]].sort()
        #            reset_wo_ik, _ = I[index[num_reset_ik:]].sort()

        #            iii = indices.long()
        #            iiii = iii[index[:num_reset_ik]]

        #            root_tensor = env.tensors['root']
        #            T_b = einops.repeat(
        #                th.eye(4, device=env.device),
        #                '... -> n ...', n=num_reset_ik).contiguous()
        #            # print(root_tensor.shape, iii.shape, T_b.shape)
        #            T_b[..., : 3, : 3] = matrix_from_quaternion(
        #                root_tensor[iiii, 3: 7])
        #            T_b[..., :3, 3] = root_tensor[iiii, :3]

        #            # if len(reset_wo_ik)>0:
        #            #     dof_tensor[reset_wo_ik, :, 0] = self.q_home
        #            #     pass
        #            dof_tensor[I, :, 0] = self.q_home[None]
        #            if len(reset_w_ik) > 0:
        #                q_ik, suc = solve_ik_from_contact(
        #                    env, reset_w_ik,
        #                    self.q_home, T_b,
        #                    self._check_col,
        #                    num_reset_ik,
        #                    # offset=0.023,
        #                    offset=0.02,
        #                    yaw_random=True,
        #                    # cone_max_theta=0.0,
        #                    cone_max_theta=math.radians(30.0),
        #                    # Oversample contacts
        #                    # and IK solutions by 32x.
        #                    query_multiplier=128
        #                )
        #                dof_tensor[reset_w_ik, :, 0] = q_ik
        #                # which envs sucessfully got their IK solution?
        #                self._ik_suc = reset_w_ik[suc]
        #            else:
        #                self._ik_suc = None
        #    elif cfg.init_type == 'box_sample':
        #        samples = th.rand(len(I), 7, device=self.device)
        #        diff = self.box_max - self.box_min
        #        noise = th.randn(len(I), 7, device=self.device) * 0.03
        #        qs = diff[None] * samples + self.box_min + noise
        #        dof_tensor[I, ..., 0] = qs
        #    elif cfg.init_type == 'ik-presample':
        #        #has_ceiling = env.scene.cur_props.env_code[I, 18] > 0
        #        has_ceil = env.scene.data['pose_meta']['case']['has_ceil'][I]
        #        height = (env.scene.data['table_pos'][I, 2]
        #                  - env.scene.data['table_dim'][I, 2] / 2)
        #        height += cfg.height_offset
        #        num_buckets, num_ik = self.cabinet_ik.shape[:2]
        #        # HACK hardcorded height assumption
        #        bucket_indices = (((height + 0.045) / 0.6) *
        #                          (num_buckets - 1)).to(th.long).clamp(0, num_buckets - 1)
        #        ik_indices = th.randint(high=num_ik,
        #                                size=(len(I),),
        #                                dtype=th.long,
        #                                device=env.device
        #                                )
        #        dof_tensor[I[has_ceil], ..., 0] = self.cabinet_ik[
        #            bucket_indices[has_ceil], ik_indices[has_ceil]
        #        ]
        #        dof_tensor[I[~has_ceil], ..., 0] = self.wall_ik[
        #            bucket_indices[~has_ceil], ik_indices[~has_ceil]
        #        ]
        #        rand_q_7 = (
        #            th.rand(len(I),
        #                    device=env.device) *
        #            (cfg.box_max[-1] - cfg.box_min[-1]) + cfg.box_min[-1])
        #        dof_tensor[I, ..., -1, 0] = rand_q_7

        #    # Currently, we always initialize velocity to zero.
        #    dof_tensor[..., 1].index_fill_(0, I, 0.0)
        # else:
        #    dof_tensor[I] = th.as_tensor(self.__episode['robot_dof'],
        #                                 dtype=dof_tensor.dtype,
        #                                 device=dof_tensor.device)[I]

        if cfg.export_robot:
            if 'robot_dof' not in self.__episode:
                self.__episode['robot_dof'] = dcn(dof_tensor)

        if cfg.ema_action:
            self._ema_action[I] = 0.0

        # self._first = False

        if self.cfg.ctrl_mode == 'jvel':
            # Currently, we always initialize control velocity to zero as well.
            self._control[I] = 0.0
            qvel = self._control
        elif self.cfg.ctrl_mode in ['jpos', 'jpos+cpos_n']:
            self._control[I] = dof_tensor[I, ..., 0]
            self._target[I] = self._control[I]
            qpos = self._control
        elif self.cfg.ctrl_mode == 'cpos_n':
            if cfg.use_effort:
                self._control[I] = 0.
            else:
                self._control[I] = dof_tensor[I, ..., 0]
                qpos = self._control
        elif self.cfg.ctrl_mode == 'CI':
            self._control[I] = 0.
        elif self.cfg.ctrl_mode == 'osc':
            self._control[I] = 0.
        else:
            raise KeyError('Unknown ctrl_mode')

        # FIXME(ycho): cannot reset with `dof_tensor`,
        # we need to wait until the hand_state becomes
        # valid again.
        # self.pose_error.reset(dof_tensor[..., :7], I)
        self.pose_error.clear(I)

        # reset ee_wrench to zero
        self.ee_wrench.index_fill_(0, I, 0.0)
        if cfg.regularize is not None:
            self.energy.index_fill_(0, I, 0.0)

        return indices, qpos, qvel

    @nvtx.annotate("step_controller")
    def step_controller(self, gym, sim, env):
        cfg = self.cfg
        # actions = self._target

        # TODO(ycho): think if this exactly what we want.
        # do we need _indexed() form in this case??
        indices = self.indices
        j_eef = self.j_eef
        # body_tensor = env.tensors['body'].clone().view(self.num_env, -1, 13)
        # ori = body_tensor[:,self.hand_idx, 3:7]
        # j_eef = get_analytic_jacobian(ori,self.j_eef,self.num_env,self.device)
        j_eef_T = th.transpose(j_eef, 1, 2)

        if cfg.track_object:
            assert (cfg.clip_bound)
            with nvtx.annotate("track_object"):
                # FIXME(ycho): hardcoded introspection !!!
                # obj_ids = env.scene.body_ids.long()
                # obj_pos = env.tensors['body'][obj_ids, ..., :3]
                # cur_props.ids.long()
                obj_ids = env.scene.data['obj_id'].long()
                obj_pos = env.tensors['root'][obj_ids, :3]
                obj_rad = env.scene.data['obj_radius']  # cur_props.radii
                # NOTE(ycho): object-specific clipping bounds
                obj_bound = th.stack(
                    [obj_pos - obj_rad[..., None] - cfg.obj_margin,
                     obj_pos + obj_rad[..., None] + cfg.obj_margin],
                    dim=-2)
                self.pose_error.update_pos_bound(obj_bound)

        cache = {}
        if self.cfg.ctrl_mode == 'cpos_n':
            # Solve damped least squares.
            hand_state = env.tensors['body'][self.hand_ids]
            update_indices = None
            if self.delay_counter is not None:
                # update gain and joint target for envs where delay is finished
                update_indices = th.argwhere(
                    self.delay_counter == 0).squeeze(-1)
                q_pos = env.tensors['dof'][..., 0]
                self.pose_error.update(
                    hand_state,
                    q_pos=q_pos,
                    action=None,
                    j_eef=j_eef,
                    relative=(cfg.target_type == 'rel'),
                    indices=update_indices,
                    update_pose=False,
                    recompute_orn=True
                )
            b = self.pose_error(
                hand_state,
                env.tensors['dof'][..., 0],
                j_eef
            )

            if cfg.use_effort:
                self.controller.update_gain(self.gains,
                                            indices=update_indices)
                self._control[..., :7] = self.controller(
                    self.mm,
                    env.tensors['dof'][..., 1], b
                )
                # (efforts will be applied later)
            else:
                # 4,6,7 / 4,7 / 4,7
                self._control[..., :7] = env.tensors['dof'][..., 0] + b
                gym.set_dof_position_target_tensor_indexed(
                    sim, gymtorch.unwrap_tensor(self._control),
                    gymtorch.unwrap_tensor(indices),
                    len(indices)
                )

        elif self.cfg.ctrl_mode == 'CI' or self.cfg.ctrl_mode == 'osc':
            with nvtx.annotate("OSC"):
                hand_state = env.tensors['body'][self.hand_ids]
                # [1] Relative
                if self.cfg.target_type == 'rel':
                    b = self.pose_error(hand_state[..., :7],
                                        axa=True)
                else:
                    raise ValueError('`abs` technically supported now')
                assert (self.cfg.use_effort)
                hand_vel = hand_state[..., 7:]
                self._control[..., :7] = self.controller(
                    self.mm,
                    j_eef,
                    hand_vel,
                    b,
                    cache=cache)

        elif self.cfg.ctrl_mode == 'jvel':
            self._control[...] = self._target
            # NOTE(ycho): assumes rank of action == 1
            if self.cfg.use_effort:
                jvel = env.tensors['dof'][..., 1]
                self._control[...] = self.cfg.KD_pos * (self._control - jvel)
                self.ee_wrench = self.j_eef
                # (efforts will be applied later)
            else:
                gym.set_dof_velocity_target_tensor_indexed(
                    sim, gymtorch.unwrap_tensor(self._control),
                    gymtorch.unwrap_tensor(indices),
                    len(indices)
                )
        elif self.cfg.ctrl_mode == 'jpos':
            self._control[..., :7] = self._target
            if cfg.rel_clamp_joint_target is not None:
                self._control[..., :7] = (self._control[..., :7]
                                          .clamp(self.__reduced_q_lo,
                                                 self.__reduced_q_hi))
            if self.cfg.use_effort:
                jpos = env.tensors['dof'][..., 0]
                jvel = env.tensors['dof'][..., 1]
                self._control[...] = self.kp * (
                    self._control - jpos) - self.kd * (jvel)
                self.ee_wrench = self.j_eef
            else:
                gym.set_dof_position_target_tensor_indexed(
                    sim, gymtorch.unwrap_tensor(self._control[..., :7]),
                    gymtorch.unwrap_tensor(indices),
                    len(indices)
                )
        elif self.cfg.ctrl_mode == 'jpos+cpos_n':
            hand_state = env.tensors['body'][self.hand_ids]
            update_indices = None
            if self.delay_counter is not None:
                # update gain and joint target for envs where delay is finished
                update_indices = th.argwhere(
                    self.delay_counter == 0).squeeze(-1)
                q_pos = env.tensors['dof'][..., 0]
                self.pose_error.update(
                    hand_state,
                    q_pos=q_pos,
                    action=None,
                    j_eef=j_eef,
                    relative=(cfg.target_type == 'rel'),
                    indices=update_indices,
                    update_pose=False,
                    recompute_orn=True
                )
            b = self.pose_error(
                hand_state,
                env.tensors['dof'][..., 0],
                j_eef
            )
            if cfg.use_effort:
                jpos = env.tensors['dof'][..., 0]
                jvel = env.tensors['dof'][..., 1]
                self.controller.update_gain(self.gains,
                                            indices=update_indices)
                # average the efforts
                self._control[..., :7] = 0.5 * (self.controller(
                    self.mm,
                    env.tensors['dof'][..., 1], b
                ) + (self.cfg.KP_pos * (self._target - jpos)
                     - self.cfg.KD_pos * (jvel))
                )
                self.ee_wrench = self.j_eef  # ???
            else:
                # average the joint position targets
                self._control[..., :7] = 0.5 * (
                    env.tensors['dof'][..., 0] + b
                    + self._target
                )
                gym.set_dof_position_target_tensor_indexed(
                    sim, gymtorch.unwrap_tensor(self._control),
                    gymtorch.unwrap_tensor(indices),
                    len(indices)
                )

        if self.cfg.use_effort:
            # NOTE(ycho): be default we zero-out
            # possible actuation forces against the fingers.
            self._control[:, 7:] = 0.

            if self.cfg.add_control_noise:
                samples = th.randn_like(self._control[:, :7])
                self._control[:, :7] += self.cfg.control_noise_mag * samples

            if cfg.regularize in ('energy', 'torque'):
                current_energy = (
                    self._control[..., :7]
                    if cfg.regularize == 'torque' else self._control
                    [..., :7] * env.tensors['dof'][..., :7, 1])
                self.energy += th.abs(current_energy).sum(dim=-1)

            if True:
                # if not th.isfinite(self._control).all():
                #    jpos = env.tensors['dof'][..., 0]
                #    jvel = env.tensors['dof'][..., 1]
                #    raise ValueError('trying to output NaN torque')
                with nvtx.annotate("apply_effort"):
                    gym.set_dof_actuation_force_tensor_indexed(
                        sim, gymtorch.unwrap_tensor(self._control),
                        gymtorch.unwrap_tensor(indices),
                        len(indices)
                    )
            if self.delay_counter is not None:
                self.delay_counter -= 1
            # env.scene.apply_guide_force(gym, sim, env)

    def apply_gains(self, gym, sim, env, gains: th.Tensor):
        cfg = self.cfg

        # Only active if using variable gains
        if cfg.gain != 'variable':
            return

        if cfg.use_effort:
            if cfg.ctrl_mode in ('jpos', 'jvel'):
                self.kp = gains[..., :7]
                self.kd = gains[..., 7:] * th.sqrt(self.kp)
            else:
                self.controller.update_gain(gains)
        else:
            # Set internal controller gains
            # TODO(ycho): consider skipping this part...
            # actions_np = dcn(actions)
            # gains = actions_np[..., -14:]
            # TODO(ycho): necessary?
            gains = dcn(gains)
            kp = gains[..., :7]
            kd = gains[..., 7:] * np.sqrt(kp)
            for i in range(self.num_env):
                props = gym.get_actor_dof_properties(
                    env.envs[i],
                    self.handles[i]
                )
                props['stiffness'] = kp[i]
                props['damping'] = kd[i]
                gym.set_actor_dof_properties(env.envs[i],
                                             self.handles[i],
                                             props)

    @nvtx.annotate("Franka.apply_actions")
    def apply_actions(self, gym, sim, env, actions,
                      done=None):
        """ Set the actuation targets for the simulator. """
        if actions is None:
            print('actions is None.')
            return
        cfg = self.cfg

        if cfg.ema_action:
            n_subgoal = self._ema_action.shape[-1]
            actions[...,
                    :n_subgoal] = (cfg.ema_action * actions[...,
                                                            :n_subgoal] + (1 - cfg.ema_action) * self._ema_action)
            self._ema_action = actions[..., :n_subgoal].clone()

        if self.delay_counter is not None:
            self.delay_counter[:] = th.randint(1, self.cfg.max_control_delay,
                                               (self.num_env, ),
                                               device=self.device)
        if cfg.regularize == 'action':
            # scale = [1/cfg.max_pos, 1/cfg.max_pos, 1/cfg.max_pos,
            #          1/cfg.max_ori, 1/cfg.max_ori, 1/cfg.max_ori]
            # scale = th.as_tensor(scale, dtype=actions.dtype, device=actions.device)
            # # scale = (1/cfg.KP_pos) * th.ones_like(actions[..., -7:-14])
            self.energy[:] = th.linalg.norm(
                env.prev_action[..., -7: -14],
                ord=2, dim=-1)
            # self.energy[:] = th.linalg.norm(actions[..., :6] * scale, ord=2, dim=-1)
        elif cfg.regularize is not None:
            self.energy[:] = 0

        if cfg.ctrl_mode in ('CI', 'osc'):
            state = env.tensors['body'][self.hand_ids]
            self.pose_error.update(state, actions * (~done[..., None]),
                                   relative=(cfg.target_type == 'rel'))
            if cfg.lock_orn:
                self.pose_error.target[..., 3:7].fill_(0)
                self.pose_error.target[..., 4] = 1.0
            if cfg.gain == 'variable':
                self.controller.update_gain(actions[..., -12:])
        elif cfg.ctrl_mode in ('cpos_n', 'cpos_a'):
            q_pos = env.tensors['dof'][..., 0]
            state = env.tensors['body'][self.hand_ids, ..., :7]
            gains = actions[..., -14:]
            if self.delay_counter is not None:
                # only update cartesian pose in here
                self.pose_error.pose_error.update(
                    state, actions, relative=(cfg.target_type == 'rel')
                )
                self.gains = gains.clone()
                # update gain and target for envs where delay is zero
                update_indices = th.argwhere(
                    self.delay_counter == 0).squeeze(-1)
                self.pose_error.update(state, q_pos, actions, self.j_eef,
                                       relative=(cfg.target_type == 'rel'),
                                       indices=update_indices,
                                       update_pose=False)
                self.controller.update_gain(actions[..., -14:],
                                            indices=update_indices)
            else:
                self.gains = gains.clone()
                self.pose_error.update(state, q_pos, actions, self.j_eef,
                                       relative=(cfg.target_type == 'rel'))
                if cfg.lock_orn:
                    self.pose_error.pose_error.target[..., 3:7].fill_(0)
                    self.pose_error.pose_error.target[..., 4] = 1.0
                self.apply_gains(gym, sim, env, gains)
        elif cfg.ctrl_mode in ('jpos',):
            # ++ JOINT ++
            if done is not None:
                keep = ~done[..., None]
                actions.mul_(keep)
                if cfg.target_type == 'abs':
                    actions[done, ..., :7] = env.tensors['dof'][done, ..., 0]
            if cfg.target_type == 'rel':
                actions = actions.clone()
                actions[..., 0:7].add_(
                    env.tensors['dof'][..., 0])

            # TODO(ycho): I guess this is fine,
            # but why does this code-path not use the controller?
            self._target[...] = actions[..., 0:7]
            gains = actions[..., -14:]
            self.apply_gains(gym, sim, env, gains)
        elif cfg.ctrl_mode in ['jpos+cpos_n']:
            assert (cfg.target_type == 'rel')

            # Clear 'null' actions
            if done is not None:
                keep = ~done[..., None]
                actions.mul_(keep)

            # parse action
            d_joint = actions[..., 0:7]
            d_eepos = actions[..., 7:13]
            gains = actions[..., 13:27]

            # process jpos
            self._target[...] = (
                env.tensors['dof'][..., 0] + d_joint
            )

            # process cpos-n
            q_pos = env.tensors['dof'][..., 0]
            state = env.tensors['body'][self.hand_ids, ..., :7]
            if self.delay_counter is not None:
                # only update cartesian pose in here
                self.pose_error.pose_error.update(
                    state, d_eepos, relative=(cfg.target_type == 'rel')
                )
                self.gains = gains
                # update gain and target for envs where delay is zero
                update_indices = th.argwhere(
                    self.delay_counter == 0).squeeze(-1)
                self.pose_error.update(state, q_pos, d_eepos, self.j_eef,
                                       relative=(cfg.target_type == 'rel'),
                                       indices=update_indices,
                                       update_pose=False)
                self.controller.update_gain(gains,
                                            indices=update_indices)
            else:
                # ????
                self.gains = gains
                self.pose_error.update(state, q_pos, d_eepos, self.j_eef,
                                       relative=(cfg.target_type == 'rel'))
                if cfg.lock_orn:
                    self.pose_error.pose_error.target[..., 3:7].fill_(0)
                    self.pose_error.pose_error.target[..., 4] = 1.0
                self.apply_gains(gym, sim, env, gains)
        else:
            raise ValueError(F'Unknown ctrl_mode = {cfg.ctrl_mode}')
