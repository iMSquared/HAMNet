#!/usr/bin/env python3

from isaacgym import gymtorch, gymapi

# GENERAL PACKAGES
import numpy as np
np.float = np.float32

from typing import Optional, Dict, Union, Mapping, Tuple, List
from dataclasses import dataclass, InitVar, replace
from pathlib import Path
from copy import deepcopy
from gym import spaces
import json
from functools import partial
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter

# MODELS
from ham.models.common import (transfer, map_struct)
from ham.models.rl.v6.ppo import PPO
from ham.models.rl.generic_state_encoder import MLPStateEncoder
from ham.models.rl.nets import (
    VNet,
    PiNet,
    NoOpPiNet,
    NoOpVNet
)
from ham.util.torch_util import dcn

# ENV
from ham.env.arm_env import (ArmEnvConfig, make_arm_env,
                             OBS_BOUND_MAP, _identity_bound,
                             _sanitize_bounds)
from ham.env.env.wrap.base import WrapperEnv
from ham.env.env.wrap.normalize_env import NormalizeEnv
from ham.env.env.wrap.monitor_env import MonitorEnv
from ham.env.env.wrap.popdict import PopDict

# APP/CONFIG/RUNTIME
from ham.env.util import set_seed
from ham.util.config import (ConfigBase, recursive_replace_map)
from ham.util.hydra_cli import hydra_cli
from ham.util.path import RunPath, ensure_directory
from ham.train.ckpt import last_ckpt, step_from_ckpt
from ham.train.hf_hub import (upload_ckpt, HfConfig, GroupConfig)
from ham.train.wandb import with_wandb, WandbConfig
from ham.train.util import (
    assert_committed
)

# ENV WRAPPERS
from env_wrappers import (
    AddPhysParams,
    AddPrevAction,
    AddObjectKeypoint,
    AddObjectFullCloud,
    AddSceneFullClouds,
    RelGoal,
    UnicornEmbedClouds,
    TuneDomainSampling,
)


# DEBUGGING ENV WRAPPERS
from ham.env.env.wrap.draw_bbox_kpt import DrawGoalBBoxKeypoint, DrawObjectBBoxKeypoint
from ham.env.env.wrap.draw_clouds import DrawClouds
from ham.env.env.wrap.draw_mesh_edge import DrawMeshEdge
from env_wrappers import (DrawGoalPose,
                          DrawObjPose,
                          DrawPosBound,
                          DrawDebugLines,
                          DrawCom)

from icecream import ic


@dataclass
class PolicyConfig(ConfigBase):
    """ Actor-Critic policy configuration. """
    actor: PiNet.Config = PiNet.Config()
    value: VNet.Config = VNet.Config()

    dim_state: InitVar[Optional[int]] = None
    dim_act: InitVar[Optional[Tuple[int, ...]]] = None

    def __post_init__(self,
                      dim_state: Optional[int] = None,
                      dim_act: Optional[int] = None):
        if dim_state is not None:
            self.actor = replace(self.actor, dim_feat=dim_state)
            self.value = replace(self.value, dim_feat=dim_state)
        if dim_act is not None:
            self.actor = replace(self.actor, dim_act=dim_act)


@dataclass
class NetworkConfig(ConfigBase):
    """ Overall network configuration. """
    state: MLPStateEncoder.Config = MLPStateEncoder.Config()
    policy: PolicyConfig = PolicyConfig()

    obs_space: InitVar[Union[int, Dict[str, int], None]] = None
    act_space: InitVar[Optional[int]] = None

    def __post_init__(self, obs_space=None, act_space=None):
        self.state = replace(self.state,
                             obs_space=obs_space,
                             act_space=act_space)
        try:
            policy = replace(self.policy,
                             dim_state=self.state.state.dim_out,
                             dim_act=act_space)
            self.policy = policy
        except AttributeError:
            pass


@dataclass
class Config(WandbConfig, HfConfig, GroupConfig, ConfigBase):
    # WandbConfig parts
    project: str = 'arm-ppo'
    use_wandb: bool = True
    # HfConfig (huggingface) parts
    hf_repo_id: Optional[str] = 'yycho0108/pkm-arm'
    use_hfhub: bool = True
    # General experiment / logging
    force_commit: bool = False
    description: str = ''
    path: RunPath.Config = RunPath.Config(root='/tmp/ham/ppo-arm/')
    global_device: Optional[str] = None

    env: ArmEnvConfig = ArmEnvConfig(which_robot='franka')
    agent: PPO.Config = PPO.Config()

    # State/Policy network configurations
    net: NetworkConfig = NetworkConfig()

    # Loading / continuing from prevous runs
    load_ckpt: Optional[str] = None
    strict_load_ckpt: bool = True

    # Configure inputs
    remove_state: bool = False
    remove_robot_state: bool = False
    remove_all_state: bool = False
    add_phys_params: bool = False
    add_com: bool = False
    add_keypoint: bool = False
    add_object_full_cloud: bool = False
    add_goal_full_cloud: bool = False
    add_scene_full_cloud: bool = False
    scene_cloud: AddSceneFullClouds.Config = AddSceneFullClouds.Config()
    add_prev_action: bool = True
    const_norm_act: bool = False
    zero_out_prev_action: bool = False

    # Determines which inputs, even if they remain
    # in the observation dict, are not processed
    # by the state representation network.
    state_net_blocklist: Optional[List[str]] = None

    # Determine goal format.
    use_rel_goal: bool = False
    use_6d_rel_goal: bool = False
    add_abs_goal: bool = False

    # Embed point cloud.
    use_unicorn: bool = False
    unicorn: UnicornEmbedClouds.Config = UnicornEmbedClouds.Config()
    remove_cloud: bool = False

    # Apply curriculum.
    use_tune_domain_sampling: bool = False
    tune_domain_sample: TuneDomainSampling.Config = TuneDomainSampling.Config()

    # Apply normalization.
    use_norm: bool = True
    normalizer: NormalizeEnv.Config = NormalizeEnv.Config()

    # Monitor progress.
    use_monitor: bool = True
    monitor: MonitorEnv.Config = MonitorEnv.Config()

    # Mixed-precision training.
    use_amp: bool = False

    # Draw debug constructs.
    draw_debug_lines: bool = False
    draw_patch_attn: bool = False
    draw_clouds: bool = False
    draw_com: bool = False
    draw_joint_torque: bool = False
    draw_mesh_edge: bool = False

    # Extra policy configs
    hack_noop_policy: bool = False
    override_noop_policy_log_std: bool = False
    split_noop_policy_log_std: bool = False
    train_noop_policy_log_std: bool = False

    # Recursively invoke __post_init__()
    # only if `finalize = True`.
    finalize: bool = False

    def __post_init__(self):
        self.group = F'{self.machine}-{self.env_name}-{self.model_name}-{self.tag}'
        self.name = F'{self.group}-{self.env.seed:06d}'
        if not self.finalize:
            return
        # WARNING: VERY HAZARDOUS
        use_dr_on_setup = self.env.single_object_scene.use_dr_on_setup
        use_dr = self.env.single_object_scene.use_dr
        self.env = recursive_replace_map(
            self.env, {
                'single_object_scene.use_dr_on_setup': use_dr_on_setup,
                'single_object_scene.use_dr': use_dr,
            })

        if self.global_device is not None:
            dev_id: int = int(str(self.global_device).split(':')[-1])
            self.env = recursive_replace_map(self.env, {
                'graphics_device_id': (dev_id if self.env.use_viewer else -1),
                'compute_device_id': dev_id,
                'th_device': self.global_device,
            })
            self.agent = recursive_replace_map(self.agent, {
                'device': self.global_device})


def setup(cfg: Config):
    # Maybe it's related to jit
    if cfg.global_device is not None:
        th.cuda.set_device(cfg.global_device)
    th.backends.cudnn.benchmark = True

    commit_hash = assert_committed(force_commit=cfg.force_commit)
    path = RunPath(cfg.path)
    print(F'run = {path.dir}')
    return path


class AddTensorboardWriter(WrapperEnv):
    def __init__(self, env):
        super().__init__(env)
        self._writer = None

    def set_writer(self, w):
        self._writer = w

    @property
    def writer(self):
        return self._writer


def load_env(cfg: Config, path,
             freeze_env: bool = False, **kwds):
    env = make_arm_env(cfg.env)
    env.setup()
    env.gym.prepare_sim(env.sim)
    env.refresh_tensors()
    env.reset()

    env = AddTensorboardWriter(env)

    obs_bound = None
    if cfg.use_norm:
        obs_bound = {}

        # Populate `obs_bound` with defaults
        # from `ArmEnv`.
        obs_bound['goal'] = OBS_BOUND_MAP.get(cfg.env.goal_type)
        obs_bound['object_state'] = OBS_BOUND_MAP.get(
            cfg.env.object_state_type)
        obs_bound['hand_state'] = OBS_BOUND_MAP.get(cfg.env.hand_state_type)
        obs_bound['robot_state'] = OBS_BOUND_MAP.get(cfg.env.robot_state_type)

        if cfg.normalizer.norm.stats is not None:
            obs_bound.update(deepcopy(cfg.normalizer.norm.stats))

    print(obs_bound)

    def __update_obs_bound(key, value, obs_bound,
                           overwrite: bool = True):
        if not cfg.use_norm:
            return
        if value is None:
            obs_bound.pop(key, None)

        if key in obs_bound:
            if overwrite:
                print(F'\t WARN: key = {key} already in obs_bound !')
            else:
                raise ValueError(F'key = {key} already in obs_bound !')

        obs_bound[key] = value
    update_obs_bound = partial(__update_obs_bound, obs_bound=obs_bound)

    if cfg.env.task.use_pose_goal:
        if cfg.add_goal_full_cloud:
            update_obs_bound('goal_cloud',
                             OBS_BOUND_MAP.get('cloud'))
        else:
            update_obs_bound('goal',
                             OBS_BOUND_MAP.get(cfg.env.goal_type))

    # Crude check for mutual exclusion
    # Determines what type of privileged "state" information
    # the policy will receive, as observation.
    assert (
        np.count_nonzero(
            [cfg.remove_state, cfg.remove_robot_state, cfg.remove_all_state])
        <= 1)
    if cfg.remove_state:
        env = PopDict(env, ['object_state'])
        update_obs_bound('object_state', None)
    elif cfg.remove_robot_state:
        env = PopDict(env, ['hand_state'])
        update_obs_bound('hand_state', None)
    elif cfg.remove_all_state:
        env = PopDict(env, ['hand_state', 'object_state'])
        update_obs_bound('hand_state', None)
        update_obs_bound('object_state', None)

    if cfg.add_phys_params:
        env = AddPhysParams(env, 'phys_params', add_com=cfg.add_com)
        if cfg.add_com:
            update_obs_bound('phys_params',
                             OBS_BOUND_MAP.get('phys_params_w_com'))
        else:
            update_obs_bound('phys_params', OBS_BOUND_MAP.get('phys_params'))

    if cfg.add_keypoint:
        env = AddObjectKeypoint(env, 'object_keypoint')
        update_obs_bound('object_keypoint', OBS_BOUND_MAP.get('keypoint'))

    if cfg.add_object_full_cloud:
        # mutually exclusive w.r.t. `use_cloud`
        # i.e. the partial point cloud coming from
        # the camera.
        # assert (cfg.camera.use_cloud is False)
        goal_key = None
        if cfg.add_goal_full_cloud:
            goal_key = 'goal_cloud'
        env = AddObjectFullCloud(env,
                                 'cloud',
                                 goal_key=goal_key,
                                 hack_canonical=False)
        update_obs_bound('cloud', OBS_BOUND_MAP.get('cloud'))
        if goal_key is not None:
            update_obs_bound(goal_key, OBS_BOUND_MAP.get('cloud'))

    if cfg.add_scene_full_cloud:
        env = AddSceneFullClouds(cfg.scene_cloud, env)
        update_obs_bound(cfg.scene_cloud.key_focal,
                         OBS_BOUND_MAP.get('cloud'))
        update_obs_bound(cfg.scene_cloud.key_peripheral,
                         OBS_BOUND_MAP.get('cloud'))
        if cfg.scene_cloud.key_hand_focal is not None:
            update_obs_bound(cfg.scene_cloud.key_hand_focal,
                             OBS_BOUND_MAP.get('cloud'))

    if cfg.add_prev_action:
        env = AddPrevAction(env, 'previous_action',
                            zero_out=cfg.zero_out_prev_action)
        if not cfg.const_norm_act:
            update_obs_bound('previous_action', _identity_bound(
                env.observation_space['previous_action'].shape
            ))
        else:
            act_mean = (env.action_space.low + env.action_space.high) / 2
            act_var = (-env.action_space.low + env.action_space.high) / 2
            update_obs_bound('previous_action', _sanitize_bounds(
                (act_mean, act_var)
            ))

    if cfg.use_tune_domain_sampling:
        env = TuneDomainSampling(env,
                                 cfg.tune_domain_sample)

    # Use relative goal between current object pose
    # and the goal pose, instead of absolute goal.
    if cfg.use_rel_goal:
        env = RelGoal(env, 'goal',
                      use_6d=cfg.use_6d_rel_goal,
                      add_abs_goal=cfg.add_abs_goal
                      )
        if cfg.use_6d_rel_goal:
            update_obs_bound('goal',
                             OBS_BOUND_MAP.get('relpose6d'))
        else:
            update_obs_bound('goal',
                             OBS_BOUND_MAP.get('relpose'))
        if cfg.add_abs_goal:
            if cfg.use_6d_rel_goal:
                update_obs_bound('abs_goal',
                                 OBS_BOUND_MAP.get('pose6d'))
            else:
                update_obs_bound('abs_goal',
                                 OBS_BOUND_MAP.get('pose'))

    # == DRAW, LOG, RECORD ==
    if cfg.draw_debug_lines:
        check_viewer = kwds.pop('check_viewer', True)
        env = DrawDebugLines(DrawDebugLines.Config(
            draw_workspace=kwds.pop('draw_workspace', False),
            draw_wrench_target=kwds.pop('draw_wrench_target', False),
            draw_cube_action=kwds.pop('draw_hand_action', False)
        ), env,
            check_viewer=check_viewer)
        env = DrawObjectBBoxKeypoint(env)
        env = DrawGoalBBoxKeypoint(env)
        env = DrawGoalPose(env,
                           check_viewer=check_viewer)
        env = DrawObjPose(env,
                          check_viewer=check_viewer)
        # Some alternative visualizations are available below;
        # [1] draw the goal as a "pose" frame axes
        # env = DrawTargetPose(env,
        #                      check_viewer=check_viewer)
        # [2] Draw franka EE boundary
        if cfg.env.franka.track_object:
            env = DrawPosBound(env,
                               check_viewer=check_viewer)
        # [3] Draw input point cloud observations as spheres.
        # Should usually be prevented, so check_viewer=True
        if cfg.draw_clouds:
            env = DrawClouds(env,
                             check_viewer=True,
                             stride=4,
                             style='cross',
                             cloud_key='peripheral_cloud'
                             )

        if cfg.draw_com:
            env = DrawCom(env)
        if cfg.draw_mesh_edge:
            env = DrawMeshEdge(env, True)

    # == MONITOR PERFORMANCE ==
    if cfg.use_monitor:
        env = MonitorEnv(cfg.monitor, env)

    # == Normalize environment ==
    # NOTE(ycho): normalization must come after the monitoring code,
    # since it overwrites env statistics.
    if cfg.use_norm:
        cfg = recursive_replace_map(cfg,
                                    {'normalizer.norm.stats': obs_bound})
        env = NormalizeEnv(cfg.normalizer, env, path)

        if cfg.load_ckpt is not None:
            ckpt_path = Path(cfg.load_ckpt)

            if ckpt_path.is_file():
                # Try to select stats from matching timestep.
                step = ckpt_path.stem.split('-')[-1]

                def ckpt_key(ckpt_file):
                    return (step in str(ckpt_file.stem).rsplit('-')[-1])
                stat_dir = ckpt_path.parent / '../stat/'
            else:
                # Find the latest checkpoint.
                ckpt_key = step_from_ckpt
                stat_dir = ckpt_path / '../stat'

            if stat_dir.is_dir():
                stat_ckpt = last_ckpt(stat_dir, key=ckpt_key)
                print(F'Also loading env stats from {stat_ckpt}')
                env.load(stat_ckpt,
                         strict=False)

                # we'll freeze env stats by default, if loading from ckpt.
                if freeze_env:
                    env.normalizer.eval()
            else:
                stat_ckpt = last_ckpt(cfg.load_ckpt + "_stat", key=ckpt_key)
                print(F'Also loading env stats from {stat_ckpt}')
                env.load(stat_ckpt,
                         strict=False)

    if cfg.use_unicorn:
        env = UnicornEmbedClouds(env, cfg.unicorn)

    if cfg.remove_cloud:
        pops = []
        for k in env.observation_space.keys():
            if ('cloud' in k) and ('embed_' not in k):
                pops.append(k)
        env = PopDict(env, pops)

    return cfg, env


def load_agent(cfg, env, path, writer):
    device = cfg.agent.device
    ic(cfg)

    # Override `dim_act` since it's included in the observation.
    cfg.net.state.state.dim_act = 0
    state_net = MLPStateEncoder.from_config(cfg.net.state)

    # Create policy/value networks.
    actor_net = NoOpPiNet(NoOpPiNet.Config(
        dim_act=cfg.net.policy.actor.dim_act,
        override_log_std=cfg.override_noop_policy_log_std,
        split_log_std=cfg.split_noop_policy_log_std,
        train_log_std=cfg.train_noop_policy_log_std
    )).to(device)
    value_net = NoOpVNet(NoOpVNet.Config(
        dim_val=env.num_rew)
    ).to(device)

    agent = PPO(
        cfg.agent,
        env,
        state_net,
        actor_net,
        value_net,
        path,
        writer,
        extra_nets=None
    ).to(device)

    if cfg.load_ckpt is not None:
        ckpt: str = last_ckpt(cfg.load_ckpt, key=step_from_ckpt)
        print(F'Load agent from {ckpt}')
        agent.load(last_ckpt(cfg.load_ckpt, key=step_from_ckpt),
                   strict=cfg.strict_load_ckpt)
    return agent


@with_wandb
def inner_main(cfg: Config, env, path):
    """ Same as main(), but after finalizing `cfg`.  """
    commit_hash = assert_committed(force_commit=cfg.force_commit)
    writer = SummaryWriter(path.tb_train)
    writer.add_text('meta/commit-hash',
                    str(commit_hash),
                    global_step=0)
    env.unwrap(target=AddTensorboardWriter).set_writer(writer)
    agent = load_agent(cfg, env, path, writer)

    ic(agent)

    try:
        th.cuda.empty_cache()
        with th.cuda.amp.autocast(enabled=cfg.use_amp):
            for step in agent.learn(name=F'{cfg.name}@{path.dir}'):

                # Periodically anneal log-std
                if (cfg.override_noop_policy_log_std and
                        cfg.split_noop_policy_log_std and
                        step % cfg.monitor.log_period == 0):
                    for ii, ls in enumerate(
                            agent.actor_net.log_std.mean(dim=-1)):
                        writer.add_scalar(F'log/{ii:02d}/logstd', ls,
                                          global_step=step)

                if (cfg.override_noop_policy_log_std
                    and (not cfg.train_noop_policy_log_std)
                    and isinstance(agent.actor_net, NoOpPiNet)
                        and (step % 1024) == 0):
                    agent.actor_net.log_std -= 0.000367
                # same annealing logic for non-subnet policy
                if (not cfg.hack_noop_policy and
                    not cfg.net.policy.actor.train_log_std
                        and (step % 1024) == 0):
                    agent.actor_net.log_std -= 0.000367

    finally:
        # Dump final checkpoints.
        agent.save(path.ckpt / 'last.ckpt')
        if hasattr(env, 'save'):
            env.save(path.stat / 'env-last.ckpt')

        # Dump curriculum state.
        if cfg.use_tune_domain_sampling:
            tune = env.unwrap(target=TuneDomainSampling)
            th.save(dcn(tune.spec._p_sel),
                    path.stat / 'sel-prob.pth')

        # Upload trained model/env stats to huggingface.
        if cfg.use_hfhub and (cfg.hf_repo_id is not None):
            upload_ckpt(
                cfg.hf_repo_id,
                (path.ckpt / 'last.ckpt'),
                cfg.name)
            upload_ckpt(
                cfg.hf_repo_id,
                (path.stat / 'env-last.ckpt'),
                cfg.name + '_stat')


@hydra_cli(config_path='../src/ham/data/cfg/', config_name='rss_train')
def main(cfg: Config):
    ic.configureOutput(includeContext=True)
    cfg = recursive_replace_map(cfg, {'finalize': True})

    path = setup(cfg)
    set_seed(cfg.env.seed)
    cfg, env = load_env(cfg, path)

    # Update `cfg` elements from `env`.
    obs_space = map_struct(
        env.observation_space,
        lambda src, _: src.shape,
        base_cls=spaces.Box,
        dict_cls=(Mapping, spaces.Dict)
    )

    # Apply state input blocklist.
    if cfg.state_net_blocklist is not None:
        for key in cfg.state_net_blocklist:
            obs_space.pop(key, None)

    # Parse action space.
    act_space = None
    if isinstance(env.action_space, spaces.Box):
        act_space = env.action_space.shape
    elif isinstance(env.action_space, spaces.Discrete):
        act_space = env.action_space.n
    elif isinstance(env.action_space, spaces.MultiDiscrete):
        # act_space = env.action_space.nvec
        act_space = [
            len(env.action_space.nvec),
            int(env.action_space.nvec[0])
        ]

    # Update config.
    cfg = replace(cfg, net=replace(cfg.net,
                                   obs_space=obs_space,
                                   act_space=act_space,
                                   ))

    # Start main entrypoint.
    return inner_main(cfg, env, path)


if __name__ == '__main__':
    main()
