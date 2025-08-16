#!/usr/bin/env python3

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
import pickle
from pathlib import Path

from typing import Optional, Mapping
from dataclasses import dataclass, replace
from ham.models.common import map_struct
from ham.env.env.wrap.record_viewer import RecordViewer
from ham.env.env.wrap.log_episodes import LogEpisodes
from ham.env.env.wrap.reset_ig_camera import reset_ig_camera
from ham.env.env.wrap.draw_clouds import DrawClouds
from ham.env.env.wrap.draw_mesh_edge import DrawMeshEdge
import torch as th
import numpy as np
import einops
from ham.env.arm_env import OBS_BOUND_MAP
from ham.util.torch_util import dcn
from ham.util.hydra_cli import hydra_cli
from ham.util.config import recursive_replace_map
from ham.env.util import (
    set_seed,
    draw_sphere,
    draw_cloud_with_sphere,
    draw_patch_with_cvxhull,
)
from ham.env.env.wrap.normalize_env import NormalizeEnv
from ham.env.env.wrap.draw_patch_attn import DrawPatchAttention
from ham.env.env.wrap.export_render_data import ExportRenderData
from ham.env.env.wrap.export_action_traj import ExportActionTraj
from ham.env.env.wrap.draw_force_sensor import DrawForceSensor
from ham.models.common import CountDormantNeurons, map_tensor
from ham.util.path import ensure_directory
from ham.train.ckpt import save_ckpt
from omegaconf import OmegaConf

from icecream import ic
from gym import spaces

from train import (
    Config as TrainConfig,
    load_agent,
    load_env)

import cv2
import matplotlib
matplotlib.rcParams['backend'] = 'Agg'


@dataclass
class Config(TrainConfig):
    sample_action: bool = True

    draw_debug_lines: bool = True

    draw_patch_attn: bool = False
    sync_frame_time: bool = False

    draw_mesh_edge: bool = False
    draw_force_sensor: bool = False

    test_steps: int = 16384


@hydra_cli(config_path='../src/ham/data/cfg/', config_name='rss_show')
def main(cfg: Config):
    ic.configureOutput(includeContext=True)
    cfg = recursive_replace_map(cfg, {'finalize': True})
    ic(cfg)

    # Maybe it's related to jit
    if cfg.global_device is not None:
        th.cuda.set_device(cfg.global_device)
    path, writer = None, None
    _ = set_seed(cfg.env.seed)
    cfg, env = load_env(cfg, path, freeze_env=True,
                        check_viewer=False,
                        draw_workspace=True
                        )

    if cfg.draw_mesh_edge:
        env = DrawMeshEdge(env, True)

    if cfg.draw_force_sensor:
        env = DrawForceSensor(env)

    # Update cfg elements from `env`.
    obs_space = map_struct(
        env.observation_space,
        lambda src, _: src.shape,
        base_cls=spaces.Box,
        dict_cls=(Mapping, spaces.Dict)
    )
    if cfg.state_net_blocklist is not None:
        for key in cfg.state_net_blocklist:
            obs_space.pop(key, None)
    dim_act = (
        env.action_space.shape if isinstance(
            env.action_space,
            spaces.Box) else env.action_space.n)
    cfg = replace(cfg, net=replace(cfg.net,
                                   obs_space=obs_space,
                                   act_space=dim_act
                                   ))
    agent = load_agent(cfg, env, None, None)

    if cfg.draw_patch_attn:
        draw_env = env.unwrap(target=DrawPatchAttention)
        if isinstance(draw_env, DrawPatchAttention):
            draw_env.patch_attn_fn.register(
                agent.state_net.feature_encoders['cloud']
            )
        else:
            raise ValueError('failed to unwrap')

    reset_ig_camera(env,
                    offset=(1.0, 0.0, 0.5)
                    )
    agent.eval()
    ic(agent)

    try:
        for (act, obs, rew, done, info) in agent.test(
                sample=cfg.sample_action, steps=cfg.test_steps):
            if cfg.sync_frame_time:
                env.gym.sync_frame_time(env.sim)
    finally:
        pass


if __name__ == '__main__':
    main()
