#!/usr/bin/env python3

from dataclasses import dataclass
import torch as th
import pickle

from ham.env.env.iface import EnvIface
from ham.env.env.wrap.base import WrapperEnv
from ham.util.config import ConfigBase
from ham.util.path import ensure_directory
from ham.util.torch_util import dcn
from collections import defaultdict
from pathlib import Path
from ham.env.env.wrap.normalize_env import NormalizeEnv

class ExportActionTraj(WrapperEnv):

    @dataclass
    class Config(ConfigBase):
        export_dir: str = '/tmp/docker/ham-action_traj'
        save_unwrap: bool = True

    def __init__(self, env: EnvIface, cfg: Config):
        super().__init__(env)
        self.cfg = cfg
        self.__episode_id = th.arange(env.num_env,
                                      dtype=th.long,
                                      device=env.device)
        self.__count: int = 0
        self.__out_dir = ensure_directory(cfg.export_dir)

    def step(self, actions: th.Tensor):
        # == step normally...
        out = super().step(actions)
        if self.cfg.save_unwrap:
            norm = self.env.unwrap(target=NormalizeEnv)
            act = norm._wrap_act(actions)
        else:
            act = actions
        joint_pos = self.tensors['dof'][..., :, 0]
        root_tensor = self.tensors['root']
        obj_pose = root_tensor[self.scene.cur_props.ids.long(),
                               ..., :7]
        data = dict(
            actions= act,
            joint_pos=joint_pos,
            obj_pose = obj_pose,
            episode_id=self.__episode_id
        )
        data = {k: dcn(v) for (k, v) in data.items()}
        with open(self.__out_dir / F'{self.__count:04d}.pkl', 'wb') as fp:
            pickle.dump(data, fp)
        self.__count += 1
        # == increment episode id ==
        obs, rew, done, info = out
        # This episode-indexing scheme gives unique,
        # _but_ not necessarily continguous (though
        # ultimately dense) indices for all episodes.
        self.__episode_id += (self.num_env * done.long())
        return out
    
    def sort(self):
        sorted_data = defaultdict(lambda: defaultdict(list))
        for data_file in list(sorted(self.__out_dir.rglob('*.pkl'))):
            try:
                with open(data_file, 'rb') as fp:
                    data = pickle.load(fp)
                episode_id = data['episode_id']
                num_env: int = len(episode_id)
                for i in range(num_env):
                    for k, v in data.items():
                        sorted_data[episode_id[i]][k].append(
                            (v[i] if len(v) > 0 else v))
            except:
                continue
        sorted_data = [sorted_data[k] for k in sorted(sorted_data.keys())]
        p = ensure_directory(self.__out_dir/'sorted')
        for i in range(len(sorted_data)):
            with open(p/f'{i}.pkl', 'wb') as fp:
                pickle.dump(sorted_data[i], fp)
