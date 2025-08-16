#!/usr/bin/env python3

from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import torch as th
import pickle

from ham.env.env.iface import EnvIface
from ham.env.env.wrap.base import WrapperEnv
from ham.util.config import ConfigBase
from ham.util.path import ensure_directory
from ham.util.torch_util import dcn


class ExportRenderData(WrapperEnv):

    @dataclass
    class Config(ConfigBase):
        export_dir: str = '/tmp/docker/ham-render-data'
        log_routing: bool = False

    def __init__(self, env: EnvIface, cfg: Config):
        super().__init__(env)
        self.cfg = cfg
        self._episode_id = th.arange(env.num_env,
                                     dtype=th.long,
                                     device=env.device)
        self._count: int = -1
        self._out_dir = ensure_directory(cfg.export_dir)

    def _query_and_save(self, save=True):

        # [1] Environment;
        # box dims
        box_dim = self.scene.data['table_geom']
        # box poses
        root_tensor = self.tensors['root']
        barrier_ids = self.scene.ctx['barrier_id']
        # num_env x num_body x 7
        box_pose = root_tensor[barrier_ids, ..., :7]

        # [2] Object;
        obj_pose = root_tensor[self.scene.data['obj_id'].long(),
                               ..., :7]
        obj_scale = self.scene.data['rel_scale']

        # [3] Robot;
        joint_pos = self.tensors['dof'][..., :, 0]

        robot_pose = root_tensor[self.robot.indices.long(),
                                 ...,
                                 :7]

        # [3] robot hand
        body_tensors = self.tensors['body']
        hand_indices = self.robot.ee_body_indices.long()
        hand_pose = body_tensors[hand_indices, :]

        # [5] Bundle+Export;
        data = dict(
            # object_file = <unknown>
            name=self.scene.data['obj_name'],#cur_props.names,
            box_dim=box_dim,
            box_pose=box_pose,
            goal=self.task.goal,
            obj_pose=obj_pose,
            obj_scale=obj_scale,
            robot_pose=robot_pose,
            joint_pos=joint_pos,
            episode_id=self._episode_id,
            hand_pose=hand_pose
        )

        if self.cfg.log_routing:
            data['routing'] = dcn(self._net._weights)

        data = {k: dcn(v) for (k, v) in data.items()}
        self._count += 1
        if save:
            self._save(data)
        else:
            return data

    def register(self, net):
        self._net = net

    def _save(self, data):
        with open(self._out_dir / F'{self._count:04d}.pkl', 'wb') as fp:
            pickle.dump(data, fp)

    def step(self, actions: th.Tensor):
        # == step normally...
        out = super().step(actions)
        self._query_and_save()
        # == increment episode id ==
        obs, rew, done, info = out
        # This episode-indexing scheme gives unique,
        # _but_ not necessarily continguous (though
        # ultimately dense) indices for all episodes.
        self._episode_id += (self.num_env * done.long())
        return out


def rearrange_data_by_episode(data_path: str,
                              save_path: str,
                              div_with_suc: bool = False):
    sorted_data = defaultdict(lambda: defaultdict(list))
    save_path = Path(save_path)
    for data_file in list(sorted(Path(data_path).rglob('*.pkl'))):
        with open(data_file, 'rb') as fp:
            data = pickle.load(fp)
        # print(data)
        episode_id = data['episode_id']
        # print(episode_id)
        num_env: int = len(episode_id)
        for i in range(num_env):
            for k, v in data.items():
                if v is not None:
                    sorted_data[episode_id[i]][k].append(
                        (v[i] if len(v) > 0 else v))
    sorted_data = [sorted_data[k] for k in sorted(sorted_data.keys())]
    ensure_directory(save_path)
    for c, data in enumerate(sorted_data):
        with open(save_path / F'{c:04d}.pkl', 'wb') as fp:
            pickle.dump(data, fp)
