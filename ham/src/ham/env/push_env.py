#!/usr/bin/env python3


from typing import Iterable, Optional
from dataclasses import dataclass
import numpy as np

from isaacgym import gymapi

from ham.env.scene.tabletop_with_object_scene_v3 import TableTopWithObjectScene
from ham.env.scene.tabletop_with_multi_object_scene import (
    TableTopWithMultiObjectScene
)
from ham.env.robot.franka import Franka
from ham.env.task.push_with_hand_task import PushWithHandTask
from ham.env.env.base import EnvBase
from ham.util.torch_util import dcn

import nvtx


class PushEnv(EnvBase):
    """
    Tabletop environment.
    """

    @dataclass
    class Config(EnvBase.Config):
        # SCENES
        single_object_scene: TableTopWithObjectScene.Config = TableTopWithObjectScene.Config()
        multi_object_scene: TableTopWithMultiObjectScene.Config = TableTopWithMultiObjectScene.Config()
        which_scene: str = 'single_object'

        # ROBOTS
        franka: Franka.Config = Franka.Config()
        which_robot: str = 'virtual'

        # TASKS
        task: PushWithHandTask.Config = PushWithHandTask.Config()

        # DEBUG VISUALS
        draw_task_goal: bool = False
        draw_obj_pos_2d: bool = False
        draw_obj_vel: bool = False

        render: bool = False

    def __init__(self, cfg: Config, writer=None, task_cls=None):
        if cfg.which_scene == 'multi_object':
            scene = TableTopWithMultiObjectScene(cfg.multi_object_scene)
            self.scene_cfg = cfg.multi_object_scene
        else:
            scene = TableTopWithObjectScene(cfg.single_object_scene)
            self.scene_cfg = cfg.single_object_scene

        self.robot_cfg = None
        if cfg.which_robot == 'franka':
            self.robot_cfg = cfg.franka
            robot = Franka(self.robot_cfg)
        else:
            raise KeyError(F'Unknown robot = {cfg.which_robot}')

        # FIXME(ycho): brittle multiplexing between
        # pushtask and PushWithCubeTask!!!
        if task_cls is None:
            task = PushWithHandTask(cfg.task, writer=writer)
        else:
            task = task_cls(cfg.task, writer=writer)
        super().__init__(cfg,
                         scene, robot, task)
        self.configure()
        self.__path = []
        self._actions_path = []

    @property
    def action_space(self):
        return self.robot.action_space

    @property
    def observation_space(self):
        return None

    def _draw_obj_pos_2d(self):
        cfg = self.cfg
        gym = self.gym

        obj_ids = self.scene.cur_props.ids.to(
            self.cfg.th_device)
        pos_3d = dcn(self.tensors['root'][
            obj_ids.long(), :3])

        for i in range(cfg.num_env):
            verts = np.zeros(shape=(2, 3), dtype=np.float32)
            verts[:, :3] = pos_3d[i]
            verts[0, 2] += - 0.2
            verts[1, 2] += + 0.2

            gym.add_lines(self.viewer,
                          self.envs[i],
                          1,
                          verts[None],
                          np.asarray([0, 0, 1], dtype=np.float32))

    def _draw_task_goal(self):
        # Just for debugging
        cfg = self.cfg
        gym = self.gym
        if self.viewer is None:
            return

        goals = dcn(self.task.goal)

        circle_verts = np.zeros(shape=(1, 128, 3),
                                dtype=np.float32)
        angles = np.linspace(-np.pi, np.pi, num=128)[None, :]
        circle_verts[..., 0] = np.cos(angles)
        circle_verts[..., 1] = np.sin(angles)
        circle_verts = circle_verts * dcn(
            # self.task.goal_radius_samples
            self.scene.data['goal_radius']
        )[:, None, None]

        for i in range(cfg.num_env):
            verts = np.zeros(shape=(2, 3), dtype=np.float32)
            verts[:, :3] = goals[i, ..., :3]
            verts[0, 2] -= 0.3
            verts[1, 2] += 0.3

            gym.add_lines(self.viewer,
                          self.envs[i],
                          1,
                          verts[None],
                          np.asarray([1, 0, 0], dtype=np.float32))

            # circle indicating goal region
            points = np.zeros(shape=(128, 3), dtype=np.float32)
            points[:, :3] = goals[None, i, ..., :3] + circle_verts[i]
            verts = np.stack(
                [points[:-1], points[1:]],
                axis=-2)
            colors = np.zeros_like(verts[..., 0, :])
            colors[..., 0] = 1.0
            gym.add_lines(self.viewer,
                          self.envs[i],
                          verts.shape[0],
                          verts,
                          colors
                          )

    @nvtx.annotate('PushEnv.step()', color="cyan")
    def step(self, *args, **kwds):
        with nvtx.annotate('PushEnv.step.A()'):
            out = super().step(*args, **kwds)

        with nvtx.annotate('PushEnv.step.B()'):
            # FIXME(ycho): super fragile fix
            if (self.viewer is not None) or hasattr(self.gym, 'lines'):
                self.gym.clear_lines(self.viewer)

            if self.viewer is not None:
                if self.cfg.draw_task_goal:
                   self._draw_task_goal()
                if self.cfg.draw_obj_pos_2d:
                    self._draw_obj_pos_2d()

        with nvtx.annotate('PushEnv.step.C()'):
            return out

    def create_assets(self):
        self.cfg
        gym = self.gym
        sim = self.sim
        outputs = super().create_assets()
        return outputs

    def create_envs(self):
        # Create all the default envs
        outputs = super().create_envs()
        cfg = self.cfg
        gym = self.gym
        sim = self.sim
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        return outputs

    def setup(self):
        """
        * load assets.
        * allocate buffers related to {scene, robot, task}.
        """
        return super().setup()

    @nvtx.annotate('PushEnv.reset_indexed()', color="orange")
    def reset_indexed(self, indices: Optional[Iterable[int]] = None):
        cfg = self.cfg
        return super().reset_indexed(indices)

    def reset(self):
        return super().reset()

    def apply_actions(self, actions):
        cfg = self.cfg
        return self.robot.apply_actions(
            self.gym, self.sim, self,
            actions, done=self.buffers['done'])

    def compute_feedback(self, *args, **kwds):
        return super().compute_feedback(*args, **kwds)

    @nvtx.annotate('PushEnv.compute_observations()', color="blue")
    def compute_observations(self):
        return {}
