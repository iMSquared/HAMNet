#!/usr/bin/env python3

from ham.env.env.iface import EnvIface
from ham.env.env.wrap.base import WrapperEnv

from typing import Optional
from dataclasses import dataclass
from ham.util.config import ConfigBase

import torch as th
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter_add


class MonitorEnv(WrapperEnv):
    """
    Monitor Env. logs the following statistics:

    Average:
    > [approx] avg. episode length.
    > reward.
    > [approx] discounted returns.
    > [approx] undiscounted returns.

    Total:
    > number of interactions (mul. by num_env)
    > number of episodes.
    > number of successful episodes.

    WARN(ycho): to avoid annoyingnesses with empty tensors,
    we do not apply masking.
    """

    @dataclass
    class Config(ConfigBase):
        gamma: float = 0.99
        log_period: int = 1024
        verbose: bool = True

    def __init__(self,
                 cfg: Config,
                 env: EnvIface):
        super().__init__(env)
        self.cfg = cfg

        N: int = env.num_env
        R: int = env.num_rew
        T: int = env.num_type

        self.returns = th.zeros(N, R, dtype=th.float,
                                device=env.device)
        self.discounted_returns = th.zeros(N, R, dtype=th.float,
                                           device=env.device)
        self.cumulative_reward = th.zeros(N, R, dtype=th.float,
                                          device=env.device)
        self.discounted_cumulative_reward = th.zeros(N, R, dtype=th.float,
                                                     device=env.device)
        self.true_cumulative_return = th.zeros(N, R, dtype=th.float,
                                               device=env.device)
        self.episode_step = th.zeros(N, dtype=th.int32,
                                     device=env.device)

        # Type-wise statistics
        # TODO(ycho): type<->reward-wise statistics?
        # or is it too much??
        self.episode_count = th.zeros(T,
                                      dtype=th.int32,
                                      device=self.device)
        self.last_episode_count = th.zeros(T,
                                           dtype=th.int32,
                                           device=self.device)
        self.success_count = th.zeros(T,
                                      dtype=th.int32,
                                      device=self.device)
        self.last_success_count = th.zeros(T,
                                           dtype=th.int32,
                                           device=self.device)

        self.step_count: int = 0
        self.reward_log = None

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def setup(self):
        return self.env.setup()

    def reset_indexed(self, *args, **kwds) -> th.Tensor:
        return self.env.reset_indexed(*args, **kwds)

    def reset(self) -> th.Tensor:
        return self.env.reset()

    def _add_scalar(self, k, v, step: Optional[int] = None):
        if step is None:
            step = self.step_count
        if self.writer is not None:
            return self.writer.add_scalar(k, v, global_step=step)
        elif self.cfg.verbose:
            print(F'{k}={v}')

    def _log(self, env_type):
        self._add_scalar('env/eplen',
                         self.episode_step.float().mean().item())
        self._add_scalar('env/num_interactions',
                         self.step_count * self.num_env)
        self._add_scalar('env/num_steps', self.step_count)

        # Log returns
        for i in range(self.num_rew):
            self._add_scalar(F'env/return/{i:02d}',
                             self.returns[..., i].mean().item())
            self._add_scalar(F'env/discounted_return/{i:02d}',
                             self.discounted_returns[..., i].mean().item())

        # Typewise logs
        total_episode_count = self.episode_count.sum().item()
        total_success_count = self.success_count.sum().item()
        last_total_episode_count = self.last_episode_count.sum().item()
        last_total_success_count = self.last_success_count.sum().item()
        self._add_scalar('env/num_episodes', total_episode_count)
        self._add_scalar('env/num_success', total_success_count)
        total_suc_rate = (0.0 if total_episode_count <= 0
                          else total_success_count / total_episode_count)
        self._add_scalar('env/suc_rate', total_suc_rate)

        total_d_suc = (total_success_count - last_total_success_count)
        total_d_eps = (total_episode_count - last_total_episode_count)
        total_cur_suc_rate = (
            0.0 if total_d_eps <= 0 else total_d_suc /
            total_d_eps)
        self._add_scalar('env/cur_suc_rate', total_cur_suc_rate)

        # Add episode returns.
        for i in range(self.num_rew):
            self._add_scalar(
                F'env/avg_episode_return/{i:02d}',
                (self.cumulative_reward[..., i].sum() / total_d_eps).item())
            total_steps = self.cfg.log_period * self.num_env
            self._add_scalar(
                F'env/avg_reward/{i:02d}',
                (self.cumulative_reward[..., i].sum() / total_steps).item())
            self._add_scalar(
                F'env/avg_episode_discounted_return/{i:02d}',
                (self.discounted_cumulative_reward[..., i].sum() / total_d_eps).item())
            self._add_scalar(
                F'env/avg_true_return/{i:02d}',
                (self.true_cumulative_return[..., i].sum() / total_d_eps).item())

        # Add individual reward logs.
        # TODO(ycho): is this still necessary?
        # Now that we have vectorized rewards... hmm
        for k in self.reward_names:
            self._add_scalar(
                f'rew/avg_reward_{k}',
                (self.reward_log[k].sum() / total_steps).item())
            self._add_scalar(
                f'rew/avg_episode_reward_{k}',
                (self.reward_log[k].sum() / total_d_eps).item())
            self._add_scalar(
                f'rew/avg_discounted_reward_{k}',
                (self.reward_log[f'discounted_{k}'].sum() / total_d_eps).item())
            self.reward_log[k].fill_(0)
            self.reward_log[f'discounted_{k}'].fill_(0)

        # Type-wise statistics
        acc_suc_rate = self.success_count / self.episode_count
        d_suc = (self.success_count - self.last_success_count)
        d_eps = (self.episode_count - self.last_episode_count)
        sr = (d_suc / d_eps)
        cur_suc_rate = th.where(d_eps <= 0, 0 * sr, sr)
        cum_rew = scatter_add(self.cumulative_reward, env_type, dim=0)
        avg_eps_rew = cum_rew / d_eps
        avg_rew = cum_rew / (self.cfg.log_period * self.num_env / self.num_type)
        discounted_cumulative_reward = scatter_add(
            self.discounted_cumulative_reward, env_type, dim=0)
        avg_episode_discounted_return = discounted_cumulative_reward / d_eps
        true_return = scatter_add(
            self.true_cumulative_return,
            env_type, dim=0) / d_eps

        for i in range(self.num_type):
            prefix = F'env/{i:02d}'
            self._add_scalar(
                F'{prefix}/num_episodes',
                self.episode_count[i].item())
            self._add_scalar(
                F'{prefix}/num_success',
                self.success_count[i].item())
            self._add_scalar(F'{prefix}/suc_rate', acc_suc_rate[i].item())
            self._add_scalar(F'{prefix}/cur_suc_rate', cur_suc_rate[i].item())

            # Add episode returns.
            for j in range(self.num_rew):
                self._add_scalar(
                    F'{prefix}/avg_episode_return/{j:02d}',
                    avg_eps_rew[i, j].item())
                self._add_scalar(
                    F'{prefix}/avg_reward/{j:02d}', avg_rew[i, j].item())
                self._add_scalar(
                    F'{prefix}/avg_episode_discounted_return/{j:02d}',
                    avg_eps_rew[i, j].item())
                self._add_scalar(
                    F'{prefix}/avg_episode_discounted_return/{j:02d}',
                    avg_episode_discounted_return[i, j].item())
                self._add_scalar(
                    F'{prefix}/avg_true_return/{j:02d}',
                    true_return[i, j].item())

        # Reset counts and statistics.
        self.last_success_count[:] = self.success_count
        self.last_episode_count[:] = self.episode_count

        self.cumulative_reward.fill_(0)
        self.discounted_cumulative_reward.fill_(0)
        self.true_cumulative_return.fill_(0)

    def step(self, action: th.Tensor):
        cfg = self.cfg
        obs, rew, done, info = super().step(action)

        with th.no_grad():
            resetf = (~done).float()

            # Update episodic statistics.
            self.cumulative_reward.add_(rew)
            self.discounted_cumulative_reward.mul_(cfg.gamma).add_(rew)
            self.true_cumulative_return.add_(
                rew.mul(th.pow(cfg.gamma, self.episode_step[..., None]))
            )

            # update returns.
            self.returns.add_(rew)
            self.discounted_returns.mul_(cfg.gamma).add_(rew)

            # update episode steps.
            self.episode_step.add_(1)

            # track total number of episodes.
            self.episode_count += scatter_add(
                done.to(self.episode_count.dtype),
                info['env_type'], dim=0)
            if 'success' in info:
                self.success_count += scatter_add(
                    info['success'].to(
                        self.success_count.dtype),
                    info['env_type'], dim=0)
            self.step_count += 1

            if self.reward_log is None:
                self.reward_log = {}
                for k, v in info['reward'].items():
                    self.reward_log[k] = v.clone()
                    self.reward_log[f'discounted_{k}'] = v.clone()
                self.reward_names = list(info['reward'].keys())
            else:
                for k, v in info['reward'].items():
                    self.reward_log[k].add_(v)
                    self.reward_log[f'discounted_{k}'].add_(
                        v.mul(th.pow(cfg.gamma, self.episode_step[..., None]))
                    )

            if (self.step_count % cfg.log_period == 0):
                self._log(info['env_type'])

            # Reset cumulative statistics.
            self.returns.mul_(resetf[..., None])
            self.discounted_returns.mul_(resetf[..., None])
            self.episode_step[done] = 0

            # consistent = (done == self.env.buffers['done']).all()
            # if not consistent.item():
            #    raise ValueError('incons')
            # print(self.episode_step.float().max(),
            #       self.episode_step.float().mean(),
            #       self.env.buffers['step'].float().max(),
            #       self.env.buffers['step'].float().mean(),
            #       )

        return obs, rew, done.clone(), info

    def __getattr__(self, attr):
        """ delegate missing attributes to self.env. """
        return getattr(self.env, attr)
