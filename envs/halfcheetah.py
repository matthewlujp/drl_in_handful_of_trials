from .pybulletenvs import make
import gym
import torch
import numpy as np


class SurrogateHalfCheetah:
    OBSERVATION_SIZE = 27
    ACTION_SIZE = 6

    def __init__(self):
        self._env = make("HalfCheetahBulletEnv")
        self._prev_pos = None

    def step(self, a):
        _next_obs, r, done, info = self._env.step(a)
        pos = self._env.robot_body.current_position()
        next_obs = np.concatenate([pos[:1] - self._prev_pos[:1], _next_obs])
        return next_obs, r, done, info

    def reset(self):
        _obs = self._env.reset()
        pos = self._env.robot_body.current_position()
        self._prev_pos = pos
        return np.concatenate([pos[:1] - self._prev_pos[:1], _obs])

    def render(self, mode='human'):
        return self._env.render(mode)

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        box = self._env.observation_space
        new_box = gym.spaces.Box(
            low = np.concatenate([np.array([-np.inf]), self._env.observation_space.low]),
            high = np.concatenate([np.array([np.inf]), self._env.observation_space.high]),
        )
        return new_box


class EnvironmentManager:
    @classmethod
    def new_env(cls):
        return SurrogateHalfCheetah()

    @staticmethod
    def cost_func(states_seq, actions_seq):
        """
        Params
        states_seq: torch.tensor([seq_len, batch, state])
        actions_seq: torch.tensor([seq_len, batch, action])

        Return
        costs: torch.tensor([batch])
        """
        seq_len, batch_size, _ = states_seq.size()
        state_costs = torch.zeros(batch_size)
        action_costs = torch.zeros(batch_size)

        # for states in states_seq:
        #     state_costs += (1. - states[:, 0])**2
        state_costs -= states_seq[-1, :, 0]

        for actions in actions_seq:
            action_costs += 0.001 * (actions**2).sum(dim=1)

        return state_costs + action_costs

