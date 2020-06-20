import gym
import torch
import numpy as np


class EnvironmentManager:
    MAX_TORQUE = 2.

    @classmethod
    def new_env(cls):
        return gym.make("Pendulum-v0")

    @staticmethod
    def cost_func(states_seq, actions_seq):
        """
        Params
        states_seq: torch.tensor([seq_len, batch, (cos(theta), sin(theta), theta_dot)])
        actions_seq: torch.tensor([seq_len, batch, (torque, )])

        Return
        costs: torch.tensor([batch])
        """
        seq_len, batch_size, _ = states_seq.size()
        state_costs = torch.zeros(batch_size)
        action_costs = torch.zeros(batch_size)

        for states in states_seq:
            state_costs += (1. - states[:, 0])**2

        for actions in actions_seq:
            action_costs += 0.001 * actions[:, 0]**2


        return state_costs

