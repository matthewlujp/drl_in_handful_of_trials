""" MPC controller based on "Deep Learning with Handful of Trials."
https://arxiv.org/abs/1805.12114
"""
from optimizers import CEMOptimizer
from dynamics_model import DynamicsModel

from gym.spaces import Discrete, MultiDiscrete, Box 
import torch
import numpy as np
import pickle
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import clamp



def space_extractor(space):
    """
    Params
    space: state_space or action_space

    Return
    dimension(int), lower_bounds(np.ndarray), upper_bounds(np.ndarray)
    """
    if type(space) == Discrete:
        dim = 1
        upper_bounds = np.array([space.n - 1])
        lower_bounds = np.array([0])
    elif type(space) == MultiDiscrete:
        dim = space.shape[0]
        upper_bounds = space.nvec - 1
        lower_bounds = np.zeros_like(space.nvec)
    elif type(space) == Box:
        dim = space.shape[0]
        upper_bounds = space.high
        lower_bounds = space.low
    else:
        raise ValueError

    return dim, lower_bounds, upper_bounds



class MPC:
    def __init__(
        self, dynamics_model, state_space, action_space,
        ensemble_size, particle_size, cem_sample_size, cem_elite_rate, cem_max_iter, plan_horizon,
        cost_func=None):
        """
        Params
        observation_space: (Discrete, MultiDiscrete, Box)
        action_space: (Discrete, MultiDiscrete, Box)
        state_cost_func: A function which takes a state vector and return its cost.
                         The first element of the input vector is reward prediction, hence, returning it is sufficient normally.
        action_cost_func: A function which takes an action vector and return its cost.
        """
        assert isinstance(state_space, (Discrete, MultiDiscrete, Box))
        assert isinstance(action_space, (Discrete, MultiDiscrete, Box))
        assert particle_size % ensemble_size == 0

        self.state_space = state_space
        self.state_dim, self.state_lower_bounds, self.state_upper_bounds = space_extractor(self.state_space)

        self.action_space = action_space
        self.action_dim, self.action_lower_bounds, self.action_upper_bounds = space_extractor(self.action_space)

        self.ensemble_size = ensemble_size
        self.particle_size = particle_size
        self.cem_sample_size = cem_sample_size
        self.cem_elite_size = int(self.cem_sample_size * cem_elite_rate)
        self.plan_horizon = plan_horizon
        self.cem_max_iter = cem_max_iter

        self._cost_func = cost_func

        self.has_model_trained = False

        self.dynamics_model = dynamics_model
        self.cem_optimizer = CEMOptimizer(
            self.plan_horizon * self.action_dim, self.cem_max_iter, self.cem_sample_size, self.cem_elite_size,
            np.tile(self.action_upper_bounds, self.plan_horizon),
            np.tile(self.action_lower_bounds, self.plan_horizon))

    def predict(self, inputs):
        self.dynamics_model.eval()
        mean, std = self.dynamics_model(inputs)
        return mean.detach().to('cpu'), std.detach().to('cpu')

    def act(self, obs, prev_sol=None):
        """
        Return
        action: (np.array [action_dim])
        """
        if not self.has_model_trained:
            # return random actions
            if isinstance(self.action_space, (Discrete, MultiDiscrete)):
                return np.random.randint(self.action_upper_bounds[0] + 1, size=(self.action_dim)), None
            elif isinstance(self.action_space, Box):
                return np.random.uniform(self.action_lower_bounds, self.action_upper_bounds), None
            return np.zeros(self.action_dim)

        if prev_sol is None:
            init_mean = np.ones(self.plan_horizon * self.action_dim) * np.tile((self.action_upper_bounds + self.action_lower_bounds) / 2., self.plan_horizon)
        else:
            assert prev_sol.shape == (self.plan_horizon-1, self.action_dim)
            init_mean = np.concatenate((
                prev_sol.reshape(-1),
                np.ones(self.action_dim) * (self.action_upper_bounds + self.action_lower_bounds) / 2.))
            
        init_var = np.ones_like(init_mean) * 0.5
        evaluator = self.get_action_sequence_cost_evaluator(obs)
        sol = self.cem_optimizer.obtain_solution(init_mean, init_var, evaluator)
        sol = sol.reshape(self.plan_horizon, self.action_dim)
        # assert sol.shape == (self.plan_horizon, self.action_dim)
        return self._preproc_action(sol[0]), sol[1:]

    def _preproc_action(self, actions):
        """Preprocess action according to action_space class.
        e.g., discretization.

        Params
        actions: (np.array [?, ?, ..., action_dim])
        """
        # assert actions.shape[-1] == self.action_dim, "action dim expected {}, got {}".format(self.action_dim, actions.shape[-1])
        
        if isinstance(self.action_space, (Discrete, MultiDiscrete)):
            actions = np.round(actions).astype(int)
        actions = np.clip(actions, self.action_lower_bounds, self.action_upper_bounds)  
        return actions

    def _postproc_state(self, states):
        """Preprocess state according to state_space class.
        
        Params
        states: (torch.tensor [?, ?, ..., state_dim])
        """
        # assert isinstance(states, torch.Tensor)
        # assert states.shape[-1] == self.state_dim, "state dim expected {}, got {}".format(self.state_dim, states.shape[-1])
        
        if isinstance(self.state_space, (Discrete, MultiDiscrete)):
            states = torch.round(states)
        states = clamp(states, torch.from_numpy(self.state_lower_bounds), torch.from_numpy(self.state_upper_bounds))
        return states

    def get_action_sequence_cost_evaluator(self, current_state):
        """Return a cost evaluator of action which run steps from current state based on the internal dynamics model.
        Return
        func(action_sequence_samples: torch.tensor [samples_num, planning_horizon]): a function to 
        """
        def cost_evaluator(action_sequence_samples):
            """
            Params
            action_sequence_samples: (np.array [samples_num, seq_len x action_dim]) Continuous unbounded action sequence.

            Return
            costs: (torch.tensor [samples_num])
            """
            self.dynamics_model.eval()

            costs = torch.zeros(self.cem_sample_size, self.particle_size)

            current_states = torch.tensor(current_state, dtype=torch.float32).repeat((self.cem_sample_size, 1))  # -> [samples_size, state_dim]
            current_states = current_states.view(self.cem_sample_size, 1, self.state_dim).repeat(1, self.particle_size, 1)  # -> [samples_size, particle_size, state_dim]

            action_sequence_samples = np.transpose(action_sequence_samples.reshape((self.cem_sample_size, self.plan_horizon, self.action_dim)), (1,0,2))
            # assert action_sequence_samples.shape == (self.plan_horizon, self.cem_sample_size, self.action_dim), action_sequence_samples.shape

            states_seq = [current_states]
            actions_seq = []
            for actions in action_sequence_samples:
                actions_t = torch.from_numpy(self._preproc_action(actions)).type(torch.float32)  # [sample_size, action_dim]
                actions_t = actions_t.view(self.cem_sample_size, 1, self.action_dim).repeat(1, self.particle_size, 1)
                actions_seq.append(actions_t)
                model_in = torch.cat((current_states, actions_t), dim=2)  # [samples_size, particle_size, state_dim + action_dim]
                expended_model_in = self._expand(model_in)  # -> [ensemble_size, samples_size x particle_size // ensemble_size, action_dim]

                mean, logvar = self.dynamics_model(expended_model_in)

                _predictions = mean + torch.randn_like(mean) * torch.exp(0.5*logvar)
                delta_predictions = self._flatten(_predictions)  # [samples_size, particle_size, state_dim]
                next_states_predicted = self._postproc_state(current_states + delta_predictions)

                current_states = next_states_predicted
                states_seq.append(current_states)

            states_seq = torch.stack(states_seq)
            # assert states_seq.size() == torch.Size([self.plan_horizon+1, self.cem_sample_size, self.particle_size, self.state_dim]), states_seq.size()
            actions_seq = torch.stack(actions_seq)
            # assert actions_seq.size() == torch.Size([self.plan_horizon, self.cem_sample_size, self.particle_size, self.action_dim]), actions_seq.size()

            costs = self._calc_cost(states_seq, actions_seq)
            # assert costs.size() == torch.Size([self.cem_sample_size, self.particle_size])
            return costs.mean(1).detach().cpu().numpy()

        return cost_evaluator
            
    def _calc_cost(self, states_seq, actions_seq):
        """
        Params
        states_seq: (torch.tensor [action_seq+1, samples_size, particle_size, state_dim])
        actions_seq: (torch.tensor [action_seq, samples_size, particle_size, action_dim])

        Return
        costs: (torch.tensor [samples_size, particle_size])
        """
        assert states_seq.size() == torch.Size([self.plan_horizon+1, self.cem_sample_size, self.particle_size, self.state_dim]), states_seq.size()
        assert actions_seq.size() == torch.Size([self.plan_horizon, self.cem_sample_size, self.particle_size, self.action_dim]), actions_seq.size()

        costs = self._cost_func(states_seq.view(self.plan_horizon+1, -1, self.state_dim), actions_seq.view(self.plan_horizon, -1, self.action_dim))
        costs = costs.view(self.cem_sample_size, self.particle_size)
        # assert costs.size() == torch.Size([self.cem_sample_size, self.particle_size]), costs.size()
        return costs

    def _expand(self, x):
        """Expend input to feed into ensemble bootstrap networks.
        [samples_size, particle_size, ?] -> [ensemble_size, samples_size x particle_size // ensemble_size, ?]
        """
        # assert len(x.size()) == 3, "x size {}".format(x.size())
        # assert x.size(0) == self.cem_sample_size
        # assert x.size(1) == self.particle_size
        feature_dim = x.size(2)
        return x.view(self.cem_sample_size, self.ensemble_size, self.particle_size//self.ensemble_size, feature_dim)\
            .transpose(0,1).contiguous()\
            .view(self.ensemble_size, self.cem_sample_size*self.particle_size//self.ensemble_size, feature_dim)
        
    def _flatten(self, x):
        """Flatten ensemble bootstrap form to normal form.
        [ensemble_size, samples_size x particle_size // ensemble_size, ?] -> [samples_size, particle_size, ?]
        """
        # assert len(x.size()) == 3, "x size {}".format(x.size())
        # assert x.size(0) == self.ensemble_size
        # assert x.size(1) == self.cem_sample_size * self.particle_size / self.ensemble_size
        feature_dim = x.size(2)
        return x.view(self.ensemble_size, self.cem_sample_size, self.particle_size//self.ensemble_size, feature_dim)\
            .transpose(0, 1).contiguous().view(self.cem_sample_size, self.particle_size, feature_dim)
        

