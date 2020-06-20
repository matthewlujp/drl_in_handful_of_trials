import torch



def clamp(tensor, min_tensor, max_tensor):
    return torch.min(torch.max(tensor, min_tensor), max_tensor)


class AccumulatedValue:
    def __init__(self):
        self._acc_val = 0
        self._acc_sample_size = 0

    def reset(self):
        self._acc_val = 0
        self._acc_sample_size = 0

    def accumulate(self, new_batch_val, bath_size):
        self._acc_val = (self._acc_val * self._acc_sample_size +
                         new_batch_val * bath_size) / (self._acc_sample_size + bath_size)
        self._acc_sample_size += bath_size
        return self._acc_val

    def item(self):
        return self._acc_val



class Trajectory:
    """Data structure to save trajectory data.
    Add observations, actions, and rewards during one episode.
    Finally, by calling to_tensor method, observations tensor ([horizon, observation dim]),
    actions tensor ([horizon, action dim]), and rewards tensor ([horizon]).
    """

    def __init__(self, observation_dim, action_dim):
        self.state_dim = observation_dim
        self.action_dim = action_dim

        self.observations = []
        self.actions = []
        self.rewards = []

    def add_action(self, action):
        """
        Params
        action: (np.ndarray [action_dim])
        """
        assert action.shape[-1] == self.action_dim, "action size {}".format(
            action.shape)
        self.actions.append(action)

    def add_observation(self, observation):
        """
        Params
        observation: (np.ndarray [state_dim])
        """
        assert observation.shape[-1] == self.state_dim, "observation size {}".format(
            observation)
        self.observations.append(observation)

    def add_reward(self, reward):
        self.rewards.append(reward)

    def to_tensor(self, device_name=None):
        """
        Return
        (
            observations tensor ([horizon, observation dim]),
            actions tensor ([horizon, action dim]),
            rewards tensor ([horizon]),
        )
        """
        observations_t = torch.tensor(self.observations, dtype=torch.float32)
        actions_t = torch.tensor(self.actions, dtype=torch.float32)
        rewards_t = torch.tensor(self.rewards, dtype=torch.float32)
        if device_name:
            observations_t = observations_t.to(device_name)
            action_t = action_t.to(device_name)
            rewards_t = rewards_t.to(device_name)

        return observations_t, actions_t, rewards_t
