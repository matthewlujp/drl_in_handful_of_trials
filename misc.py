import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats
import numpy as np



def truncated_normal(size):
    return torch.tensor(stats.truncnorm(-2, 2, loc=np.zeros(size)).rvs()).type(torch.float32)


def new_linear_net_params(ensemble_size, in_features, out_features, device):
    w = truncated_normal(size=(ensemble_size, in_features, out_features)).to(
        device) / (2.0 * np.sqrt(in_features))
    w = nn.Parameter(w)
    b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features,
                                 dtype=torch.float32, device=device))
    return w, b


def swish(x):
    return x * torch.sigmoid(x)



class EnsembleDataset(Dataset):
    def __init__(self, input_data, target_data, ensemble_size):
        assert isinstance(input_data, (list, tuple)), "input data type {}".format(type(input_data))
        assert isinstance(target_data, (list, tuple)), "target data type {}".format(type(target_data))
        assert len(input_data) == len(target_data), "input data length {}, target data length {}".format(
            len(input_data), len(target_data))
        assert len(input_data[0].size()) == 1, "input_data element size {}".format(input_data[0].size())
        assert len(target_data[0].size()) == 1, "target_data element size {}".format(target_data[0].size())

        self.input_data = input_data
        self.target_data = target_data
        self.ensemble_size = ensemble_size

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, i):
        """
        Return
        (
            input_v: (torch.tensor [ensemble_size, in_features]),
            target_v: (torch.tensor [ensemble_size, out_features]),
        )
        """
        input_vs = self.input_data[i].view(1, -1).repeat(self.ensemble_size, 1)
        target_vs = self.target_data[i].view(1, -1).repeat(self.ensemble_size, 1)
        return input_vs, target_vs
