from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import DataLoader
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import AccumulatedValue, clamp, Trajectory
from misc import new_linear_net_params, swish, EnsembleDataset



class DynamicsModel(nn.Module):
    def __init__(self, ensemble_size, in_dim, out_dim, device_name='cpu'):
        super().__init__()
        
        self.ensemble_size = ensemble_size
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.device_name = device_name if torch.cuda.is_available() else 'cpu'

        self.lin0_w, self.lin0_b = new_linear_net_params(ensemble_size, in_dim, 200, self.device_name)
        self.lin1_w, self.lin1_b = new_linear_net_params(ensemble_size, 200, 200, self.device_name)
        self.lin2_w, self.lin2_b = new_linear_net_params(ensemble_size, 200, 200, self.device_name)
        self.lin3_w, self.lin3_b = new_linear_net_params(ensemble_size, 200, out_dim*2, self.device_name)

        self.inputs_mu = nn.Parameter(torch.zeros(in_dim, device=self.device_name), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.ones(in_dim, device=self.device_name), requires_grad=False)

        self.max_logvar = nn.Parameter(torch.ones(1, out_dim, dtype=torch.float32, device=self.device_name) / 2.0)
        self.min_logvar = nn.Parameter(torch.ones(1, out_dim, dtype=torch.float32, device=self.device_name) * -10.0)

    def fit_input_stats(self, data):
        """
        Params
        data: (list of tensors [torch.tensor([feature_dim])])
        """
        assert isinstance(data, list)
        assert isinstance(data[0], torch.Tensor)
        data_t = torch.cat([t[None,:] for t in data], dim=0)
        self.inputs_mu.data = torch.mean(data_t, dim=0).to(self.device_name)
        sigma = torch.std(data_t, dim=0)
        sigma[sigma < 1e-12] = 1.0
        self.inputs_sigma.data = sigma.to(self.device_name)

    def forward(self, x):
        """
        Params
        x: (torch.tensor [ensemble_size, batch_size, in_features])

        Return
        mean, std (torch.tensor [ensemble_size, batch_size, out_features])
        """
        org_device = x.device
        x = x.to(self.lin0_b.device)

        x = (x - self.inputs_mu) / self.inputs_sigma
        x = swish(x.matmul(self.lin0_w) + self.lin0_b)
        x = swish(x.matmul(self.lin1_w) + self.lin1_b)
        x = swish(x.matmul(self.lin2_w) + self.lin2_b)
        x = x.matmul(self.lin3_w) + self.lin3_b

        mean = x[:, :, :self.out_dim]

        logvar = x[:, :, self.out_dim:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean.to(org_device), logvar.to(org_device)

    def logvar_penalty_term(self):
        return (self.max_logvar - self.min_logvar).sum()



class DynamicsModelTrainer:
    def __init__(self, dynamics_model, device_name, ensamble_size):
        self.dynamics_model = dynamics_model
        self.device_name = device_name
        self.ensemble_size = ensamble_size
        self.train_inputs=[]
        self.train_targets=[]
        self.val_inputs=[]
        self.val_targets=[]

        self.optim = torch.optim.Adam(self.dynamics_model.parameters(), weight_decay=0.01)

    def train(self, trajectories, iters=10):
        """Train internal dynamics model.
        Params
        trajectories: list of Trajectory class objects

        Return
        (
            train_losses (list),
            validation_losses (list),
            train_accuracies (list),
            validation_accuracies (list)
        )
        """
        assert len(trajectories) > 0
        assert isinstance(trajectories[0], Trajectory)

        self.dynamics_model.train()

        # append new trajectory data
        new_inputs, new_targets = [], []
        for tr in trajectories:
            obs_t, actions_t, r_t = tr.to_tensor()  # [horizon, obs_dim], [horizon, actions_dim], [horizon]
            new_inputs.extend(list(torch.cat((obs_t[:-1], actions_t), dim=-1)))
            new_targets.extend(list(obs_t[1:]-obs_t[:-1]))
        idx = np.random.permutation(len(new_inputs))
        self.train_inputs.extend([new_inputs[i] for i in idx])
        self.train_targets.extend([new_targets[i] for i in idx])

        self.dynamics_model.fit_input_stats(self.train_inputs)

        train_dataset = EnsembleDataset(self.train_inputs, self.train_targets, self.ensemble_size)
        train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)

        train_losses, val_losses = [], []
        
        with tqdm(total=iters) as pbar:
            pbar.write("TRAIN DYNAMICS MODEL ............")
            for itr in range(iters):
                # train
                self.dynamics_model.train()
                acc_train_loss = AccumulatedValue()
                for batch_in, batch_tar in train_dl:
                    batch_size = len(batch_in)
                    batch_in.transpose_(0, 1)  # to ensemble first
                    # assert batch_in.size() == (self.ensemble_size, batch_size, self.state_dim + self.action_dim), "batch_in size {}".format(batch_in.size())
                    batch_tar.transpose_(0, 1)  # to ensemble first
                    # assert batch_tar.size() == (self.ensemble_size, batch_size, 1 + self.state_dim), "batch_tar size {}".format(batch_tar.size())
                    batch_in = batch_in.to(self.device_name)
                    batch_tar = batch_tar.to(self.device_name)

                    mean, logvar = self.dynamics_model(batch_in)
                    train_loss = (((mean - batch_tar) ** 2) * torch.exp(-logvar)).sum(-1) + logvar.sum(-1)
                    # assert train_loss.size() == (self.ensemble_size, batch_size), "train_loss size {}".format(train_loss.size())
                    train_loss = train_loss.mean()
                    train_loss += 0.01 * self.dynamics_model.logvar_penalty_term()

                    self.optim.zero_grad()
                    train_loss.backward()
                    self.optim.step()

                    acc_train_loss.accumulate(train_loss.item(), batch_size)
                train_losses.append(acc_train_loss.item())


                # pbar.write("[ITER {}] train loss:{:.3f},  validation loss:{:.3f}".format(
                #     itr, acc_train_loss.item(), acc_val_loss.item()))
                pbar.update()

            pbar.write("final train loss: {:.3f}".format(train_losses[-1]))

        return train_losses
