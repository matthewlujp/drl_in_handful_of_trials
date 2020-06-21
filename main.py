from controller import MPC, space_extractor
from dynamics_model import DynamicsModel, DynamicsModelTrainer

import gym
import torch
import numpy as np
from tqdm import tqdm
import pickle
from argparse import ArgumentParser
import copy
import shutil
import importlib
import os
import sys

from logger import SimpleLogger
from utils import AccumulatedValue, Trajectory



# ENSEMBLE_SIZE = 5
# PARTICLE_SIZE = 10
# CEM_SAMPLE_SIZE = 50
# CEM_ELITE_RATE = 0.1
# CEM_MAX_ITER = 5
# TRAJECTORIES_PER_EPOCH = 1
# DYNAMICS_MODEL_TRAIN_ITER_PER_EPOCH = 5
# PLAN_HORIZON = 10
# TASK_HORIZON = 1000

ENSEMBLE_SIZE = 5
PARTICLE_SIZE = 10
CEM_SAMPLE_SIZE = 20
CEM_ELITE_RATE = 0.1
CEM_MAX_ITER = 5
TRAJECTORIES_PER_EPOCH = 1
DYNAMICS_MODEL_TRAIN_ITER_PER_EPOCH = 5
PLAN_HORIZON = 10
TASK_HORIZON = 1000




def run_simulation(env, controller, should_visualize=False):
    """
    Params
    env: (gym.env)
    controller: (MPC)
    should_visualize: bool
    """
    traj = Trajectory(env.observation_space.shape[0], env.action_space.shape[0])
    obs = env.reset()
    traj.add_observation(obs)
    done = False
    action_seq = None
    
    for _ in range(TASK_HORIZON):
        a, action_seq = controller.act(obs, action_seq)
        _a = a[0] if isinstance(env.action_space, gym.spaces.Discrete) else a
        next_obs, r, done, _ = env.step(_a)
        if should_visualize:
            env.render(mode='human')

        traj.add_action(a)
        traj.add_observation(next_obs)
        traj.add_reward(r)
        obs = next_obs

        if done:
            break

    return traj



class Trainer:
    def __init__(self, args):
        self.save_dir = args.save_dir
        if "test" in self.save_dir and os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        if os.path.exists(self.save_dir):
            raise ValueError("Directory {} already exists.".format(self.save_dir))
        os.makedirs(self.save_dir)
        self.ckpt_dir = os.path.join(self.save_dir, 'checkpoints')
        os.makedirs(self.ckpt_dir)
        self.log_dir = os.path.join(self.save_dir, 'logs')
        os.makedirs(self.log_dir)
        print("Train data will be saved in\n   {}\n   {}".format(self.ckpt_dir, self.log_dir))

        print(args.env)
        env_mod = importlib.import_module("envs.{}".format(args.env))
        self.env_man = env_mod.EnvironmentManager()

        self.max_epochs = args.max_epochs
        self.epoch = 0
        self.plan_horizon = PLAN_HORIZON
        self.ckpt_save_steps = args.ckpt_save_steps
        self.device_name = args.device

        self.env = self.env_man.new_env()

        state_dim, _, _ = space_extractor(self.env.observation_space)
        action_dim, _, _ = space_extractor(self.env.action_space)
        self.dynamics_model = DynamicsModel(ENSEMBLE_SIZE, state_dim + action_dim, state_dim, device_name=self.device_name)
        self.dynamics_model.share_memory()
        
        self.controller = MPC(
            self.dynamics_model, self.env.observation_space, self.env.action_space,
            ENSEMBLE_SIZE, PARTICLE_SIZE, CEM_SAMPLE_SIZE, CEM_ELITE_RATE, CEM_MAX_ITER, PLAN_HORIZON,
            cost_func=self.env_man.cost_func)

        self.trainer = DynamicsModelTrainer(self.dynamics_model, self.device_name, ENSEMBLE_SIZE)

        self.logger = SimpleLogger(["mean_reward", "model_train_loss"])
        

        if args.ckpt_filepath:
            self.load_ckpt(args.ckpt_filepath)

    def train(self):
        for epoch in range(self.epoch, self.max_epochs):
            print("\nEPOCH {}:".format(epoch))

            self.epoch = epoch

            # collect samples
            acc_reward = AccumulatedValue()
            trajectories = []
            with tqdm(total=TRAJECTORIES_PER_EPOCH) as pbar:
                pbar.write("COLLECTING {} SAMPLES ............".format(TRAJECTORIES_PER_EPOCH))

                for i in range(TRAJECTORIES_PER_EPOCH):
                    traj = run_simulation(self.env, self.controller)
                    trajectories.append(traj)
                    _, _, rs_t = traj.to_tensor()
                    acc_reward.accumulate(rs_t.sum().item(), 1)
                    pbar.write("sample {}  reward: {}".format(i+1, rs_t.sum().item()))
                    pbar.update()

                pbar.write("mean reward: {}".format(acc_reward.item()))

            self.logger.append('mean_reward', acc_reward.item())

            train_losses = self.trainer.train(trajectories, iters=DYNAMICS_MODEL_TRAIN_ITER_PER_EPOCH)
            self.logger.extend("model_train_loss", train_losses)
            self.controller.has_model_trained = True

            # save logger
            self.logger.save(os.path.join(self.log_dir, "epoch{}".format(epoch)))
            
            # save checkpoint
            if epoch % self.ckpt_save_steps == 0 and epoch > 0:
                self.save_ckpt("ckpt_epoch{}".format(epoch))

        # save the final model
        self.save_ckpt("final_model_epoch{}".format(self.epoch+1))


    def eval(self):
        """Run sample trajectory.
        """
        traj = run_simulation(self.env, self.controller, should_visualize="DISPLAY" in os.environ)
        _, _, rs_t = traj.to_tensor()
        print("reward: {}".format(torch.sum(rs_t)))

        # TODO: save with time stamp
        with open(os.path.join(self.log_dir, "eval.pickle"), 'wb') as f:
            pickle.dump(traj, f)


    def save_ckpt(self, filename):
        save_data = {
            'epoch': self.epoch,
            'logger': self.logger,
            'plan_horizon': self.plan_horizon,
            'model_state_dict': self.dynamics_model.state_dict(),
        }

        filepath = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(filepath):
            raise ValueError("Checkpoint file {} already exists".format(filepath))
        with open(filepath, 'wb') as f:
            torch.save(save_data, f)


    def load_ckpt(self, filepath):
        with open(filepath, 'rb') as f:
            if self.device_name == 'cpu':
                ckpt = torch.load(f, map_location='cpu')
            else:
                ckpt = torch.load(f)

        self.epoch = ckpt['epoch']
        self.logger = ckpt['logger']
        self.plan_horizon = ckpt['plan_horizon']
        self.dynamics_model.load_state_dict(ckpt['model_state_dict'])
        self.controller.has_model_trained = True






if __name__ == '__main__':
    parser = ArgumentParser(description="Test MBRL(handful) with gym.")
    parser.add_argument("-save_dir", default="results/test", help="Path to a directory to save training data.")
    parser.add_argument('-env', default='cartpole', choices={'cartpole', 'acrobot', 'pendulum', 'halfcheetah'})
    parser.add_argument("-max_epochs", default=100, type=int, help="Max train epochs.")
    parser.add_argument('-ckpt_filepath', help="Path to checkpoint file to load.")
    parser.add_argument('-ckpt_save_steps', default=5, type=int, help="How frequent a checkpoint is saved.")
    parser.add_argument('-device', default='cpu', choices={'cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'})
    parser.add_argument('-e', '--eval', action='store_true', help='Run sample trajectory with trained controller.')
    args = parser.parse_args()

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    

    trainer = Trainer(args)
    if not args.eval:
        trainer.train()
    else:
        trainer.eval()

    
