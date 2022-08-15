from os import path
from typing import Type, Union

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import wandb
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from tqdm import tqdm

from algorithms.common import ReplayBuffer
from algorithms.model import ImpalaResNetCNN, AtariCNN, ActionNetwork, InverseDynamicsNetwork

gym.logger.set_level(40)


class BCOAgent(object):
    def __init__(self,
                 device: torch.device,
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 feature_extractor: Type[Union[ImpalaResNetCNN, AtariCNN]],
                 q_network_hidden_size: int,
                 batch_size: int,
                 pol_lr: float,
                 inv_dyn_lr: float,
                 update_period: int,
                 num_epochs_idm: int) -> None:
        self.device = device
        self.observation_space = observation_space
        self.action_space = action_space

        self.policy = ActionNetwork(
            observation_space=observation_space,
            action_space=action_space,
            cnn_module=feature_extractor,
            hidden_size=q_network_hidden_size
        ).to(self.device)

        self.inv_dynamics = InverseDynamicsNetwork(
            observation_space=observation_space,
            action_space=action_space,
            cnn_module=feature_extractor,
            hidden_size=q_network_hidden_size
        ).to(self.device)

        self.batch_size = batch_size
        self.num_epochs_idm = num_epochs_idm
        self.update_period = update_period

        self.pol_optim = optim.Adam(self.policy.parameters(), lr=pol_lr)
        self.inv_dyn_optim = optim.Adam(self.policy.parameters(), lr=inv_dyn_lr)

        self.steps_since_last_update = 0
        self.global_step = 0
        self.epoch = 0
        self.inv_dyn_learn_step = 0
        self.policy_learn_step = 0

        self.demonstrations = None
        self.online_samples = ReplayBuffer()

    @torch.no_grad()
    def act(self, obs: np.array) -> torch.Tensor:
        pi = self.pi(obs)
        return pi.sample().item()

    @torch.no_grad()
    def pi(self, obs: np.array) -> torch.distributions.Categorical:
        obs = torch.FloatTensor(obs).to(self.device)

        if obs.shape == self.observation_space.shape:
            obs = obs.unsqueeze(0)

        return Categorical(logits=self.policy(obs))

    @property
    def demonstrations(self):
        return self._demonstrations

    @demonstrations.setter
    def demonstrations(self, value):
        self._demonstrations = value

    def step(self, experience):
        self.online_samples.append(experience)

        self.global_step += 1

        if self.steps_since_last_update >= self.update_period:
            print(f"Update starting at global step {self.global_step}")
            self.epoch += 1
            self._update_inv_dynamics_model()
            self._update_policy_model()

            self.online_samples.clear()
            self.steps_since_last_update = 0

            wandb.log({'epoch': self.epoch, 'global_step': self.global_step})
            print("Update finished")
        else:
            self.steps_since_last_update += 1

    def _update_inv_dynamics_model(self):
        for _ in range(self.num_epochs_idm):
            dataloader = DataLoader(self.online_samples, shuffle=True, batch_size=self.batch_size, num_workers=4, pin_memory=True)
            self.inv_dynamics.train(True)
            for obs, act, _, next_obs, done in tqdm(dataloader, desc='Updating IDM'):
                pred = self.inv_dynamics(obs.to(self.device), next_obs.to(self.device))
                loss = F.cross_entropy(pred, act.to(self.device))

                self.inv_dyn_optim.zero_grad(set_to_none=True)
                loss.backward()
                self.inv_dyn_optim.step()

                self.inv_dyn_learn_step += 1
                if self.inv_dyn_learn_step % 10 == 0:
                    wandb.log({'inverse_dynamics_loss': loss,
                               'inv_dynamics_learn_step': self.inv_dyn_learn_step,
                               'global_step': self.global_step})

    def _update_policy_model(self):
        dataloader = DataLoader(self._demonstrations, shuffle=True, batch_size=self.batch_size, num_workers=4, pin_memory=True)
        self.inv_dynamics.eval()
        self.policy.train(True)
        for obs, _, _, next_obs, _ in tqdm(dataloader, desc='Updating policy model'):
            obs = obs.to(self.device)
            next_obs = next_obs.to(self.device)

            inv_pred_act = torch.argmax(self.inv_dynamics(obs, next_obs), dim=1)
            pred_act = self.policy(obs)

            loss = F.cross_entropy(pred_act, inv_pred_act)

            self.pol_optim.zero_grad(set_to_none=True)
            loss.backward()
            self.pol_optim.step()

            self.policy_learn_step += 1
            if self.policy_learn_step % 10 == 0:
                wandb.log({'policy_loss': loss,
                           'policy_learn_step': self.policy_learn_step,
                           'global_step': self.global_step})

    def save(self, save_path):
        checkpoint_filename = f'checkpoint_{self.global_step}.bco'
        checkpoint = {
            'policy_model_state_dict': self.policy.state_dict(),
            'invd_model_state_dict': self.inv_dynamics.state_dict(),
            'policy_optim_state_dict': self.pol_optim.state_dict(),
            'invd_optim_state_dict': self.inv_dyn_optim.state_dict()
        }
        checkpoint_full_path = path.join(save_path, checkpoint_filename)
        torch.save(checkpoint, checkpoint_full_path)
        return checkpoint_full_path

    def load(self, checkpoint_filename):
        checkpoint = torch.load(checkpoint_filename)
        self.policy.load_state_dict(checkpoint['policy_model_state_dict'])
        self.inv_dynamics.load_state_dict(checkpoint['invd_model_state_dict'])
        self.pol_optim.load_state_dict(checkpoint['policy_optim_state_dict'])
        self.inv_dyn_optim.load_state_dict(checkpoint['invd_optim_state_dict'])
