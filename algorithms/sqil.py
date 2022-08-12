from os import path
from typing import Type, Union

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import wandb
from torch.distributions import Categorical

from algorithms.common import ReplayBuffer, Experience
from algorithms.model import ImpalaResNetCNN, AtariCNN, ActionNetwork


class SQILAgent(object):
    def __init__(self,
                 device: torch.device,
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 feature_extractor: Type[Union[ImpalaResNetCNN, AtariCNN]],
                 q_network_hidden_size: int,
                 batch_size: int,
                 lr: float,
                 temp: float,
                 gamma: float,
                 sync_rate: int,
                 replay_buffer_capacity: int = 500000) -> None:

        self.device = device
        self.observation_space = observation_space

        self.online_q_network = ActionNetwork(
            observation_space=observation_space,
            action_space=action_space,
            cnn_module=feature_extractor,
            hidden_size=q_network_hidden_size
        ).to(self.device)
        self.target_q_network = ActionNetwork(
            observation_space=observation_space,
            action_space=action_space,
            cnn_module=feature_extractor,
            hidden_size=q_network_hidden_size
        ).to(self.device)
        for p in self.target_q_network.parameters():
            p.requires_grad = False
        self.target_q_network.eval()

        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.demo_buffer = ReplayBuffer(replay_buffer_capacity)

        self.batch_size = batch_size
        self.temp = temp
        self.gamma = gamma
        self.sync_rate = sync_rate

        self.optim = optim.Adam(self.online_q_network.parameters(), lr=lr)

        self.global_step = 0

    @torch.no_grad()
    def act(self, obs: np.array) -> torch.Tensor:
        pi = self.pi(obs)
        return pi.sample().item()

    @torch.no_grad()
    def pi(self, obs: np.array) -> torch.distributions.Categorical:
        obs = torch.FloatTensor(obs).to(self.device)

        if obs.shape == self.observation_space.shape:
            obs = obs.unsqueeze(0)

        q = self.online_q_network(obs)
        v = self._v_from_q(q)
        dist = torch.exp((q - v) / self.temp)
        dist = dist / torch.sum(dist)
        return Categorical(dist)

    def _v_from_q(self, q_value):
        return self.temp * torch.logsumexp(q_value / self.temp, dim=1, keepdim=True)

    def set_demonstrations(self, demonstrations_iterator):
        self.demo_buffer.clear()
        for experience in demonstrations_iterator:
            sqil_demo_experience = Experience(
                experience.obs,
                experience.action,
                1.0,
                experience.next_obs,
                experience.done)
            self.demo_buffer.append(sqil_demo_experience)

    def step(self, experience):
        sqil_experience = Experience(
            experience.obs,
            experience.action,
            0.0,
            experience.next_obs,
            experience.done)

        self.replay_buffer.append(sqil_experience)

        if len(self.replay_buffer) > 10*self.batch_size:
            self._learn()

        self.global_step += 1

    def _learn(self):
        # Sample online and demonstration batches
        online_batch = self.replay_buffer.sample(self.batch_size // 2)
        demo_batch = self.demo_buffer.sample((self.batch_size + 1) // 2)

        online_obs, online_action, online_reward, online_next_obs, online_done = online_batch
        expert_obs, expert_action, expert_reward, expert_next_obs, expert_done = demo_batch

        online_action = np.expand_dims(online_action, 1)
        online_reward = np.expand_dims(online_reward, 1)
        online_done = np.expand_dims(online_done, 1)

        expert_action = np.expand_dims(expert_action, 1)
        expert_reward = np.expand_dims(expert_reward, 1)
        expert_done = np.expand_dims(expert_done, 1)

        # Convert to Tensors and put on GPU
        batch_obs = torch.FloatTensor(np.concatenate((online_obs, expert_obs), axis=0)).to(self.device)
        batch_action = torch.FloatTensor(np.concatenate((online_action, expert_action), axis=0)).to(self.device)
        batch_reward = torch.FloatTensor(np.concatenate((online_reward, expert_reward), axis=0)).to(self.device)
        batch_next_obs = torch.FloatTensor(np.concatenate((online_next_obs, expert_next_obs), axis=0)).to(self.device)
        batch_done = torch.FloatTensor(np.concatenate((online_done, expert_done), axis=0)).to(self.device)

        # Normalize input
        obs_low = torch.FloatTensor(self.observation_space.low).to(self.device)
        obs_high = torch.FloatTensor(self.observation_space.high).to(self.device)
        obs_range = obs_high - obs_low
        batch_obs = (batch_obs - obs_low) / obs_range
        batch_next_obs = (batch_next_obs - obs_low) / obs_range

        # Compute loss
        with torch.no_grad():
            next_q = self.target_q_network(batch_next_obs)
            next_v = self._v_from_q(next_q)
            y = batch_reward + (1 - batch_done) * self.gamma * next_v

        loss = F.mse_loss(self.online_q_network(batch_obs).gather(1, batch_action.long()), y)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        self.optim.step()

        if self.global_step % self.sync_rate == 0:
            self.target_q_network.load_state_dict(self.online_q_network.state_dict())

        if self.global_step % 10 == 0:
            log_dict = {'loss': loss}
            dist = self.pi(torch.FloatTensor(expert_obs))
            log_dict['train_entropy'] = torch.mean(dist.entropy()).item()
            log_dict['train_kl'] = F.kl_div(torch.log(dist.probs), torch.FloatTensor(expert_action).to(self.device))
            wandb.log(log_dict, step=self.global_step)

    def save(self, save_path):
        checkpoint_filename = f'checkpoint_{self.global_step}.sqil'
        checkpoint = {
            'model_state_dict': self.online_q_network.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
        }
        checkpoint_full_path = path.join(save_path, checkpoint_filename)
        torch.save(checkpoint, checkpoint_full_path)
        return checkpoint_full_path

    def load(self, checkpoint_filename):
        checkpoint = torch.load(checkpoint_filename)
        self.online_q_network.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optim_state_dict'])