import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import copy

from typing import List, Dict

from torch.distributions import Categorical
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from data import ReplayBuffer, ReplayBufferTorch


class SQILAgent(pl.LightningModule):
    def __init__(self,
                 q_network: nn.Module,
                 env: gym.Env,
                 expert_demonstrations: ReplayBuffer,
                 batch_size: int,
                 episode_max_length: int,
                 lr: float,
                 alpha: float,
                 gamma: float,
                 sync_rate: int,
                 online_buffer_capacity: int = 500000) -> None:
        super().__init__()

        self.online_q_network = q_network
        self.target_q_network = copy.deepcopy(q_network)
        for p in self.target_q_network.parameters():
            p.requires_grad = False
        self.target_q_network.eval()

        self.env = env
        self.obs = None
        self.episode_step = 0
        self.episode_reward = 0

        self.expert_demonstrations = expert_demonstrations
        self.online_buffer = ReplayBuffer(online_buffer_capacity)

        self.save_hyperparameters('batch_size', 'episode_max_length', 'lr', 'alpha', 'gamma', 'sync_rate')

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        q = self.online_q_network(obs)
        v = self.get_v(q).squeeze()
        dist = torch.exp((q - v) / self.hparams.alpha)
        # print(dist)
        dist = dist / torch.sum(dist)
        # print(dist)
        c = Categorical(dist)
        a = c.sample()
        return a.item()

    def get_v(self, q_value):
        return self.hparams.alpha * torch.log(torch.sum(torch.exp(q_value / self.hparams.alpha), dim=1, keepdim=True))

    def on_train_start(self) -> None:
        # Populate online buffer
        while len(self.online_buffer) < self.hparams.episode_max_length:
            obs = self.env.reset()
            for _ in range(self.hparams.episode_max_length):
                action = self(torch.FloatTensor(obs).to(self.device))
                next_obs, _, done, _ = self.env.step(action)
                self.online_buffer.append((obs, action, 0., next_obs, done))
                obs = next_obs
                if done:
                    break
        self.obs = self.env.reset()
        self.episode_step = 0
        self.episode_reward = 0

    def training_step(self, batch):
        # Take step in environment
        action = self(torch.FloatTensor(self.obs).to(self.device))
        next_obs, reward, done, _ = self.env.step(action)
        self.episode_reward += reward
        self.episode_step += 1
        self.online_buffer.append((self.obs, action, 0.,  next_obs, done))

        online_obs, online_action, online_reward, online_next_obs, online_done = batch['online']
        expert_obs, expert_action, expert_reward, expert_next_obs, expert_done = batch['expert']

        expert_reward = torch.zeros_like(expert_reward)
        expert_reward[:] = 1.0
        online_done = online_done.float()
        expert_done = expert_done.float()

        batch_obs = torch.cat([online_obs, expert_obs], dim=0).squeeze()
        batch_action = torch.cat([online_action, expert_action], dim=0).unsqueeze(1)
        batch_reward = torch.cat([online_reward, expert_reward], dim=0).unsqueeze(1)
        batch_next_obs = torch.cat([online_next_obs, expert_next_obs], dim=0).squeeze()
        batch_done = torch.cat([online_done, expert_done], dim=0).unsqueeze(1)

        # Compute loss
        with torch.no_grad():
            next_q = self.target_q_network(batch_next_obs)
            next_v = self.get_v(next_q)
            y = batch_reward + (1 - batch_done) * self.hparams.gamma * next_v

        loss = F.mse_loss(self.online_q_network(batch_obs).gather(1, batch_action.long()), y)

        if self.global_step % self.hparams.sync_rate == 0:
            self.target_q_network.load_state_dict(self.online_q_network.state_dict())

        log_dict = {
            'loss': loss
        }
        if done or self.episode_step == self.hparams.episode_max_length:
            log_dict['reward'] = self.episode_reward
            self.obs = self.env.reset()
            self.episode_step = 0
            self.episode_reward = 0
        else:
            self.obs = next_obs

        self.log_dict(log_dict, on_step=True)
        return loss

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.online_q_network.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def train_dataloader(self) -> Dict[str, DataLoader]:
        online_dataset = ReplayBufferTorch(replay_buffer=self.online_buffer,
                                           sample_size=self.hparams.episode_max_length)
        expert_dataset = ReplayBufferTorch(replay_buffer=self.expert_demonstrations,
                                           sample_size=self.hparams.episode_max_length)

        online_dataloader = DataLoader(online_dataset,
                                       batch_size=self.hparams.batch_size // 2,
                                       pin_memory=True)
        expert_dataloader = DataLoader(expert_dataset,
                                       batch_size=(self.hparams.batch_size + 1) // 2,
                                       pin_memory=True)

        return {'online': online_dataloader, 'expert': expert_dataloader}

