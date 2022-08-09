import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np

from typing import List

from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from data import ReplayBufferTorch


class BCAgent(pl.LightningModule):

    def __init__(self, policy, env, expert_demonstrations, batch_size, lr, env_validation_period: int = 500) -> None:
        super().__init__()

        self.policy = policy
        self.env = env
        self.expert_demonstrations = expert_demonstrations
        self.env_validation_period = env_validation_period

        self.save_hyperparameters('batch_size', 'lr')

        self._loss_fn = nn.CrossEntropyLoss()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.policy(obs)).item()

    def training_step(self, batch):
        obs, actions, _, _, _ = batch

        loss = self._loss_fn(self.policy(obs.squeeze()), actions)

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        if self.global_step % self.env_validation_period == 0:
            try:
                print("Validating on live environment...")
                rewards = []
                for _ in range(3):
                    total_reward = 0
                    obs = self.env.reset()
                    for i in range(3000):
                        action = self(torch.tensor(obs, device=self.device))
                        obs, reward, done, _ = self.env.step(action)
                        total_reward += reward
                        if done:
                            break
                    rewards.append(total_reward)
                self.log("reward", np.mean(rewards).item())
            except:
                print("Exception")

        return loss

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.policy.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.expert_demonstrations, num_workers=4, batch_size=self.hparams.batch_size, pin_memory=True)

