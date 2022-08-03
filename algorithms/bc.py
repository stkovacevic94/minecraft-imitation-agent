import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np

from typing import List

from torch.optim import Optimizer


class BCAgent(pl.LightningModule):

    def __init__(self, policy, env, lr, env_validation_period: int = 500) -> None:
        super().__init__()

        self.policy = policy
        self.env = env
        self.env_validation_period = env_validation_period

        self.save_hyperparameters('lr')

        self._loss_fn = nn.CrossEntropyLoss()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.policy(obs)).item()

    def training_step(self, batch):
        obs, actions = batch

        loss = self._loss_fn(self.policy(obs), actions)

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        if self.global_step % self.env_validation_period == 0:
            rewards = []
            try:
                print("Validating on live environment...")
                for _ in range(5):
                    total_reward = 0
                    obs = self.env.reset()
                    for i in range(2000):
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

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
        return parent_parser
