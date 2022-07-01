import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np

from typing import List
from torch.optim import Optimizer


class BCAgent(pl.LightningModule):

    def __init__(self, policy, env, lr) -> None:
        super().__init__()

        self.policy = policy
        self.env = env

        self.save_hyperparameters('lr')

        self._loss_fn = nn.CrossEntropyLoss()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.policy(obs)

    def training_step(self, batch):
        obs, actions, reward, next_obs, done = batch

        logits = self(obs)

        loss = self._loss_fn(logits, actions)

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        obs, actions, reward, next_obs, done = batch

        logits = self(obs)

        loss = self._loss_fn(logits, actions)

        self.log("val_loss", loss, on_step=True, on_epoch=True)

        rewards = []
        obs = self.env.reset()
        for i in range(2000):
            action = self(torch.tensor(np.expand_dims(obs, axis=0), dtype=torch.float32, device=self.device))
            obs, reward, done, _ = self.env.step(torch.argmax(action).item())
            rewards.append(reward)
            if done:
                break

        self.log("val_reward", np.mean(rewards), on_step=True, on_epoch=True)

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.policy.parameters(), lr=self.hparams.lr)
        return [optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
        return parent_parser
