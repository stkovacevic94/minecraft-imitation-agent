from typing import Union, Type

import gym.spaces
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

gym.logger.set_level(40)


class AtariCNN(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU())

        self.output_size = 64 * 4 * 4

    def forward(self, x):
        return self.conv_layers(x)


class ImpalaResNetCNN(nn.Module):
    class _ImpalaResidual(nn.Module):

        def __init__(self, depth):
            super().__init__()
            self.conv1 = nn.Conv2d(depth, depth, 3, padding=1)
            self.conv2 = nn.Conv2d(depth, depth, 3, padding=1)

        def forward(self, x):
            out = F.relu(x)
            out = self.conv1(out)
            out = F.relu(out)
            out = self.conv2(out)
            return out + x

    def __init__(self, input_channels: int):
        super().__init__()
        layers = []
        depth_in = input_channels
        for depth_out in [32, 64, 64]:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                self._ImpalaResidual(depth_out),
                self._ImpalaResidual(depth_out),
            ])
            depth_in = depth_out
        self.conv_layers = nn.Sequential(*layers, nn.ReLU())
        self.output_size = math.ceil(64 / 8) ** 2 * depth_in

    def forward(self, x):
        return self.conv_layers(x)


class ActionNetwork(nn.Module):
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 cnn_module: Type[Union[ImpalaResNetCNN, AtariCNN]],
                 hidden_size=256):
        super().__init__()

        self.feature_extractor = cnn_module(observation_space.shape[0])
        self.conv_output_size = self.feature_extractor.output_size

        self.fc_h_a = nn.Linear(self.conv_output_size, hidden_size)
        self.fc_a = nn.Linear(hidden_size, action_space.n)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.conv_output_size)
        x = self.fc_h_a(x)
        return self.fc_a(F.relu(x))


class InverseDynamicsNetwork(nn.Module):
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 cnn_module: Type[Union[ImpalaResNetCNN, AtariCNN]],
                 hidden_size=256):
        super().__init__()

        self.feature_extractor = cnn_module(observation_space.shape[0])
        self.conv_output_size = self.feature_extractor.output_size

        self.fc_h_1 = nn.Linear(self.conv_output_size*2, hidden_size)
        self.fc_h_2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc_a = nn.Linear(hidden_size // 2, action_space.n)

    def forward(self, obs, next_obs):
        obs_features = self.feature_extractor(obs).view(-1, self.conv_output_size)
        next_obs_features = self.feature_extractor(next_obs).view(-1, self.conv_output_size)
        h = torch.cat([obs_features, next_obs_features], dim=1)
        h = F.relu(self.fc_h_1(h))
        h = F.relu(self.fc_h_2(h))
        return self.fc_a(h)
