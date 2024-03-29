import abc
from collections import namedtuple, deque

# Named tuple for storing experience steps gathered in training
from typing import Tuple

import numpy as np
import torch.distributions
from torch.utils.data import Dataset

Experience = namedtuple(
    "Experience",
    field_names=["obs", "action", "reward", "next_obs", "done"],
)


class ReplayBuffer(Dataset):
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int = None) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        obs, actions, rewards, next_obs, dones = self.buffer[item]

        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.longlong),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=bool)
        )

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (obs, action, reward, next_obs, done)
        """
        self.buffer.append(experience)

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs, actions, rewards, next_obs, dones = zip(*(self.buffer[idx] for idx in indices))

        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.longlong),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=bool)
        )


class ImitationLearningAgent(abc.ABC):
    @abc.abstractmethod
    def act(self, obs: np.ndarray):
        raise NotImplementedError()

    @abc.abstractmethod
    def pre_interaction(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def pre_episode(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def post_episode(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self, transition):
        raise NotImplementedError()