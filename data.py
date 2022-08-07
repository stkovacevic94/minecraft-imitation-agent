from abc import ABC
from collections import deque, namedtuple
from os import path
from typing import Tuple, Iterator

import numpy as np
import tqdm
import minerl

from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co

# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    "Experience",
    field_names=["obs", "action", "reward", "next_obs", "done"],
)


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (obs, action, reward, next_obs, done)
        """
        self.buffer.append(experience)

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

    def clear(self):
        self.buffer.clear()


class ReplayBufferTorch(IterableDataset, ABC):
    def __init__(self, replay_buffer: ReplayBuffer, sample_size: int):
        self.replay_buffer = replay_buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[T_co]:
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield obs[i], actions[i], rewards[i], next_obs[i], dones[i]


def load_expert_demonstrations(replay_buffer: ReplayBuffer, wrapped_env, save_path, fast_dev_run: bool = False) -> None:
    env_id = wrapped_env.unwrapped.spec.id

    # Download data if needed
    if not path.exists(path.join(save_path, env_id)):
        minerl.data.download(
            directory=save_path,
            environment=env_id,
            update_environment_variables=False,
            disable_cache=True
        )

    data_handle = minerl.data.make(env_id, save_path)
    trajectories_to_load = data_handle.get_trajectory_names()
    if fast_dev_run:
        trajectories_to_load = data_handle.get_trajectory_names()[:5]
    for trajectory_idx, stream_name in enumerate(tqdm.tqdm(trajectories_to_load)):
        for time_step, transition in enumerate(
                data_handle.load_data(stream_name=stream_name)):
            obs, action, reward, next_obs, done = transition
            wrapped_obs = wrapped_env.observation(obs)
            wrapped_action = wrapped_env.wrap_action(action)
            wrapped_next_obs = wrapped_env.observation(next_obs)

            if wrapped_action != -1:
                experience = (wrapped_obs, wrapped_action, reward, wrapped_next_obs, done)
                replay_buffer.append(experience)
