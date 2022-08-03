import dataclasses
from os import path

import tqdm
import numpy as np
import minerl
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


@dataclasses.dataclass(frozen=True)
class Transitions(Dataset):
    obs: np.ndarray
    acts: np.ndarray

    def __post_init__(self):
        """Performs input validation: check shapes & dtypes match docstring.
        Also make array values read-only.
        Raises:
            ValueError: if batch size (array length) is inconsistent
                between `obs`, `acts` and `infos`.
        """
        for val in vars(self).values():
            if isinstance(val, np.ndarray):
                val.setflags(write=False)

        if len(self.obs) != len(self.acts):
            raise ValueError(
                "obs and acts must have same number of timesteps: "
                f"{len(self.obs)} != {len(self.acts)}",
            )

    def __len__(self):
        """Returns number of transitions. Always positive."""
        return len(self.obs)

    def __getitem__(self, index) -> T_co:
        return self.obs[index], self.acts[index]


def get_dataset(wrapped_env, save_path, fast_dev_run: bool = False) -> Transitions:
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
    observations = []
    actions = []
    trajectories_to_load = data_handle.get_trajectory_names()
    if fast_dev_run:
        trajectories_to_load = data_handle.get_trajectory_names()[:2]
    for trajectory_idx, stream_name in enumerate(tqdm.tqdm(trajectories_to_load)):
        for time_step, transition in enumerate(
                data_handle.load_data(stream_name=stream_name)):
            obs, action, reward, next_obs, done = transition
            wrapped_obs = wrapped_env.observation(obs) if hasattr(wrapped_env, 'observation') else obs
            wrapped_action = wrapped_env.wrap_action(action) if hasattr(wrapped_env, 'wrap_action') else action
            if wrapped_action != 7:
                observations.append(wrapped_obs.squeeze())
                actions.append(wrapped_action)
    return Transitions(np.stack(observations), np.array(actions, dtype=np.longlong))
