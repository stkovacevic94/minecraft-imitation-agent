from os import path

import pandas as pd
import torch
import tqdm
import minerl
from torch.utils.data import Dataset


class MineRLDataset(Dataset):
    def __init__(self, wrapped_env, save_path, remove_noop=True) -> None:
        env_id = wrapped_env.unwrapped.spec.id
        self.wrapped_env = wrapped_env

        # Download data if needed
        if not path.exists(path.join(save_path, env_id)):
            minerl.data.download(
                directory=save_path,
                environment=env_id,
                update_environment_variables=False,
                disable_cache=True
            )

        data_handle = minerl.data.make(env_id, save_path)
        data = []
        for trajectory_idx, stream_name in enumerate(tqdm.tqdm(data_handle.get_trajectory_names()[:2])):
            for time_step, transition in enumerate(
                    data_handle.load_data(stream_name=stream_name)):
                obs, action, reward, next_obs, done = transition

                if remove_noop and wrapped_env.wrap_action(action) != 7:
                    data.append({
                        "trajectory_idx": trajectory_idx,
                        "time_step": time_step,
                        "obs": obs,
                        "action": action,
                        "reward": reward,
                        "next_obs": next_obs,
                        "done": done
                    })

        self.transitions = pd.DataFrame(data)

    def __len__(self):
        return self.transitions.shape[0]

    def __getitem__(self, idx):
        item = self.transitions.iloc[idx]

        wrapped_obs = self.wrapped_env.observation(item['obs'])
        wrapped_next_obs = self.wrapped_env.observation(item['next_obs'])
        wrapped_action = self.wrapped_env.wrap_action(item['action'])

        th_wrapped_obs = torch.tensor(wrapped_obs, dtype=torch.float32)
        th_action = torch.tensor(wrapped_action, dtype=torch.long)
        th_reward = torch.tensor(item['reward'], dtype=torch.float32)
        th_wrapped_next_obs = torch.tensor(wrapped_next_obs, dtype=torch.float32)
        th_done = torch.tensor(item['done'])

        return th_wrapped_obs, th_action, th_reward, th_wrapped_next_obs, th_done
