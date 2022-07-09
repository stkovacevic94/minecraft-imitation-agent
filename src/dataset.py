import argparse
import os
import os.path as path

import numpy as np
import torch
import minerl
import tqdm
import pandas as pd

from torch.utils.data import Dataset


# TODO: Replace this with generic dataset with @dataclasses and np.memmap
class MineRLDataset(Dataset):
    def __init__(self, environment, action_shaping, data_dir) -> None:
        self.__download_dataset(environment, data_dir)

        # Process only successful trajectories
        self.transitions = self.transitions[self.transitions['trajectory_success']]

        # Shape actions and remove no-ops
        self.transitions['shaped_action'] = self.transitions['action'].apply(lambda action: action_shaping(action))
        self.transitions = self.transitions[self.transitions['shaped_action'] != -1]

    def __download_dataset(self, environment, data_dir):
        assert environment in ["MineRLNavigate-v0", "MineRLNavigateDense-v0", "MineRLTreechop-v0"]

        self.data_env_dir = path.join(data_dir, environment)
        if path.exists(self.data_env_dir):
            print("Dataset found. Loading existing dataset.")
            self.transitions = pd.read_pickle(path.join(self.data_env_dir, "transitions.pkl"))
            return

        print("Dataset not found. Creating it now...")
        cache_dir = path.join(data_dir, "cache")
        # Download data if needed
        if not path.exists(path.join(cache_dir, environment)):
            minerl.data.download(
                directory=cache_dir,
                environment=environment,
                update_environment_variables=False,
                disable_cache=True
            )

        os.makedirs(self.data_env_dir)

        data_handle = minerl.data.make(environment, cache_dir)
        data = []
        for trajectory_idx, stream_name in enumerate(tqdm.tqdm(data_handle.get_trajectory_names())):
            for time_step, transition in enumerate(
                    data_handle.load_data(stream_name=stream_name, include_metadata=True)):
                obs, action, reward, next_obs, done, metadata = transition

                np.save(path.join(self.data_env_dir, f"{trajectory_idx}_{time_step}.npy"), obs['pov'])
                if done:
                    np.save(path.join(self.data_env_dir, f"{trajectory_idx}_{time_step + 1}.npy"), next_obs['pov'])
                obs.pop('pov')
                next_obs.pop('pov')

                data.append({
                    "trajectory_idx": trajectory_idx,
                    "time_step": time_step,
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "next_obs": next_obs,
                    "done": done,
                    "trajectory_success": bool(metadata['success']),
                    "total_steps": metadata['duration_steps'],
                    "total_reward": metadata['total_reward']
                })

        self.transitions = pd.DataFrame(data)
        self.transitions.to_pickle(path.join(self.data_env_dir, "transitions.pkl"))

    def __len__(self):
        return self.transitions.shape[0]

    def __getitem__(self, idx):
        item = self.transitions.iloc[idx]

        obs_img = np.load(path.join(self.data_env_dir, f"{item['trajectory_idx']}_{item['time_step']}.npy"))
        next_obs_img = np.load(path.join(self.data_env_dir, f"{item['trajectory_idx']}_{item['time_step'] + 1}.npy"))

        obs = torch.tensor(obs_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        action = torch.tensor(item['shaped_action'], dtype=torch.long)
        reward = torch.tensor(item['reward'], dtype=torch.float32)
        next_obs = torch.tensor(next_obs_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        done = torch.tensor(item['done'])

        return obs, action, reward, next_obs, done

    @staticmethod
    def add_data_specific_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument('--data_path', type=str, required=True, help='Dataset path')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        return parent_parser


if __name__ == '__main__':
    from wrappers import ActionManager
    action_manager = ActionManager()

    # TODO: Improve action_manager dependency with dataset
    # TODO: Add parameter for environment
    dataset = MineRLDataset("MineRLTreechop-v0", action_manager.action_id, "../dataset")

    dataset[0]
