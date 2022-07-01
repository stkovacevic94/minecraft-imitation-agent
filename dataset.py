import argparse
import os
import os.path as path

import cv2
import minerl
import torch
import pandas as pd

from torch.utils.data import Dataset


def prepare_dataset(environment, data_dir):
    assert environment in ["MineRLNavigate-v0", "MineRLNavigateDense-v0", "MineRLTreechop-v0"]

    data_env_dir = path.join(data_dir, environment)
    if path.exists(data_env_dir):
        print("Dataset found. Loading existing dataset.")
        return data_env_dir, pd.read_pickle(path.join(data_env_dir, "transitions.pkl"))

    print("Dataset not found. Creating it now...")
    cache_dir = path.join(data_dir, "cache")
    # Download data if needed
    if not path.exists(path.join(cache_dir, environment)):
        print("Raw dataset not found. Downloading it now...")
        minerl.data.download(
            directory=cache_dir,
            environment=environment,
            update_environment_variables=False,
            disable_cache=True
        )

    os.makedirs(data_env_dir)

    data_handle = minerl.data.make(environment, cache_dir)
    data = []
    for trajectory_idx, stream_name in enumerate(data_handle.get_trajectory_names()):
        for time_step, transition in enumerate(data_handle.load_data(stream_name=stream_name, include_metadata=True)):
            obs, action, reward, next_obs, done, metadata = transition

            cv2.imwrite(path.join(data_env_dir, f"{trajectory_idx}_{time_step}.bmp"), obs['pov'])
            if done:
                cv2.imwrite(path.join(data_env_dir, f"{trajectory_idx}_{time_step+1}.bmp"), next_obs['pov'])
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

    data_frame = pd.DataFrame(data)
    data_frame.to_pickle(path.join(data_env_dir, "transitions.pkl"))

    return data_env_dir, data_frame


class MineRLDataset(Dataset):
    def __init__(self, action_shaping, environment_name, data_dir) -> None:

        image_dir, transitions = prepare_dataset(environment_name, data_dir)
        self.image_dir = image_dir

        # Process only successful trajectories
        transitions = transitions[transitions['trajectory_success']]

        # Shape actions and remove no-ops
        transitions['shaped_action'] = transitions['action'].apply(lambda action: action_shaping(action))
        transitions = transitions[transitions['shaped_action'] != -1]

        self.transitions = transitions

    def __len__(self):
        return self.transitions.shape[0]

    def __getitem__(self, idx):
        item = self.transitions.iloc[idx]

        obs_img = cv2.imread(path.join(self.image_dir, f"{item['trajectory_idx']}_{item['time_step']}.bmp"))
        next_obs_img = cv2.imread(path.join(self.image_dir, f"{item['trajectory_idx']}_{item['time_step'] + 1}.bmp"))

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
