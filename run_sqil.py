import argparse
import os

import gym
import numpy as np
import torch
import wandb
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from algorithms.common import Experience
from algorithms.sqil import SQILAgent
from algorithms.model import AtariCNN
from wrappers import ActionShaping, ExtractPOVAndTranspose, create_data_iterator


def validate(agent: SQILAgent, validation_dataset: TensorDataset):
    dataloader = DataLoader(validation_dataset, batch_size=1024)
    probs = []
    entropies = []
    for batch in dataloader:
        obs, actions = batch
        dist = agent.pi(obs)
        entropies.append(torch.mean(dist.entropy()).item())
        probs.append(dist.probs.gather(1, actions.cuda().long()).mean().item())
    return np.mean(probs), np.mean(entropies)


def train_sqil(hparams):
    torch.backends.cudnn.benchmark = True

    os.makedirs(hparams.logdir, exist_ok=True)
    wandb.init(
        project="master-thesis",
        group="SQIL",
        job_type='train',
        dir=hparams.logdir,
        config=hparams,
        settings=wandb.Settings(start_method="fork") if os.name == 'posix' else None, )

    env = ExtractPOVAndTranspose(ActionShaping(gym.make("MineRLTreechop-v0")))
    env.seed(95)
    if hparams.fast_dev_run:
        num_to_load = 10000
        val_to_load = 1000
    else:
        num_to_load = None
        val_to_load = 10000
    demo_iterator = iter(create_data_iterator(env, hparams.data_path, num_to_load))

    val_obs = torch.zeros(size=(val_to_load, )+env.observation_space.shape)
    val_act = torch.zeros(size=(val_to_load, 1))
    for i in range(val_to_load):
        experience = next(demo_iterator)
        val_obs[i] = torch.FloatTensor(experience.obs)
        val_act[i] = experience.action
    val_dataset = TensorDataset(val_obs, val_act)

    agent = SQILAgent(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        observation_space=env.observation_space,
        action_space=env.action_space,
        feature_extractor=AtariCNN,
        q_network_hidden_size=512,
        batch_size=hparams.batch_size,
        lr=hparams.learning_rate,
        temp=hparams.temp,
        gamma=hparams.gamma,
        sync_rate=hparams.sync_rate)

    agent.set_demonstrations(demo_iterator)
    for episode in tqdm(range(hparams.max_episodes), desc='Episode'):
        obs = env.reset()
        episode_reward = 0
        for _ in tqdm(range(hparams.episode_max_length), desc=f'Episode {episode} rollout'):
            action = agent.act(obs.copy())
            next_obs, reward, done, _ = env.step(action)
            agent.step(Experience(obs, action, reward, next_obs, done))

            obs = next_obs
            episode_reward += reward

            if agent.global_step % 10 == 0:
                log_dict = {}
                for p in list(
                        filter(lambda param: param[1].grad is not None, agent.online_q_network.named_parameters())):
                    log_dict[f'gradients/norm/{p[0]}'] = p[1].grad.data.norm(2).item()
                wandb.log(log_dict, step=agent.global_step)
            if done:
                break
        if episode % 5 == 0:
            checkpoint_full_path = agent.save(wandb.run.dir)
            wandb.save(checkpoint_full_path)
        likelihood, entropy = validate(agent, validation_dataset=val_dataset)
        wandb.log({'reward': episode_reward,
                   'episode': episode,
                   'val_likelihood': likelihood,
                   'val_entropy': entropy},
                  step=agent.global_step)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', type=str, default="./logs", help='Root directory path for logs')
    parser.add_argument('--data_path', type=str, required=True, help='Dataset path')
    parser.add_argument('--max_episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--fast_dev_run', action="store_true", help='Run 1 epoch for test')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--episode_max_length', type=int, default=3000, help='Max episode length')
    parser.add_argument('--temp', type=float, default=3, help='Soft Q Learning temperature parameter')
    parser.add_argument('--gamma', type=float, default=0.99, help='MDP discount parameter gamma')
    parser.add_argument('--sync_rate', type=int, default=100, help='Rate at which target network and online network sync')

    args = parser.parse_args()

    train_sqil(args)
