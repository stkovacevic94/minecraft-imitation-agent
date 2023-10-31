import argparse
import os

import gym
import torch
import wandb
from torch.nn.functional import cross_entropy
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm

from algorithms.bco import BCOAgent
from algorithms.common import Experience, ReplayBuffer
from algorithms.model import ImpalaResNetCNN, AtariCNN
from wrappers import ActionShaping, ExtractPOVAndTranspose, create_data_iterator


def validate(agent: BCOAgent, validation_dataset: Dataset, episode: int):
    batch_size = 64
    dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    idm_loss = torch.zeros(size=(len(dataloader), 1)).to(agent.device)
    idm_acc = torch.zeros_like(idm_loss).to(agent.device)
    policy_loss = torch.zeros_like(idm_loss).to(agent.device)
    policy_acc = torch.zeros_like(idm_loss).to(agent.device)
    cross_acc = torch.zeros_like(idm_loss).to(agent.device)
    for idx, batch in tqdm(enumerate(dataloader), desc=f'Validation (Episode {episode})'):
        obs, act, _, next_obs, _ = batch

        obs = obs.to(agent.device)
        act = act.to(agent.device)
        next_obs = next_obs.to(agent.device)

        idm_predict = agent.inv_dynamics(obs, next_obs)
        idm_loss[idx] = cross_entropy(idm_predict, act).detach()
        idm_acc[idx] = torch.sum(torch.argmax(idm_predict, dim=1) == act) / batch_size

        pol_predict = agent.policy(obs)
        policy_loss[idx] = cross_entropy(pol_predict, act).detach()
        policy_acc[idx] = torch.sum(torch.argmax(pol_predict, dim=1) == act) / batch_size
        cross_acc[idx] = torch.sum(torch.argmax(pol_predict, dim=1) == torch.argmax(idm_predict, dim=1)) / batch_size

    return (idm_loss.mean().item(),
            idm_acc.mean().item(),
            policy_loss.mean().item(),
            policy_acc.mean().item(),
            cross_acc.mean().item())


def create_datasets(data_iterator, val_size_ratio=0.1):
    buffer = ReplayBuffer()
    for experience in data_iterator:
        buffer.append(experience)
    val_size = int(val_size_ratio*len(buffer))
    return torch.utils.data.random_split(buffer, [len(buffer)-val_size, val_size])


def train_bco(hparams):
    torch.backends.cudnn.benchmark = True

    os.makedirs(hparams.logdir, exist_ok=True)
    wandb.init(
        project="master-thesis",
        group="BCO",
        job_type='train',
        dir=hparams.logdir,
        config=hparams,
        settings=wandb.Settings(start_method="fork") if os.name == 'posix' else None, )

    env = ExtractPOVAndTranspose(ActionShaping(gym.make("MineRLTreechop-v0")))
    env.seed(95)

    num_to_load = 50000 if hparams.fast_dev_run else None
    demo_iterator = iter(create_data_iterator(env, hparams.data_path, num_to_load))

    train_demonstrations, val_demonstrations = create_datasets(demo_iterator)

    agent = BCOAgent(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        observation_space=env.observation_space,
        action_space=env.action_space,
        feature_extractor=AtariCNN,
        q_network_hidden_size=512,
        batch_size=hparams.batch_size,
        pol_lr=hparams.plr,
        inv_dyn_lr=hparams.ilr,
        first_update_period=hparams.up,
        alpha=hparams.a,
        num_epochs_idm=hparams.idm_epoch)

    validate(agent, val_demonstrations, 0)
    agent.demonstrations = train_demonstrations
    for episode in tqdm(range(hparams.max_episodes), desc='Episode'):
        obs = env.reset()
        episode_reward = 0
        for _ in tqdm(range(hparams.episode_max_length), desc=f'Episode {episode} rollout'):
            action = agent.act(obs.copy())
            next_obs, reward, done, _ = env.step(action)
            agent.step(Experience(obs, action, None, next_obs, done))

            obs = next_obs
            episode_reward += reward

            if done:
                break
        if episode % 5 == 0:
            checkpoint_full_path = agent.save(wandb.run.dir)
            wandb.save(checkpoint_full_path)
        metrics = validate(
            agent=agent,
            validation_dataset=val_demonstrations,
            episode=episode)
        wandb.log({'reward': episode_reward,
                   'episode': episode,
                   'epoch': agent.epoch,
                   'global_step': agent.global_step,
                   'val_idm_loss': metrics[0],
                   'val_idm_acc': metrics[1],
                   'val_policy_loss': metrics[2],
                   'val_policy_acc': metrics[3],
                   'val_cross_acc': metrics[4]})
    wandb.finish()


if __name__ == "__main__":
    gym.logger.set_level(40)
    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', type=str, default="./logs", help='Root directory path for logs')
    parser.add_argument('--data_path', type=str, required=True, help='Dataset path')
    parser.add_argument('--max_episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--fast_dev_run', action="store_true", help='Run 1 epoch for test')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    parser.add_argument('--plr', type=float, default=0.007, help='Learning rate for policy')
    parser.add_argument('--ilr', type=float, default=0.007, help='Learning rate for inverse dynamics')
    parser.add_argument('--episode_max_length', type=int, default=10000, help='Max episode length')
    parser.add_argument('--up', type=int, default=100, help='Inverse dynamics and policy update period (in env steps)')
    parser.add_argument('--a', type=float, default=0.001, help='Alpha parameter in BCO(alpha)')
    parser.add_argument('--idm_epoch', type=int, default=1, help='Inverse dynamics number of epoch per update')

    args = parser.parse_args()

    train_bco(args)
