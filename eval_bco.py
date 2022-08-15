import gym
import numpy as np
import torch

import wandb

from algorithms.model import AtariCNN
from algorithms.bco import BCOAgent
from wrappers import ActionShaping, ExtractPOVAndTranspose

if __name__ == "__main__":

    # Create environment
    env = ExtractPOVAndTranspose(ActionShaping(gym.make("MineRLTreechop-v0")))

    # Download checkpoint locally (if not already cached)
    api = wandb.Api()
    run = api.run('stkovacevic94/master-thesis/1yhstxpk')
    print(run.config)

    # Load the agent
    agent = BCOAgent(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        observation_space=env.observation_space,
        action_space=env.action_space,
        feature_extractor=AtariCNN,
        q_network_hidden_size=512,
        batch_size=run.config['batch_size'],
        pol_lr=run.config['plr'],
        inv_dyn_lr=run.config['ilr'],
        update_period=run.config['up']
    )
    agent.load(r'C:\Users\stkov\Downloads\checkpoint_1506000.bco')

    # Test the agent
    done = False
    obs = env.reset()
    total_reward = 0
    while not done:
        env.render()
        action = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)
        obs_t = torch.FloatTensor(obs).to(agent.device)
        next_obs_t = torch.FloatTensor(obs).to(agent.device)
        p_act = agent.policy(obs_t).detach().cpu().numpy()
        p_inv = agent.inv_dynamics(obs_t, next_obs_t).detach().cpu().numpy()
        print("Policy prediction")
        print(p_act)
        print("Inv dynamics prediction")
        print(p_inv)

        obs = next_obs
        total_reward += reward
