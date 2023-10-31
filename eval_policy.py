import os

import gym
import numpy as np
import torch

import wandb

from algorithms.model import AtariCNN
from algorithms.sqil import SQILAgent
from wrappers import ActionShaping, ExtractPOVAndTranspose

if __name__ == "__main__":

    # Create environment
    env = ExtractPOVAndTranspose(ActionShaping(gym.make("MineRLTreechop-v0")))

    # Download checkpoint locally (if not already cached)
    api = wandb.Api()
    run = api.run("stkovacevic94/master-thesis/36stdhtb")
    #file_dir = run.file('checkpoint_0.sqil').download(replace=True)
    #print(file_dir)
    print(run.config)

    # Load the agent
    agent = SQILAgent(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        observation_space=env.observation_space,
        action_space=env.action_space,
        feature_extractor=AtariCNN,
        q_network_hidden_size=512,
        batch_size=run.config['batch_size'],
        lr=run.config['learning_rate'],
        temp=run.config['temp'],
        gamma=run.config['gamma'],
        sync_rate=run.config['sync_rate']
    )
    agent.load('C:/Users/stkov/Downloads/checkpoint_106981.sqil')

    # Test the agent
    done = False
    obs = env.reset()
    total_reward = 0
    while not done:
        env.render()
        action = agent.act(obs.copy())
        obs, reward, done, _ = env.step(action)
        total_reward += reward
