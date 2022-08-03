import os
import logging

import gym
import numpy as np
import torch

import wandb
from algorithms.bc import BCAgent
from model import ImpalaResNetCNN, PolicyNetwork
from wrappers import ActionShaping, ExtractPOVTransposeAndNormalize

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Create environment
    env = ExtractPOVTransposeAndNormalize(ActionShaping(gym.make("MineRLTreechop-v0")))

    # Download checkpoint locally (if not already cached)
    run = wandb.init()
    artifact = run.use_artifact('skovacevic94/master-thesis/model-3dz5e3w7:v0', type='model')
    artifact_dir = artifact.download()

    # Load the agent
    model = PolicyNetwork(env.action_space.n, 3, ImpalaResNetCNN, 512)
    agent = BCAgent.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"), policy=model, env=env)

    # Test the agent
    done = False
    obs = env.reset()
    total_reward = 0
    while not done:
        env.render()
        action_distribution = agent(torch.tensor(np.expand_dims(obs, axis=0), dtype=torch.float32))
        action = torch.argmax(action_distribution).item()
        print(action_distribution)
        print(action)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
