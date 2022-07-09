import os
import logging

import minerl
import gym
import numpy as np
import torch

import wandb
from bc import BCAgent
from src.policies import CNNPolicy
from wrappers import ActionShaping, ActionManager, ObservationShaping

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Create environment
    env = ObservationShaping(ActionShaping(gym.make("MineRLTreechop-v0"), ActionManager()))

    # Download checkpoint locally (if not already cached)
    run = wandb.init()
    artifact = run.use_artifact('skovacevic94/master-thesis/model-3dz5e3w7:v0', type='model')
    artifact_dir = artifact.download()

    # Load the agent
    policy = CNNPolicy(env.action_space.n)
    agent = BCAgent.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"), policy=policy, env=env)

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
