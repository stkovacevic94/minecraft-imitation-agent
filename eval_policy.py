import os
import logging

import gym
import numpy as np
import torch

import wandb
from algorithms.bc import BCAgent
from model import AtariCNN, Model
from wrappers import ActionShaping, ExtractPOVTransposeAndNormalize

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Create environment
    env = ExtractPOVTransposeAndNormalize(ActionShaping(gym.make("MineRLTreechop-v0")))

    # Download checkpoint locally (if not already cached)
    run = wandb.init()
    artifact = run.use_artifact('stkovacevic94/master-thesis/model-eb9vlat0:v2', type='model')
    artifact_dir = artifact.download()

    # Load the agent
    model = Model(env.action_space.n, 3, AtariCNN, 512)
    agent = BCAgent.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"), policy=model, env=env, expert_demonstrations=None)

    # Test the agent
    done = False
    obs = env.reset()
    total_reward = 0
    while not done:
        env.render()
        action = agent(torch.tensor(np.expand_dims(obs, axis=0), dtype=torch.float32))
        print(action)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
