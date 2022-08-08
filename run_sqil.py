import argparse
import os
import logging

import gym
import wandb
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

from algorithms.sqil import SQILAgent
from data import ReplayBuffer, load_expert_demonstrations
from model import ImpalaResNetCNN, Model, AtariCNN
from wrappers import ActionShaping, ExtractPOVTransposeAndNormalize


def main(hparams):
    deterministic = False
    if hparams.seed is not None:
        seed_everything(hparams.seed, workers=True)
        #deterministic = True

    env = ExtractPOVTransposeAndNormalize(ActionShaping(gym.make("MineRLTreechop-v0")))
    expert_demonstrations = ReplayBuffer(capacity=500000)
    load_expert_demonstrations(expert_demonstrations, env, hparams.data_path, hparams.fast_dev_run)

    q_network = Model(
        num_actions=env.action_space.n,
        image_channels=3,
        cnn_module=AtariCNN,
        hidden_size=512)

    agent = SQILAgent(q_network=q_network,
                      env=env,
                      expert_demonstrations=expert_demonstrations,
                      batch_size=hparams.batch_size,
                      episode_max_length=hparams.episode_max_length,
                      lr=hparams.learning_rate,
                      alpha=hparams.alpha,
                      gamma=hparams.gamma,
                      sync_rate=hparams.sync_rate)

    os.makedirs(hparams.logdir, exist_ok=True)
    wandb_logger = WandbLogger(
        settings=wandb.Settings(start_method="fork") if os.name == 'posix' else None,
        project="master-thesis",
        group="SQIL",
        job_type='train',
        log_model='all',
        save_dir=hparams.logdir)
    wandb_logger.watch(q_network, log='all')

    trainer = pl.Trainer(
        gpus=1,
        max_steps=hparams.max_steps,
        logger=wandb_logger,
        track_grad_norm=2,
        callbacks=[ModelSummary(max_depth=1), ModelCheckpoint(every_n_train_steps=10000)],
        deterministic=deterministic,
        fast_dev_run=hparams.fast_dev_run
    )

    trainer.fit(agent)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=None, help='Specific seed for reproducibility')
    parser.add_argument('--logdir', type=str, default="./logs", help='Root directory path for logs')
    parser.add_argument('--data_path', type=str, required=True, help='Dataset path')
    parser.add_argument('--max_steps', type=int, default=10**6, help='Number of steps to train')
    parser.add_argument('--fast_dev_run', action="store_true", help='Run 1 epoch for test')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--episode_max_length', type=int, default=3000, help='Max episode length')
    parser.add_argument('--alpha', type=float, default=4, help='Soft Q Learning alpha parameter')
    parser.add_argument('--gamma', type=float, default=0.9, help='MDP discount parameter gamma')
    parser.add_argument('--sync_rate', type=int, default=50, help='Rate at which target network and online network sync')

    args = parser.parse_args()

    main(args)
