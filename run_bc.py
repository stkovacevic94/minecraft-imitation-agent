import argparse
import os
import logging

import gym
import wandb
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

from algorithms.bc import BCAgent
from data import ReplayBuffer, load_expert_demonstrations
from model import ImpalaResNetCNN, Model, AtariCNN
from wrappers import ActionShaping, ExtractPOVTransposeAndNormalize


def main(hparams):
    deterministic = False
    if hparams.seed is not None:
        seed_everything(hparams.seed, workers=True)
        deterministic = True

    env = ExtractPOVTransposeAndNormalize(ActionShaping(gym.make("MineRLTreechop-v0")))
    expert_demonstrations = ReplayBuffer(capacity=500000)
    load_expert_demonstrations(expert_demonstrations, env, hparams.data_path, hparams.fast_dev_run)

    policy_network = Model(
        num_actions=env.action_space.n,
        image_channels=3,
        cnn_module=AtariCNN,
        hidden_size=512)
    agent = BCAgent(
        policy=policy_network,
        env=env,
        expert_demonstrations=expert_demonstrations,
        batch_size=hparams.batch_size,
        lr=hparams.lr,
        env_validation_period=500)

    os.makedirs(hparams.logdir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="master-thesis",
        group="BC",
        job_type='train',
        log_model='all',
        save_dir=hparams.logdir,
        settings=wandb.Settings(start_method="fork") if os.name == 'posix' else None)
    wandb_logger.watch(policy_network, log='all')

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=hparams.max_epochs,
        logger=wandb_logger,
        track_grad_norm=2,
        callbacks=[ModelSummary(max_depth=1), ModelCheckpoint(save_top_k=3, monitor="reward")],
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
    parser.add_argument('--max_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--fast_dev_run', action="store_true", help='Run 1 epoch for test')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')

    args = parser.parse_args()

    main(args)
