import argparse
import os
import logging

import minerl
import gym
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from bc import BCAgent
from dataset import MineRLDataset
from model import PolicyModel, ImpalaResNetCNN
from wrappers import ActionShaping, ActionManager, ObservationShaping


def main(hparams):
    deterministic = False
    if hparams.seed is not None:
        seed_everything(hparams.seed, workers=True)
        deterministic = True

    action_manager = ActionManager()
    env = ObservationShaping(ActionShaping(gym.make("MineRLTreechop-v0"), action_manager))

    # TODO: Improve action_manager dependency with dataset
    # TODO: Add parameter for environment
    dataset = MineRLDataset("MineRLTreechop-v0", action_manager.action_id, hparams.data_path)
    data_loader = DataLoader(dataset, batch_size=hparams.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = PolicyModel(num_actions=env.action_space.n, image_channels=3, cnn_module=ImpalaResNetCNN, hidden_size=512)
    agent = BCAgent(model, env, hparams.lr)

    os.makedirs(hparams.logdir, exist_ok=True)
    # TODO: group should be parameter
    wandb_logger = WandbLogger(
        project="master-thesis",
        group="BC-Treechop",
        job_type='train',
        log_model='all',
        save_dir=hparams.logdir)
    wandb_logger.watch(model, log='all')

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=hparams.max_epochs,
        val_check_interval=0.2,
        logger=wandb_logger,
        track_grad_norm=2,
        callbacks=[ModelSummary(max_depth=1), ModelCheckpoint(save_top_k=3, monitor="val_reward")],
        deterministic=deterministic,
        fast_dev_run=hparams.fast_dev_run
    )

    trainer.fit(agent, data_loader)


def add_training_specific_args(parent_parser: argparse.ArgumentParser):
    parser = parent_parser.add_argument_group("Training")
    parser.add_argument('--max_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--fast_dev_run', action="store_true", help='Run 1 epoch for test')
    return parent_parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()

    # PROGRAM level args
    parser.add_argument('--seed', type=int, default=None, help='Specific seed for reproducibility')
    parser.add_argument('--logdir', type=str, default="./logs", help='Root directory path for logs')
    # TRAINER level args
    parser = add_training_specific_args(parser)
    # MODEL level args
    parser = BCAgent.add_model_specific_args(parser)
    # DATA level args
    parser = MineRLDataset.add_data_specific_args(parser)

    args = parser.parse_args()

    main(args)
