# make dataset (torch.utils.data.Dataset)
# - specify cache paths, transforms
# - note: torchvision.datasets classes (e.g. CIFAR10) are subclasses of this
# - question: can i use a gcp bucket as a cache path?
# make data module (pl.LightningDataModule)
# - encapsulates data loader factories
# - implements train_dataloader, val_dataloader, test_dataloader
# make model (torch.nn.Module)
# - specifying latent dimensions, shapes of layers, other hyperparameters
# - encapsulates the model definition
# - implements forward
# make experiment (pl.LightningModule) from model
# - specifying the model, any settings for optimization and loss
# - encapsulates how optimization is performed
# - implements training_step, configure_optimizers
# make trainer (pl.Trainer)
# - specifying callbacks, directory for checkpoints, etc
# call trainer with experiment, data module (`trainer.fit(experiment, data_module)`)
# - does this make a data loader? i think so. pl.LightningDataModule has some data loader factory methods


import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from hyperbolic_vae.config import CHECKPOINTS_PATH
from hyperbolic_vae.datasets.cifar10_v2 import cifar10_data_module
from hyperbolic_vae.models.vae_euclidean import VAEEuclideanExperiment, VisualizeVAEEuclideanLatentSpace
from hyperbolic_vae.training.generate_callback import GenerateCallback


def train_latent_dim(latent_dim: int = 64):
    vae_experiment = VAEEuclideanExperiment(
        data_channels=3,
        hidden_size=32,
        latent_dim=latent_dim,
        width=32,
        height=32,
        beta=1.0,
        lr=1e-3,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINTS_PATH, f"cifar10_{latent_dim}"),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=300,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            GenerateCallback.from_data_module(cifar10_data_module, every_n_epochs=1),
            LearningRateMonitor("epoch"),
            VisualizeVAEEuclideanLatentSpace(),
        ],
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None
    trainer.fit(vae_experiment, cifar10_data_module)


if __name__ == "__main__":
    pl.seed_everything(42)
    train_latent_dim(128)
    # for latent_dim in [64, 2, 128, 256, 384]:
    # train_latent_dim(latent_dim)
