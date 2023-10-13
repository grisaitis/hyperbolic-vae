import os
import pathlib

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from hyperbolic_vae.config import CHECKPOINTS_PATH
from hyperbolic_vae.data.mnist_v2 import mnist_data_module
from hyperbolic_vae.models.vae_euclidean import VisualizeVAEEuclideanValidationSetEncodings
from hyperbolic_vae.models.vae_hyperbolic import (
    VAEHyperbolicExperiment,
    VisualizeVAEPoincareDiskValidationSetEncodings,
)
from hyperbolic_vae.training.generate_callback import GenerateCallback


def train_latent_dim(latent_dim: int = 64):
    vae_experiment = VAEHyperbolicExperiment(
        latent_dim=latent_dim,
        data_channels=1,
        width=32,
        height=32,
        beta=3.0,
        lr=1e-3,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINTS_PATH, f"mnist_poincare_{latent_dim}"),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=300,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            GenerateCallback.from_data_module(mnist_data_module, every_n_epochs=1),
            LearningRateMonitor("epoch"),
            VisualizeVAEPoincareDiskValidationSetEncodings(
                path_write_image=pathlib.Path(
                    "/home/jupyter/hyperbolic_rnaseq/figures/latent_space_poincare.png"
                ),
                range_x=(-1, 1),
                range_y=(-1, 1),
            ),
        ],
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None
    trainer.fit(
        vae_experiment,
        mnist_data_module,
    )


if __name__ == "__main__":
    pl.seed_everything(42)
    with torch.autograd.detect_anomaly(check_nan=True):
        train_latent_dim(2)
    # for latent_dim in [64, 2, 128, 256, 384]:
    # train_latent_dim(latent_dim)
