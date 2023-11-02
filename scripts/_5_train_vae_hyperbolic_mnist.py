import logging
import os
import pathlib

import pvae.ops.manifold_layers
import pytorch_lightning as pl
import torch
from geoopt.layers.stereographic import Distance2StereographicHyperplanes
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch import nn

from hyperbolic_vae.config import CHECKPOINTS_PATH
from hyperbolic_vae.data.mnist_v2 import mnist_data_module
from hyperbolic_vae.models.vae_hyperbolic_linear_wrapped import (
    VAEHyperbolicExperiment,
    VisualizeVAEPoincareDiskValidationSetEncodings,
)
from hyperbolic_vae.training.generate_callback import GenerateCallback
from hyperbolic_vae.util import ColoredFormatter


def train_latent_dim(latent_dim: int = 64):
    curvature = 1.4
    vae_experiment = VAEHyperbolicExperiment(
        image_shape=(1, 32, 32),
        latent_dim=latent_dim,
        manifold_curvature=curvature,
        encoder_last_layer_module="pvae_mobius",
        decoder_first_layer_module="geoopt_gyroplane",
        beta=1.0,
        lr=1e-3,
        loss_recon="mse",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINTS_PATH, "mnist"),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=300,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            GenerateCallback.from_data_module(mnist_data_module, every_n_epochs=1),
            LearningRateMonitor("epoch"),
            VisualizeVAEPoincareDiskValidationSetEncodings(
                range_x=(-curvature**-0.5, curvature**-0.5),
                range_y=(-curvature**-0.5, curvature**-0.5),
            ),
            EarlyStopping("val/loss_total", patience=10, verbose=True),
        ],
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None
    trainer.fit(
        vae_experiment,
        mnist_data_module,
    )


if __name__ == "__main__":
    logging.getLogger("hyperbolic_vae").setLevel("DEBUG")
    logging.getLogger("pvae").setLevel("DEBUG")
    sh = logging.StreamHandler()
    sh.setFormatter(ColoredFormatter("%(asctime)s %(name)s %(funcName)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(sh)
    pl.seed_everything(42)
    with torch.autograd.detect_anomaly(check_nan=True):
        train_latent_dim(2)
    # for latent_dim in [64, 2, 128, 256, 384]:
    # train_latent_dim(latent_dim)
