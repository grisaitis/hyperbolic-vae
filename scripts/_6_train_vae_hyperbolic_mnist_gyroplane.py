import logging
import os
import pathlib

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from hyperbolic_vae.config import CHECKPOINTS_PATH
from hyperbolic_vae.data.mnist_v2 import mnist_data_module
from hyperbolic_vae.models.vae_hyperbolic_gyroplane_decoder import (
    VAEHyperbolicGyroplaneDecoder,
    VisualizeEncodingsValidationSet,
)
from hyperbolic_vae.models.vae_hyperbolic import VisualizeVAEPoincareDiskValidationSetEncodings
from hyperbolic_vae.training.generate_callback import GenerateCallback
from hyperbolic_vae.util import ColoredFormatter


def train_latent_dim(latent_dim: int = 64):
    manifold_curvature = 1.0
    vae_experiment = VAEHyperbolicGyroplaneDecoder(
        data_shape=torch.Size([1, 32, 32]),
        latent_dim=latent_dim,
        manifold_curvature=manifold_curvature,
        beta=1.0,
        lr=1e-3,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINTS_PATH, f"mnist_poincare_{latent_dim}_gyroplane"),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=300,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, every_n_epochs=10),
            GenerateCallback.from_data_module(mnist_data_module, every_n_epochs=1),
            LearningRateMonitor("epoch"),
            VisualizeEncodingsValidationSet(
                path_write_image=pathlib.Path(
                    "/home/jupyter/hyperbolic-vae/figures/latent_space_poincare_gyroplane.png"
                ),
                range_x=(-(manifold_curvature**-0.5), manifold_curvature**-0.5),
                range_y=(-(manifold_curvature**-0.5), manifold_curvature**-0.5),
                every_n_epochs=10,
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
    logging.getLogger("hyperbolic_vae").setLevel("INFO")
    logging.getLogger("pvae").setLevel("DEBUG")
    sh = logging.StreamHandler()
    sh.setFormatter(ColoredFormatter("%(asctime)s %(name)s %(funcName)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(sh)
    pl.seed_everything(42)
    with torch.autograd.detect_anomaly(check_nan=True):
        train_latent_dim(2)
