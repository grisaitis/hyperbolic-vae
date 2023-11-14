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
from hyperbolic_vae.training.trainer_mnist import make_trainer_hyperbolic
from hyperbolic_vae.util import ColoredFormatter

vae_experiment = VAEHyperbolicExperiment(
    image_shape=(1, 32, 32),
    latent_dim=2,
    manifold_curvature=1.4,
    encoder_last_layer_module="pvae_mobius",
    decoder_first_layer_module="geoopt_gyroplane",
    beta=1.0,
    lr=1e-3,
    loss_recon="mse",
)

trainer = make_trainer_hyperbolic(vae_experiment.manifold_curvature)


if __name__ == "__main__":
    logging.getLogger("hyperbolic_vae").setLevel("DEBUG")
    logging.getLogger("pvae").setLevel("DEBUG")
    sh = logging.StreamHandler()
    sh.setFormatter(ColoredFormatter("%(asctime)s %(name)s %(funcName)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(sh)

    pl.seed_everything(42)
    with torch.autograd.detect_anomaly(check_nan=True):
        trainer.fit(
            vae_experiment,
            mnist_data_module,
        )
