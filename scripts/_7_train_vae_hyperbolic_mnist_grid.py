import itertools
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
from hyperbolic_vae.datasets.mnist_v2 import mnist_data_module
from hyperbolic_vae.models.vae_hyperbolic import (
    VAEHyperbolicExperiment,
    VisualizeVAEPoincareDiskValidationSetEncodings,
)
from hyperbolic_vae.training.generate_callback import GenerateCallback
from hyperbolic_vae.training.trainer_mnist import make_trainer_hyperbolic
from hyperbolic_vae.util import ColoredFormatter


def run_experiment(
    latent_dim: int,
    beta: float,
    curvature: float,
    encoder_last_layer_module: str,
    decoder_first_layer_module: str,
):
    trainer = make_trainer_hyperbolic(curvature)
    model = VAEHyperbolicExperiment(
        image_shape=(1, 32, 32),
        latent_dim=latent_dim,
        manifold_curvature=curvature,
        encoder_last_layer_module=encoder_last_layer_module,
        decoder_first_layer_module=decoder_first_layer_module,
        beta=beta,
        lr=1e-3,
        loss_recon="mse",
    )
    with torch.autograd.detect_anomaly(check_nan=True):
        trainer.fit(model, mnist_data_module)
    best_vae_experiment = VAEHyperbolicExperiment.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(best_vae_experiment, mnist_data_module)


if __name__ == "__main__":
    logging.getLogger("hyperbolic_vae").setLevel("DEBUG")
    logging.getLogger("pvae").setLevel("DEBUG")
    sh = logging.StreamHandler()
    sh.setFormatter(ColoredFormatter("%(asctime)s %(name)s %(funcName)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(sh)

    # pl.seed_everything(42)
    curvatures = [0.5, 1.0, 2.0]
    betas = [1.0, 2.0]
    latent_dims = [2, 5]
    encoder_last_layer_modules = ["pvae_mobius"]
    decoder_first_layer_modules = ["geoopt_gyroplane"]

    combinations = itertools.product(
        curvatures,
        betas,
        latent_dims,
        encoder_last_layer_modules,
        decoder_first_layer_modules,
    )
    for curvature, beta, latent_dim, encoder_last_layer_module, decoder_first_layer_module in combinations:
        try:
            run_experiment(
                latent_dim=latent_dim,
                beta=beta,
                curvature=curvature,
                encoder_last_layer_module=encoder_last_layer_module,
                decoder_first_layer_module=decoder_first_layer_module,
            )
        except Exception as e:
            # print exception, traceback and continue
            print(e)
            import traceback

            traceback.print_exc()
            continue
