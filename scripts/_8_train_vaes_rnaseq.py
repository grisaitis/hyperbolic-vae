import logging
import os
from pathlib import Path

import geoopt
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import Dataset

from hyperbolic_vae.config import CHECKPOINTS_PATH, PROJECT_ROOT
from hyperbolic_vae.datasets import jerby_arnon, mnist_v2
from hyperbolic_vae.models import vae_one_b
from hyperbolic_vae.models.vae_euclidean import VisualizeVAEEuclideanLatentSpace
from hyperbolic_vae.models.vae_hyperbolic_gyroplane_decoder import VisualizeEncodingsValidationSet
from hyperbolic_vae.models.vae_hyperbolic_rnaseq import VAEHyperbolicRNASeq
from hyperbolic_vae.training.generate_callback import GenerateCallback
from hyperbolic_vae.util import ColoredFormatter

logger = logging.getLogger(__name__)


def train(
    data_module: pl.LightningDataModule,
    latent_dim: int,
    latent_curvature: float,
    prior_scale: float,
    posterior_scale: str,
    hidden_layer_dim: int,
    learning_rate: float,
    beta: float,
    kl_loss_method: str,
    max_epochs: int,
    last_activation: str,
    loss_recon_method: str,
):
    input_data_shape = next(iter(data_module.train_dataloader()))[0].shape[1:]
    logger.info(f"input_data_shape from dataset: {input_data_shape}")
    vae = vae_one_b.VAE(
        input_size=input_data_shape,
        hidden_layer_dim=hidden_layer_dim,
        latent_dim=latent_dim,
        latent_curvature=latent_curvature,
        prior_scale=prior_scale,
        posterior_scale=posterior_scale,
        learning_rate=learning_rate,
        beta=beta,
        kl_loss_method=kl_loss_method,
        activation_class=nn.GELU,
        last_activation=last_activation,
        loss_recon_method=loss_recon_method,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if latent_curvature:
        latent_range = (-(latent_curvature**-0.5), latent_curvature**-0.5)
    else:
        latent_range = (-4, 4)
    callbacks = [
        ModelCheckpoint(save_weights_only=True, every_n_epochs=10),
        LearningRateMonitor("epoch"),
    ]
    is_image = input_data_shape == torch.Size([1, 28, 28])
    logger.info("is_image: %s", is_image)
    logger.info("input_data_shape[:2]: %s", input_data_shape[:2])
    logger.info("input_data_shape[:2] == (28, 28): %s", input_data_shape[:2] == (28, 28))
    if is_image:
        logger.info("Adding image callbacks")
        callbacks.append(GenerateCallback.from_data_module(data_module, every_n_epochs=1))
        callbacks.append(
            VisualizeVAEEuclideanLatentSpace(range_start=latent_range[0], range_end=latent_range[1], steps=21)
        )
    callbacks.append(VisualizeEncodingsValidationSet(latent_range, latent_range))
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINTS_PATH, "vae_b_rnaseq"),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=max_epochs,
        callbacks=callbacks,
    )

    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None
    trainer.fit(vae, data_module)


if __name__ == "__main__":
    logger.setLevel("DEBUG")
    logging.getLogger("hyperbolic_vae").setLevel("DEBUG")
    logging.getLogger("hyperbolic_vae.datasets.jerby_arnon").setLevel("DEBUG")
    logging.getLogger("hyperbolic_vae.layers").setLevel("INFO")
    logging.getLogger("hyperbolic_vae.models.vae_one_b").setLevel("INFO")
    sh = logging.StreamHandler()
    sh.setFormatter(ColoredFormatter("%(asctime)s %(name)s %(funcName)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(sh)

    pl.seed_everything(42)

    on_dataset = "jerby_arnon"
    # on_dataset = "mnist"

    if on_dataset == "mnist":
        data_module = mnist_v2.make_data_module(batch_size=64, num_workers=0)
        # last_activation, loss_recon = "none", "relaxed bernoulli"
        # last_activation, loss_recon = "sigmoid", "binary_cross_entropy"
        last_activation, loss_recon = "none", "binary_cross_entropy_with_logits"
    elif on_dataset == "jerby_arnon":
        # last_activation, loss_recon, rnaseq_normalize_method = "sigmoid", "relaxed bernoulli", "sum_to_million"
        # last_activation, loss_recon, rnaseq_normalize_method = "softplus", "MSE", "sum_to_million"
        last_activation, loss_recon, rnaseq_normalize_method = "none", "MSE", "z_score"
        jerby_arnon_dataset = jerby_arnon.get_pytorch_dataset(rnaseq_normalize_method)
        # jerby_arnon_dataset = jerby_arnon.get_fake_dataset(100, 200, "sum_to_one")
        # jerby_arnon_dataset = jerby_arnon.get_subset_jerby_arnon_dataset(7000, 10, "sum_to_one")
        data_module = jerby_arnon.make_rnaseq_data_module(jerby_arnon_dataset, batch_size=64, num_workers=0)
    with torch.autograd.detect_anomaly(check_nan=True):
        train(
            data_module=data_module,
            latent_dim=2,
            latent_curvature=1.0,
            prior_scale=2.0,
            posterior_scale="learned",
            hidden_layer_dim=100,  # default for PVAE was 100, 800 in demo
            learning_rate=1e-3,
            beta=0.5,
            kl_loss_method="logmap0_analytic",
            max_epochs=500,
            last_activation=last_activation,
            loss_recon_method=loss_recon,
        )
