import logging
import os
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import Dataset

from hyperbolic_vae.config import CHECKPOINTS_PATH
from hyperbolic_vae.datasets import jerby_arnon
from hyperbolic_vae.models.vae_hyperbolic_gyroplane_decoder import VisualizeEncodingsValidationSet
from hyperbolic_vae.models.vae_hyperbolic_rnaseq import VAEHyperbolicRNASeq
from hyperbolic_vae.training.generate_callback import GenerateCallback
from hyperbolic_vae.util import ColoredFormatter

logger = logging.getLogger(__name__)


def train(
    dataset: Dataset,
    manifold_curvature: float,
    latent_dim: int,
    batch_size: int,
):
    input_data_shape = dataset[0]["rnaseq"].shape
    logger.info(f"input_data_shape from dataset: {input_data_shape}")
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    data_module = pl.LightningDataModule.from_datasets(
        train_dataset=dataset_train,
        val_dataset=dataset_val,
        test_dataset=dataset_test,
        batch_size=batch_size,
        num_workers=0,
    )
    vae = VAEHyperbolicRNASeq(
        data_shape=input_data_shape,
        latent_dim=latent_dim,
        manifold_curvature=manifold_curvature,
        beta=1.0,
        lr=1e-3,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINTS_PATH, f"rnaseq_hyperbolic_{latent_dim}"),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=300,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, every_n_epochs=10),
            GenerateCallback.from_data_module(data_module, every_n_epochs=1),
            LearningRateMonitor("epoch"),
            VisualizeEncodingsValidationSet(
                path_write_image=Path("/home/jupyter/hyperbolic-vae/figures/latent_space_poincare_gyroplane.png"),
                range_x=(-(manifold_curvature**-0.5), manifold_curvature**-0.5),
                range_y=(-(manifold_curvature**-0.5), manifold_curvature**-0.5),
                every_n_epochs=10,
            ),
        ],
    )

    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None
    trainer.fit(vae, data_module)


if __name__ == "__main__":
    logging.getLogger("hyperbolic_vae").setLevel("INFO")
    logging.getLogger("pvae").setLevel("DEBUG")
    sh = logging.StreamHandler()
    sh.setFormatter(ColoredFormatter("%(asctime)s %(name)s %(funcName)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(sh)

    pl.seed_everything(42)

    jerby_arnon_dataset = jerby_arnon.get_pytorch_dataset()
    train(
        dataset=jerby_arnon_dataset,
        manifold_curvature=1.0,
    )
