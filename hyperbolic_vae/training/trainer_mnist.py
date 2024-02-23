import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch import nn

from hyperbolic_vae.config import CHECKPOINTS_PATH
from hyperbolic_vae.data.mnist_v2 import mnist_data_module
from hyperbolic_vae.models.vae_hyperbolic import VisualizeVAEPoincareDiskValidationSetEncodings
from hyperbolic_vae.training.generate_callback import GenerateCallback


def make_trainer_hyperbolic(curvature: float) -> pl.Trainer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINTS_PATH, "mnist_hyperbolic"),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=300,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, save_top_k=1, monitor="val/loss_total", save_last=True),
            GenerateCallback.from_data_module(mnist_data_module, every_n_epochs=1),
            LearningRateMonitor("epoch"),
            VisualizeVAEPoincareDiskValidationSetEncodings(
                range_x=(-(curvature**-0.5), curvature**-0.5),
                range_y=(-(curvature**-0.5), curvature**-0.5),
            ),
            EarlyStopping("val/loss_total", patience=10, verbose=True),
        ],
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None
    return trainer


"""
trainers...
- consumed in
    - training scripts
    - evaluation scripts
- need to differ by
    - geometry
    - data modules
- but otherwise many things are the same - GPU device, checkpoint directory, etc

currently...
- trainers are defined in training scripts
- but evaluation scripts also need them...
    - what needs to be different? what can be shared?
"""
