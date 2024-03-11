import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from hyperbolic_vae.datasets.cifar10_v1 import get_train_images, test_loader, train_loader, val_loader
from hyperbolic_vae.models.autoencoder_nonvariational import Autoencoder
from hyperbolic_vae.training.generate_callback import GenerateCallback


def train_cifar(device, latent_dim):
    FAKE_CHECKPOINT_PATH = "./fake_checkpoints"
    os.makedirs(FAKE_CHECKPOINT_PATH, exist_ok=True)
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=os.path.join(FAKE_CHECKPOINT_PATH, f"cifar10_{latent_dim}"),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=500,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            GenerateCallback(get_train_images(8), every_n_epochs=10),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(FAKE_CHECKPOINT_PATH, f"cifar10_{latent_dim}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = Autoencoder.load_from_checkpoint(pretrained_filename)
    else:
        model = Autoencoder(base_channel_size=32, latent_dim=latent_dim)
        trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result


if __name__ == "__main__":
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = {}
    for latent_dim in [64, 128, 256, 384]:
        model_ld, result_ld = train_cifar(device, latent_dim)
        model_dict[latent_dim] = {"model": model_ld, "result": result_ld}
