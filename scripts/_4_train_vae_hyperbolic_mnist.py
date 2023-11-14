from collections import defaultdict

import pytorch_lightning as pl
import torch
import torch.optim as optim

from hyperbolic_vae.data.mnist_v2 import mnist_data_module
from hyperbolic_vae.models.vae_hyperbolic import ImageVAEHyperbolic
from hyperbolic_vae.training.old_pvae_train import train

if __name__ == "__main__":
    pl.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageVAEHyperbolic().to(device)
    agg = defaultdict(list)
    train_loader = mnist_data_module.train_dataloader()
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
        amsgrad=True,
        betas=(0.9, 0.999),  # beta1=0.9, beta2=0.999
    )
    with torch.autograd.detect_anomaly(check_nan=True):
        model.init_last_layer_bias(train_loader)
        train(
            model,
            train_loader,
            0,
            agg,
            device=device,
            optimizer=optimizer,
        )
    # for latent_dim in [64, 2, 128, 256, 384]:
    # train_latent_dim(latent_dim)
