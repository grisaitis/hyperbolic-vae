import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        base_channel_size: int,
        latent_dim: int,
        act_fn: object = nn.GELU,
    ):
        """
        Inputs:
            - num_input_channels: Number of input channels of the input image. For CIFAR-10, this is 3
            - base_channel_size: Number of channels of the first convolutional layer
            - latent_dim: Dimensionality of the latent representation z
            - act_fn: The activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size  # c_hid means "number of hidden channels"
        self.net = nn.Sequential(
            # (32, 32, num_input_channels) -> (16, 16, c_hid)
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            # (16, 16, c_hid) -> (16, 16, c_hid)
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            # (16, 16, c_hid) -> (8, 8, 2*c_hid)
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            # (8, 8, 2*c_hid) -> (8, 8, 2*c_hid)
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            # (8, 8, 2*c_hid) -> (4, 4, 2*c_hid)
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            # (4, 4, 2*c_hid) -> (4 * 4 * 2*c_hid,)
            nn.Flatten(),
            # (4 * 4 * 2*c_hid,) -> (latent_dim,)
            nn.Linear(4 * 4 * 2 * c_hid, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        base_channel_size: int,
        latent_dim: int,
        act_fn: object = nn.GELU,
    ):
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(nn.Linear(latent_dim, 4 * 4 * 2 * c_hid), act_fn())
        self.net = nn.Sequential(
            # (2 * c_hid, n, 4, 4) -> (2 * c_hid, n, 8, 8)
            nn.ConvTranspose2d(
                2 * c_hid,
                2 * c_hid,
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),
            act_fn(),
            # (2 * c_hid, n, 8, 8) -> (2 * c_hid, n, 8, 8)
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            # (2 * c_hid, n, 8, 8) -> (c_hid, n, 16, 16)
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            act_fn(),
            # (c_hid, n, 16, 16) -> (c_hid, n, 16, 16)
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            # (c_hid, n, 16, 16) -> (num_input_channels, n, 32, 32)
            nn.ConvTranspose2d(
                c_hid,
                num_input_channels,
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        num_input_channels: int = 3,
        width: int = 32,
        height: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(num_input_channels, base_channel_size, latent_dim)
        self.decoder = Decoder(num_input_channels, base_channel_size, latent_dim)
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch: torch.Tensor) -> torch.Tensor:
        x, _ = batch  # we are not interested in the labels
        x_hat = self.forward(x)
        # loss = F.mse_loss(x_hat, x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=(1, 2, 3)).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)
        return loss
