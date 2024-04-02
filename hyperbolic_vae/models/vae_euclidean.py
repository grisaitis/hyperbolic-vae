import io
import logging
import pathlib

import imageio.v3 as iio
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor, optim

logger = logging.getLogger(__name__)


class VAEEuclidean(nn.Module):
    def __init__(
        self,
        data_channels: int,
        hidden_size: int,
        latent_dim: int,
        act_fn: object = nn.GELU,
    ):
        super().__init__()
        c_hid = hidden_size  # c_hid means "number of hidden channels"
        self.encoder = nn.Sequential(
            # (batch_size, num_input_channels, 32, 32)
            nn.Conv2d(data_channels, c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            # (batch_size, c_hid, 16, 16)
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            # (batch_size, c_hid, 16, 16)
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            # (batch_size, 2 * c_hid, 8, 8)
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            # (batch_size, 2 * c_hid, 8, 8)
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            # (batch_size, 2 * c_hid, 4, 4)
            nn.Flatten(),
            # (batch_size, 2 * c_hid * 4 * 4)
        )
        self.mu = nn.Linear(2 * c_hid * 4 * 4, latent_dim)
        self.log_var = nn.Linear(2 * c_hid * 4 * 4, latent_dim)
        self.decoder = nn.Sequential(
            # (batch_size, latent_dim)
            nn.Linear(latent_dim, 2 * c_hid * 4 * 4),
            act_fn(),
            # (batch_size, 2 * c_hid * 4 * 4)
            nn.Unflatten(1, (2 * c_hid, 4, 4)),
            # (batch_size, 2 * c_hid, 4, 4)
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
                data_channels,
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self.sample(mu, log_var)
        x_hat = self.decoder(z)
        return mu, log_var, z, x_hat

    def sample(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z


class VAEEuclideanExperiment(pl.LightningModule):
    def __init__(
        self,
        data_channels: int = 3,
        hidden_size: int = 32,
        latent_dim: int = 2,
        act_fn: object = nn.GELU,
        width: int = 32,
        height: int = 32,
        beta: float = 1.0,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vae = VAEEuclidean(data_channels, hidden_size, latent_dim, act_fn)
        self.example_input_array = torch.zeros(2, data_channels, width, height)
        self.beta = beta
        self.lr = lr
        self.initialization = "kaiming"

    def forward(self, x):
        return self.vae(x)

    def loss(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        x, _ = batch
        mu, log_var, z, x_hat = self.forward(x)
        # Reconstruction loss
        loss_recon = nn.functional.mse_loss(x_hat, x, reduction="sum")
        # KL divergence loss
        loss_kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss_total = loss_recon + self.beta * loss_kld
        return {
            "loss_recon": loss_recon,
            "loss_kld": loss_kld,
            "loss_total": loss_total,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, patience=10, factor=0.5, verbose=True
        # )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=20,
            min_lr=5e-5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/loss_total",
        }

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss_dict = self.loss(batch)
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()})
        return loss_dict["loss_total"]

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss_dict = self.loss(batch)
        self.log_dict({f"val/{k}": v for k, v in loss_dict.items()})
        return loss_dict["loss_total"]

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss_dict = self.loss(batch)
        self.log_dict({f"test/{k}": v for k, v in loss_dict.items()})
        return loss_dict["loss_total"]


class VisualizeVAEEuclideanLatentSpace(Callback):
    def __init__(
        self,
        interpolate_epoch_interval: int = 1,
        range_start: int = -5,
        range_end: int = 5,
        steps: int = 11,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.interpolate_epoch_interval = interpolate_epoch_interval
        self.range_start = range_start
        self.range_end = range_end
        self.normalize = normalize
        self.steps = steps

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.interpolate_epoch_interval == 0:
            images = self.interpolate_latent_space(pl_module)
            images = torch.cat(images, dim=0)

            num_rows = self.steps
            grid = torchvision.utils.make_grid(images, nrow=num_rows, normalize=self.normalize)
            str_title = f"{pl_module.__class__.__name__}_latent_space"
            trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)

    def interpolate_latent_space(self, pl_module: VAEEuclideanExperiment) -> list[Tensor]:
        latent_dim = pl_module.hparams.latent_dim
        images = []
        with torch.no_grad():
            pl_module.eval()
            for z1 in np.linspace(self.range_start, self.range_end, self.steps):
                for z2 in np.linspace(self.range_start, self.range_end, self.steps):
                    # set all dims to zero
                    z = torch.zeros(1, latent_dim, device=pl_module.device)
                    # set the fist 2 dims to the value
                    z[0, 0] = torch.tensor(z1)
                    z[0, 1] = torch.tensor(z2)
                    # generate images
                    decoder_output = pl_module.decoder(z)
                    img = pl_module.transform_decoder_output(decoder_output)
                    images.append(img)
        pl_module.train()
        return images


class VisualizeVAEEuclideanValidationSetEncodings(Callback):
    def __init__(
        self,
        range_x: tuple = (-4, 4),
        range_y: tuple = (-4, 4),
        every_n_epochs: int = 1,
        path_write_image: pathlib.Path = None,
    ) -> None:
        super().__init__()
        self.range_x = range_x
        self.range_y = range_y
        self.every_n_epochs = every_n_epochs
        logger.info("path_write_image: %s", path_write_image)
        self.path_write_image = path_write_image

    def on_train_epoch_start(self, trainer: Trainer, vae_experiment: pl.LightningModule) -> None:
        if (trainer.current_epoch - 1) % self.every_n_epochs:
            return
        with torch.no_grad():
            vae_experiment.eval()
            data_loader_val = trainer.datamodule.val_dataloader()
            # compute encodings for validation set
            df = self.get_encodings_as_dataframe(vae_experiment, data_loader_val)
            # scatter plot, (-5, 5) x (-5, 5), color by label, legend outside on the right
            fig = self.create_plotly_figure(df)
            if self.path_write_image:
                fig.write_image(self.path_write_image, scale=2)
            buf = io.BytesIO()
            fig.write_image(
                buf,
                engine="kaleido",
                format="png",
                scale=2,
            )
            buf.seek(0)
            image_array = iio.imread(buf)
            image_array = image_array.transpose(2, 0, 1)
            str_title = f"{vae_experiment.__class__.__name__}_posterior_means"
            trainer.logger.experiment.add_image(str_title, image_array, global_step=trainer.global_step)
        vae_experiment.train()

    def create_plotly_figure(self, df: pd.DataFrame) -> go.Figure:
        # if df "label" column is numeric...
        try:
            df = df.astype({"label": "int"}).astype({"label": "str"})
        except ValueError:
            pass
        df = df.sort_values(by="label")
        fig = px.scatter(
            df,
            x="mu_0",
            y="mu_1",
            color="label",
            text="label",
            range_x=self.range_x,
            range_y=self.range_y,
            title="Latent space encoding of validation set",
            # color_discrete_sequence=px.colors.sequential.Plasma_r,
        )
        fig.update_traces(mode="text")
        fig.for_each_trace(lambda t: t.update(textfont_color=t.marker.color))
        fig.update_layout(width=600, height=600)
        # fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        # fig.for_each_trace(lambda t: t.update(mode="markers+text"))
        # fig.for_each_trace(lambda t: t.update(textfont_color="black", textposition="top center"))
        return fig

    def get_encodings(self, input_tensors: Tensor, vae_experiment: VAEEuclideanExperiment) -> np.ndarray:
        input_tensors = input_tensors.to(vae_experiment.device)
        e = vae_experiment.vae.encoder(input_tensors)
        mu = vae_experiment.vae.mu(e)
        return mu.cpu().numpy()

    def get_encodings_as_dataframe(self, vae_experiment, data_loader_val):
        df_list = []
        for i, batch in enumerate(data_loader_val):
            input_tensors, labels = batch
            mu = self.get_encodings(input_tensors, vae_experiment)
            df_batch = pd.DataFrame({"label": labels, "mu_0": mu[:, 0], "mu_1": mu[:, 1]})
            df_list.append(df_batch)
        df = pd.concat(df_list, ignore_index=True)
        return df
