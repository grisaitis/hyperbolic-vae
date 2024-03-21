import logging
from abc import ABC, abstractmethod

import geoopt
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.distributions.negative_binomial import NegativeBinomial

import hyperbolic_vae
import hyperbolic_vae.layers
from hyperbolic_vae.distributions.wrapped_normal import WrappedNormal
from hyperbolic_vae.layers import ExpMap0

logger = logging.getLogger(__name__)


class BaseVAE(pl.LightningModule, ABC):
    def __init__(
        self,
        latent_dim: int,
        hidden_layer_dim: int,
        learning_rate: float,
        beta: float,
        activation_class: type,
    ):
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.learning_rate = learning_rate
        self.beta = beta
        self.activation_class = activation_class
        self._construct_encoder()
        self._construct_posterior_params()
        self._construct_decoder_first_op()
        self._construct_decoder()

    @abstractmethod
    def _construct_encoder(self):
        # differs for 1d vs 2d
        pass

    @abstractmethod
    def _construct_posterior_params(self):
        # differs for euclidean vs hyperbolic
        pass

    @abstractmethod
    def sample_posterior(self, mu: torch.Tensor, scale: torch.Tensor) -> torch.Tensor | geoopt.ManifoldTensor:
        # differs for euclidean vs hyperbolic
        pass

    @abstractmethod
    def sample_prior(self, samples_like: torch.Tensor) -> torch.Tensor | geoopt.ManifoldTensor:
        # differs for euclidean vs hyperbolic
        pass

    @abstractmethod
    def _construct_decoder_first_op(self):
        # differs for euclidean vs hyperbolic
        pass

    @abstractmethod
    def _construct_decoder(self):
        # uses self.decoder_first_op
        # differs for 1d vs 2d
        pass

    @abstractmethod
    def loss_reconstruction(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        # two approaches
        # 1. use MSE, comparing x and output
        # 2. use negative log likelihood, constructing a likelihood from output, and computing log_prob of that
        pass

    @abstractmethod
    def loss_kl(self, mu: torch.Tensor | geoopt.ManifoldTensor, scale: torch.Tensor) -> torch.Tensor:
        # has nice formula for euclidean
        # loss_kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        pass

    def forward(self, x: torch.Tensor):
        mu, scale = self.encode(x)
        z = self.sample_posterior(mu, scale)
        output = self.decoder(z)
        return mu, scale, z, output

    def loss(self, x: torch.Tensor):
        mu, scale, z, output = self.forward(x)
        loss_reconstruction = self.loss_reconstruction(x, output)
        loss_kl = self.loss_kl(mu, scale)
        loss_total = loss_reconstruction + self.beta * loss_kl
        return {
            "loss_reconstruction": loss_reconstruction,
            "loss_kl": loss_kl,
            "loss_total": loss_total,
        }

    def training_step(self, batch, batch_idx):
        loss_dict = self.loss(batch)
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()})
        if any(torch.isnan(v) for v in loss_dict.values()):
            logger.warning("NaN in loss dict: %s", loss_dict)
        return loss_dict["loss_total"]

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss_dict = self.loss(batch)
        self.log_dict({f"val/{k}": v for k, v in loss_dict.items()})
        return loss_dict["loss_total"]

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss_dict = self.loss(batch)
        self.log_dict({f"test/{k}": v for k, v in loss_dict.items()})
        return loss_dict["loss_total"]

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
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


class BaseVAEHyperpolic(BaseVAE):
    def __init__(
        self,
        manifold_curvature: float,
        **kwargs,
    ):
        self.manifold_curvature = manifold_curvature
        self.manifold = geoopt.PoincareBall(manifold_curvature)
        self.optimizer_class = geoopt.optim.RiemannianAdam
        super().__init__(**kwargs)

    def _construct_posterior_params(self):
        self.mu = nn.Sequential(
            nn.Linear(self.hidden_layer_dim, self.latent_dim),
            ExpMap0(self.manifold),
        )
        self.scale = nn.Sequential(
            nn.Linear(self.hidden_layer_dim, self.latent_dim),
            nn.Softplus(),
        )

    def sample_posterior(self, mu: torch.Tensor, scale: torch.Tensor) -> geoopt.ManifoldTensor:
        # dist = WrappedNormal(mu, scale, self.manifold)
        # z = dist.rsample()
        # or...
        z = self.manifold.wrapped_normal(*mu.shape, mean=mu, std=scale)
        return z

    def _construct_decoder_first_op(self):
        self.decoder_first_op = hyperbolic_vae.layers.GeodesicLayer(
            self.latent_dim, self.hidden_layer_dim, self.manifold
        )


class BaseVAEEuclidean(BaseVAE):
    def __init__(self, **kwargs):
        self.optimizer_class = torch.optim.Adam
        super().__init__(**kwargs)

    def _construct_posterior_params(self):
        self.mu = nn.Linear(self.hidden_layer_dim, self.latent_dim)
        self.log_var = nn.Linear(self.hidden_layer_dim, self.latent_dim)

    def sample_posterior(self, mu: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        dist = torch.distributions.Normal(mu, scale)
        z = dist.rsample()
        return z


class VAE1DMixin(BaseVAE, ABC):
    def _construct_encoder(self):
        self.encoder = nn.Sequential(
            nn.Linear(self.input_data_shape, self.hidden_layer_dim),
            self.activation_class(),
        )

    def _construct_decoder(self):
        self.decoder = nn.Sequential(
            self.decoder_first_op,
            self.activation_class(),
            nn.Linear(self.hidden_layer_dim, self.input_data_shape),
        )


class VAEImageMixin(BaseVAE, ABC):
    def __init__(self,
                 data_channels: int,
                 image_shape: tuple[int, int],
                    **kwargs):
        self.data_channels = data_channels
        self.image_shape = image_shape
    
    def _construct_encoder(self):
        data_channels = self.data_channels
        c_hid = self.hidden_layer_dim / 8
        act_fn = self.activation_class
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
    
    def _construct_decoder(self):
        data_channels = self.data_channels
        c_hid = self.hidden_layer_dim / 8
        latent_dim = self.latent_dim
        act_fn = self.activation_class
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