import logging

import geoopt
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.distributions.negative_binomial import NegativeBinomial

from hyperbolic_vae.distributions.wrapped_normal import WrappedNormal
from hyperbolic_vae.layers import ExpMap0

"""
action plan
- get something working
- figure out scvi architecture... what was the net architecture in terms of layers, activations, etc.?
"""

logger = logging.getLogger(__name__)


class VAEHyperbolicRNASeq(pl.LightningModule):
    def __init__(
        self,
        input_data_shape: torch.Size,
        latent_dim: int,
        manifold_curvature: float,
        hidden_layer_dim: int,
        learning_rate: float,
        beta: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.zeros(input_data_shape)
        self.lr = learning_rate
        self.beta = beta
        self.latent_dim = latent_dim
        self.manifold = geoopt.PoincareBall(c=manifold_curvature)
        self.prior_scale = 1.0
        self.encoder = nn.Sequential(
            nn.Linear(input_data_shape.numel(), hidden_layer_dim),
            nn.GELU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(hidden_layer_dim, latent_dim),
            ExpMap0(self.manifold),
        )
        self.scale = nn.Sequential(nn.Linear(hidden_layer_dim, latent_dim), nn.Softplus())
        self.decoder = nn.Sequential(
            geoopt.layers.stereographic.Distance2StereographicHyperplanes(
                latent_dim,
                hidden_layer_dim,
                ball=self.manifold,
            ),
            nn.GELU(),
            nn.Linear(hidden_layer_dim, input_data_shape.numel()),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        scale = self.scale(x)
        qz_x = WrappedNormal(mu, scale, self.manifold)
        z = qz_x.rsample(torch.Size([1])).squeeze(0)
        x_hat = self.decoder(z)
        return mu, scale, z, x_hat

    def loss(self, batch: torch.Tensor) -> dict:
        x = batch["rnaseq"]
        batch_shape = x.shape[0:1]
        logger.debug("x.shape: %s", x.shape)
        mu, scale, z, x_hat = self.forward(x)
        logger.info("mu mean, stddev: %s, %s", mu.mean(), mu.std())
        logger.info("scale mean, stddev: %s, %s", scale.mean(), scale.std())
        logger.debug("mu.shape: %s", mu.shape)
        logger.debug("scale.shape: %s", scale.shape)
        logger.debug("z.shape: %s", z.shape)
        logger.debug("x_hat.shape: %s", x_hat.shape)
        # reconstruction loss
        x_hat_flattened = x_hat.flatten(start_dim=1)
        x_flattened = x.flatten(start_dim=1)
        assert x_hat_flattened.shape == x_flattened.shape, f"{x_hat_flattened.shape} != {x_flattened.shape}"
        logger.debug("x_hat_flattened.shape: %s", x_hat_flattened.shape)
        logger.debug("x_flattened.shape: %s", x_flattened.shape)
        # qx_z = NegativeBinomial(1_000_000, probs=x_hat_flattened)
        # recon_loss = -qx_z.log_prob(x_flattened).sum(dim=-1)
        # mse loss
        recon_loss = (x_hat_flattened - x_flattened).pow(2).sum(dim=-1)
        logger.debug("recon_loss.shape: %s", recon_loss.shape)
        # kl loss
        ## unsqueeze z for log_prob
        z = z.unsqueeze(0)
        logger.debug("z.shape after unsqueeze: %s", z.shape)
        ## q(z|x)
        qz_x = WrappedNormal(mu, scale, self.manifold)
        logger.debug("qz_x shapes: %s, %s", qz_x.loc.shape, qz_x.scale.shape)
        log_prob_qz_x = qz_x.log_prob(z)
        logger.debug("log_prob_qz_x.shape: %s", log_prob_qz_x.shape)
        assert log_prob_qz_x.shape == torch.Size(
            [1, *batch_shape, 1]
        ), f"{log_prob_qz_x.shape} != {torch.Size([1, *batch_shape, 1])}"
        ## p(z)
        manifold_origin = self.manifold.origin(mu.shape[-1], device=self.device)
        pz = WrappedNormal(
            manifold_origin,
            torch.ones_like(manifold_origin) * self.prior_scale,
            self.manifold,
        )
        logger.debug("pz shapes: %s, %s", pz.loc.shape, pz.scale.shape)
        log_prob_pz = pz.log_prob(z)
        logger.debug("log_prob_pz.shape: %s", log_prob_pz.shape)
        assert log_prob_pz.shape == log_prob_qz_x.shape, f"{log_prob_pz.shape} != {log_prob_qz_x.shape}"
        kl_loss = log_prob_qz_x - log_prob_pz
        logger.debug("kl_loss.shape before sum: %s", kl_loss.shape)
        kl_loss = kl_loss.sum(dim=-1).squeeze(0)
        logger.debug("kl_loss.shape: %s", kl_loss.shape)
        loss = (recon_loss + self.beta * kl_loss).mean()
        return dict(loss_total=loss, recon_loss=recon_loss.mean(), kl_loss=kl_loss.mean())

    def training_step(self, batch, batch_idx):
        loss_dict = self.loss(batch)
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()})
        # if any values in dict are NaN...
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
        optimizer = geoopt.optim.RiemannianAdam(self.parameters(), lr=self.lr)
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
