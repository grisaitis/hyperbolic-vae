import logging

import geoopt
import geoopt.layers.stereographic
import numpy as np

# import pvae.ops.manifold_layers
import pvae.distributions
import pytorch_lightning as pl
import torch

# from pvae.distributions import WrappedNormal
from torch import nn
from torch.distributions import RelaxedBernoulli

import hyperbolic_vae
from hyperbolic_vae.distributions.wrapped_normal import WrappedNormal
from hyperbolic_vae.layers import ExpMap0, GeodesicLayer
from hyperbolic_vae.models.vae_euclidean import VisualizeVAEEuclideanValidationSetEncodings

logger = logging.getLogger(__name__)


# hypotheses for weird convergence
# maybe...
# - nn.Sigmoid() in decoder / RelaxedBernoulli with probs instead of logits
# - using RelaxedBernoulli instead of MSE
#   - i think there was an error with a exp / log somewhere last week with this
# - architecture - flatten + nn.Linear instead of Conv2d
# - using log_scale instead of Softplus for scale
# probably not...
# - initialization
# - using nn.Linear + ExpMap0 instead of nn.Linear + manifold.expmap0 in forward call
#   - in theory, these should be equivalent
# - Distance2StereographicHyperplanes in decoder
#   - no - in new script and working
# - WrappedNormal
#   - no - old script uses this, works fine

# follow-ups with Prof Zhang
# - report MSE for each model on test set
# - report importance-weighted autoencoder loss for each model on test set


class VAEHyperbolicGyroplaneDecoder(pl.LightningModule):
    def __init__(
        self,
        data_shape: torch.Size = torch.Size([1, 32, 32]),
        latent_dim: int = 2,
        manifold_curvature: float = 1.0,
        lr: float = 1e-3,
        beta: float = 1.0,
        prior_scale: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.zeros(data_shape)
        self.lr = lr
        self.beta = beta
        self.latent_dim = latent_dim
        self.manifold = geoopt.PoincareBall(c=manifold_curvature)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(data_shape.numel(), 64),
            nn.GELU(),
            nn.Linear(64, 16),
            nn.GELU(),
        )
        self.mu = nn.Sequential(nn.Linear(16, latent_dim), ExpMap0(self.manifold))
        self.scale = nn.Sequential(nn.Linear(16, latent_dim), nn.Softplus())
        self.decoder = nn.Sequential(
            # GeodesicLayer(latent_dim, 16, manifold=self.manifold),
            geoopt.layers.stereographic.Distance2StereographicHyperplanes(
                latent_dim,
                16,
                ball=self.manifold,
            ),
            nn.GELU(),
            nn.Linear(16, 64),
            nn.GELU(),
            nn.Linear(64, data_shape.numel()),
            nn.Sigmoid(),
            nn.Unflatten(dim=-1, unflattened_size=data_shape),
        )
        self.prior_scale = prior_scale

    def forward(self, x: torch.Tensor):
        logger.info("input x shape, mean, std: %s, %s, %s", x.shape, x.mean(), x.std())
        x = self.encoder(x)
        logger.info("x shape, mean, std: %s, %s, %s", x.shape, x.mean(), x.std())
        mu = self.mu(x)
        logger.info("mu shape, mean, std: %s, %s, %s", mu.shape, mu.mean(), mu.std())
        scale = self.scale(x)
        logger.info("scale shape, mean, std: %s, %s, %s", scale.shape, scale.mean(), scale.std())
        logger.debug("shapes: mu %s, scale %s", mu.shape, scale.shape)
        if not self.manifold.check_point_on_manifold(mu):
            logger.warning("mu not on manifold!")
        qz_x = WrappedNormal(mu, scale, self.manifold)
        z = qz_x.rsample(torch.Size([1]))
        logger.debug("z.shape after rsample: %s", z.shape)
        z = z.squeeze(0)
        logger.debug("z.shape after squeeze: %s", z.shape)
        x_hat = self.decoder(z)
        # qx_z = RelaxedBernoulli(logits=x_hat, temperature=1.0)
        return mu, scale, z, x_hat

    def loss(self, batch: torch.Tensor) -> dict:
        x, _ = batch
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
        qx_z = RelaxedBernoulli(temperature=torch.tensor(1.0), probs=x_hat_flattened)
        recon_loss = -qx_z.log_prob(x_flattened).sum(dim=-1)
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


class VisualizeEncodingsValidationSet(VisualizeVAEEuclideanValidationSetEncodings):
    def get_encodings(
        self, images: torch.Tensor, vae_experiment: VAEHyperbolicGyroplaneDecoder
    ) -> np.ndarray:
        images = images.to(vae_experiment.device)
        x = vae_experiment.encoder(images)
        mu_on_manifold = vae_experiment.mu(x)
        return mu_on_manifold.cpu().numpy()
