import logging
from abc import ABC

import geoopt
import geoopt.layers.stereographic
import pytorch_lightning as pl
import torch
import torch.nn as nn

import hyperbolic_vae
import hyperbolic_vae.distributions.wrapped_normal
import hyperbolic_vae.layers

logger = logging.getLogger(__name__)


class VAE(pl.LightningModule):
    def __init__(
        self,
        input_size: torch.Size,
        hidden_layer_dim: int,
        latent_dim: int,
        latent_curvature: float,
        learning_rate: float,
        beta: float,
        activation_class: type,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_size = input_size
        self.example_input_array = torch.zeros(input_size)
        self.hidden_layer_dim = hidden_layer_dim
        self.latent_dim = latent_dim
        self.latent_curvature = latent_curvature
        self.latent_manifold = geoopt.PoincareBall(latent_curvature) if latent_curvature else None
        self.learning_rate = learning_rate
        self.beta = beta
        self.activation_class = activation_class
        self.input_features = input_size.numel()
        self.encoder = nn.Sequential(
            *self._make_encoder_first_ops(),
            nn.Linear(self.input_features, self.hidden_layer_dim),
            self.activation_class(),
        )
        self.mu = self._make_mu()
        # self.scale = nn.Sequential(
        #     nn.Linear(self.hidden_layer_dim, self.latent_dim),
        #     nn.Softplus(),
        # )
        self.decoder = nn.Sequential(
            self._make_decoder_first_op(),
            self.activation_class(),
            nn.Linear(self.hidden_layer_dim, self.input_features),
            *self._make_decoder_last_ops(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logger.debug("type(x): %s", type(x))
        logger.debug("if tensor, shape of x: %s", x.shape if isinstance(x, torch.Tensor) else None)
        h = self.encoder(x)
        mu = self.mu(h)
        # scale = self.scale(h)
        scale = torch.ones_like(mu)
        logger.debug("mu.shape: %s", mu.shape)
        logger.debug("mu (first 5): %s", mu[:5])
        if self.latent_manifold:
            # z = self.latent_manifold.wrapped_normal(*mu.shape, mean=mu, std=scale)  # results in "ERROR: Graphs differed across invocations!"
            z_dist = hyperbolic_vae.distributions.wrapped_normal.WrappedNormal(
                loc=mu, scale=scale, manifold=self.latent_manifold
            )
            z = z_dist.rsample()
        elif self.latent_manifold and False:
            mu = self.latent_manifold.logmap0(mu)
            z = torch.distributions.Normal(loc=mu, scale=scale).rsample()
            z = self.latent_manifold.expmap0(z)
        else:
            z = torch.distributions.Normal(loc=mu, scale=scale).rsample()
        logger.debug("z.shape: %s", z.shape)
        logger.debug("z (first 5): %s", z[:5])
        assert z.shape == mu.shape, f"z.shape: {z.shape}, mu.shape: {mu.shape}"
        output = self.decoder(z)
        return mu, scale, z, output

    def _make_encoder_first_ops(self) -> tuple[nn.Module]:
        if len(self.input_size) == 1:
            return ()
        else:
            return (nn.Flatten(),)

    def _make_mu(self) -> nn.Module:
        ops = [nn.Linear(self.hidden_layer_dim, self.latent_dim)]
        if self.latent_manifold:
            ops.append(hyperbolic_vae.layers.ExpMap0(self.latent_manifold))
        return nn.Sequential(*ops)

    def _make_decoder_first_op(self) -> nn.Module:
        if self.latent_manifold:
            # version 1... we have latent vars clustering weirdly
            # return geoopt.layers.stereographic.Distance2StereographicHyperplanes(
            #     self.latent_dim,
            #     self.hidden_layer_dim,
            #     ball=self.latent_manifold,
            # )
            return hyperbolic_vae.layers.Distance2PoincareHyperplanes(
                plane_shape=self.latent_dim, num_planes=self.hidden_layer_dim, ball=self.latent_manifold, std=0.1
            )
            # return nn.Linear(self.latent_dim, self.hidden_layer_dim)
        else:
            return nn.Linear(self.latent_dim, self.hidden_layer_dim)

    def _make_decoder_last_ops(self) -> tuple[nn.Module]:
        if len(self.input_size) == 1:
            return (nn.Sigmoid(),)
        else:
            return (
                nn.Unflatten(1, self.input_size),
                nn.Sigmoid(),
            )

    def loss_recon(self, x: torch.Tensor, output: torch.Tensor):
        return torch.nn.functional.mse_loss(output, x, reduction="mean")

    def loss_kl_old(self, mu: torch.Tensor | geoopt.ManifoldTensor, scale: torch.Tensor) -> torch.Tensor:
        """
        # general formula for posterior in euclidean space:
        kl = -0.5 * torch.sum(1 + scale - mu.pow(2) - scale.exp(), dim=-1)
        # in torch.distributions.kl:
        def _kl_normal_normal(p, q):
            var_ratio = (p.scale / q.scale).pow(2)
            t1 = ((p.loc - q.loc) / q.scale).pow(2)
            return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
        # we are computing this where p is the posterior and q is the prior
        """
        logger.debug("shapes: mu: %s, scale: %s", mu.shape, scale.shape)
        var_ratio = scale.pow(2).sum(-1)  # shape (64,)
        if self.latent_manifold:
            # mu_norm = self.latent_manifold.norm(self.latent_manifold.origin(self.latent_dim), mu)
            # t1 = mu_norm.pow(2)  # shape (64,)
            mu_logmap = self.latent_manifold.logmap0(mu)
            t1 = mu_logmap.pow(2).sum(-1)
        else:
            t1 = mu.pow(2).sum(-1)
        logger.debug("var_ratio.shape: %s", var_ratio.shape)
        logger.debug("t1.shape: %s", t1.shape)
        return 0.5 * (var_ratio + t1 - 1 - var_ratio.log()).mean()

    def loss_kl(self, mu: torch.Tensor, scale: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        qz = hyperbolic_vae.distributions.wrapped_normal.WrappedNormal(
            loc=self.latent_manifold.origin(self.latent_dim),
            scale=torch.ones_like(scale),
            manifold=self.latent_manifold,
        )
        qz_x = hyperbolic_vae.distributions.wrapped_normal.WrappedNormal(
            loc=mu,
            scale=scale,
            manifold=self.latent_manifold,
        )
        return (qz_x.log_prob(z).exp() * (qz_x.log_prob(z) - qz.log_prob(z))).mean()

    def loss(self, batch: tuple) -> dict:
        x, class_labels = batch
        mu, scale, z, output = self.forward(x)
        loss_recon = self.loss_recon(x, output)
        loss_kl = self.loss_kl(mu, scale, z)
        loss_total = loss_recon + self.beta * loss_kl
        logger.info("loss values:\n recon: %s\n  kl: %s\n  total: %s", loss_recon, loss_kl, loss_total)
        return {
            "loss_reconstruction": loss_recon,
            "loss_kl": loss_kl,
            "loss_total": loss_total,
        }

    def configure_optimizers(self):
        optimizer = geoopt.optim.RiemannianAdam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val/loss_total"}

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
