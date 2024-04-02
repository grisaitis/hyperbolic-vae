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
        prior_scale: float,
        posterior_scale: str,
        learning_rate: float,
        beta: float,
        kl_loss_method: str,
        activation_class: type,
        last_activation: str,
        loss_recon_method: str,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_size = input_size
        self.example_input_array = torch.zeros(input_size)
        self.hidden_layer_dim = hidden_layer_dim
        self.latent_dim = latent_dim
        self.latent_curvature = latent_curvature
        self.latent_manifold = geoopt.PoincareBall(latent_curvature) if latent_curvature else None
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.learning_rate = learning_rate
        self.beta = beta
        self.kl_loss_method = kl_loss_method
        self.activation_class = activation_class
        self.last_activation = last_activation
        self.loss_recon_method = loss_recon_method
        self.input_features = input_size.numel()
        self.encoder = nn.Sequential(
            *self._make_encoder_first_ops(),
            nn.Linear(self.input_features, self.hidden_layer_dim),
            self.activation_class(),
        )
        logger.info("encoder:\n%s", self.encoder)
        self.mu = self._make_mu()
        logger.info("mu:\n%s", self.mu)
        if self.posterior_scale == "learned":
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_layer_dim, self.latent_dim),
                nn.Softplus(),
            )
        elif self.posterior_scale == "fixed":
            self.scale = None
        else:
            raise ValueError(f"Unrecognized posterior_scale: {self.posterior_scale}")
        self.decoder = nn.Sequential(
            self._make_decoder_first_op(),
            self.activation_class(),
            nn.Linear(self.hidden_layer_dim, self.input_features),
            *self._make_decoder_last_ops(),
        )
        logger.info("decoder:\n%s", self.decoder)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # assert x.shape == (self.batch_size,) + self.input_size, x.shape
        h = self.encoder(x)
        mu = self.mu(h)
        if self.scale:
            scale = self.scale(h)
        else:
            scale = torch.ones_like(mu)
        # assert mu.shape == (self.batch_size, self.latent_dim), mu.shape
        # logger.debug("mu (first 5): %s", mu[:5])
        if self.latent_manifold:
            # z = self.latent_manifold.wrapped_normal(*mu.shape, mean=mu, std=scale)  # results in "ERROR: Graphs differed across invocations!"
            qz_x = hyperbolic_vae.distributions.wrapped_normal.WrappedNormal(
                loc=mu, scale=scale, manifold=self.latent_manifold
            )
            z = qz_x.rsample()
        elif self.latent_manifold and False:
            mu = self.latent_manifold.logmap0(mu)
            z = torch.distributions.Normal(loc=mu, scale=scale).rsample()
            z = self.latent_manifold.expmap0(z)
        else:
            z = torch.distributions.Normal(loc=mu, scale=scale).rsample()
        # logger.debug("z.shape: %s", z.shape)
        # logger.debug("z (first 5): %s", z[:5])
        assert z.shape == mu.shape, f"z.shape: {z.shape}, mu.shape: {mu.shape}"
        # z = self.latent_manifold.logmap0(z)
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
                plane_shape=self.latent_dim, num_planes=self.hidden_layer_dim, ball=self.latent_manifold
            )
            # return nn.Linear(self.latent_dim, self.hidden_layer_dim)
        else:
            return nn.Linear(self.latent_dim, self.hidden_layer_dim)

    def _make_decoder_last_ops(self) -> list[nn.Module]:
        ops = []
        if len(self.input_size) > 1:
            ops.append(nn.Unflatten(1, self.input_size))
        if self.last_activation == "none":
            pass
        elif self.last_activation == "sigmoid":
            ops.append(nn.Sigmoid())
        elif self.last_activation == "softplus":
            ops.append(nn.Softplus())
        else:
            raise ValueError(f"Unrecognized last_activation: {self.last_activation}")
        return ops

    def loss_recon(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        if self.loss_recon_method == "MSE":
            return torch.nn.functional.mse_loss(output, x, reduction="mean")
        if self.loss_recon_method == "binary_cross_entropy":
            return torch.nn.functional.binary_cross_entropy(output, x, reduction="mean")
        if self.loss_recon_method == "binary_cross_entropy_with_logits":
            return torch.nn.functional.binary_cross_entropy_with_logits(output, x, reduction="mean")
        if self.loss_recon_method == "relaxed bernoulli":
            # temperature 0.5 is justified by Wang et al, 2020 "Relaxed Multivariate Bernoulli Distribution and Its Applications to Deep Generative Models"
            temperature = torch.Tensor([0.3])
            if self.last_activation == "none":
                likelihood = torch.distributions.RelaxedBernoulli(temperature, logits=output)
            elif self.last_activation == "sigmoid":
                likelihood = torch.distributions.RelaxedBernoulli(temperature, probs=output)
            else:
                raise ValueError(f"last_activation {self.last_activation} not compatible with relaxed bernoulli")
            return -likelihood.log_prob(x).mean()
        elif self.loss_recon_method == "negative binomial":
            raise NotImplementedError("this requires integer counts data")
        else:
            raise ValueError(f"Unrecognized loss_recon_method: {self.loss_recon_method}")

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
        if self.prior_scale != 1.0:
            raise NotImplementedError(f"prior_scale {self.prior_scale} != 1.0 not implemented")
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

    def loss_kl_log_prob(self, mu: torch.Tensor, scale: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # see https://github.com/emilemathieu/pvae/blob/master/pvae/objectives.py#L15
        scale_qz = torch.ones_like(scale) * self.prior_scale
        if self.latent_manifold:
            qz = hyperbolic_vae.distributions.wrapped_normal.WrappedNormal(
                loc=self.latent_manifold.origin(self.latent_dim),
                scale=scale_qz,
                manifold=self.latent_manifold,
            )
            qz_x = hyperbolic_vae.distributions.wrapped_normal.WrappedNormal(
                loc=mu,
                scale=scale,
                manifold=self.latent_manifold,
            )
        else:
            qz = torch.distributions.Normal(loc=torch.zeros_like(mu), scale=scale_qz)
            qz_x = torch.distributions.Normal(loc=mu, scale=scale)
        qz_x_log_prob_z = qz_x.log_prob(z)
        qz_x_prob_z = qz_x_log_prob_z.exp()
        qz_log_prob_z = qz.log_prob(z)
        return (qz_x_prob_z * (qz_x_log_prob_z - qz_log_prob_z)).mean()

    def loss_kl_logmap0_analytic(self, mu: torch.Tensor, scale: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # see https://github.com/emilemathieu/pvae/blob/master/pvae/objectives.py#L15
        if self.latent_manifold:
            mu = self.latent_manifold.logmap0(mu)
        qz_x = torch.distributions.Normal(loc=mu, scale=scale)
        scale_qz = torch.ones_like(scale) * self.prior_scale
        qz = torch.distributions.Normal(loc=torch.zeros_like(mu), scale=scale_qz)
        return torch.distributions.kl_divergence(qz_x, qz).mean()

    def loss_kl_logmap0_log_prob(self, mu: torch.Tensor, scale: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.latent_manifold:
            mu = self.latent_manifold.logmap0(mu)
            z = self.latent_manifold.logmap0(z)
        scale_qz = torch.ones_like(scale) * self.prior_scale
        qz = torch.distributions.Normal(loc=torch.zeros_like(mu), scale=scale_qz)
        qz_x = torch.distributions.Normal(loc=mu, scale=scale)
        qz_log_prob_z = qz.log_prob(z).sum(-1)
        qz_x_log_prob_z = qz_x.log_prob(z).sum(-1)
        qz_x_prob_z = qz_x_log_prob_z.exp()
        logger.debug("qz_log_prob_z.shape: %s", qz_log_prob_z.shape)
        logger.debug("qz_x_log_prob_z.shape: %s", qz_x_log_prob_z.shape)
        logger.debug("qz_x_prob_z.shape: %s", qz_x_prob_z.shape)
        res = qz_x_prob_z * (qz_x_log_prob_z - qz_log_prob_z)
        logger.debug("res.shape: %s", res.shape)
        res = res.mean()
        logger.debug("kl divergence: %s", res)
        return res

    def loss_kl(self, mu: torch.Tensor, scale: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.kl_loss_method == "logmap0_analytic":
            return self.loss_kl_logmap0_analytic(mu, scale, z)
        if self.kl_loss_method == "log_prob":
            return self.loss_kl_log_prob(mu, scale, z)
        if self.kl_loss_method == "logmap0_log_prob":
            return self.loss_kl_logmap0_log_prob(mu, scale, z)
        raise ValueError(f"Unrecognized kl_loss_method: {self.kl_loss_method}")

    def loss(self, batch: tuple) -> dict:
        x, class_labels = batch
        mu, scale, z, output = self.forward(x)
        loss_recon = self.loss_recon(x, output)
        loss_kl = self.loss_kl(mu, scale, z)
        loss_total = loss_recon + self.beta * loss_kl
        logger.debug(
            "loss values:\n recon: %s\n  kl: %s\n  total: %s", loss_recon.item(), loss_kl.item(), loss_total.item()
        )
        logger.debug("z mean and std: %s, %s", z.mean(dim=0), z.std())
        logger.debug("scale mean and std: %s, %s", scale.mean(dim=0), scale.std())
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

    def transform_decoder_output(self, output: torch.Tensor) -> torch.Tensor:
        if self.last_activation == "none" and self.loss_recon_method in (
            "binary_cross_entropy",
            "binary_cross_entropy_with_logits",
            "relaxed bernoulli",
        ):
            return torch.nn.functional.sigmoid(output)
        else:
            return output

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        mu, scale, z, output = self.forward(x)
        assert output.shape == x.shape, f"output.shape: {output.shape}, x.shape: {x.shape}"
        return self.transform_decoder_output(output)
