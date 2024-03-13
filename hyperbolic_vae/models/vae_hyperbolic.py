import logging
from typing import Literal

import geoopt
import geoopt.layers.stereographic
import geoopt.manifolds
import numpy as np
import pvae.distributions
import pvae.manifolds

from pvae.distributions import WrappedNormal
import pvae.ops.manifold_layers
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

import hyperbolic_vae.manifolds
from hyperbolic_vae.distributions.wrapped_normal import WrappedNormal
from hyperbolic_vae.models.vae_euclidean import VisualizeVAEEuclideanValidationSetEncodings

logger = logging.getLogger(__name__)


# changes
# - refactor - add decoder_first_layer_module argument to VAEHyperbolic
# - refactor - replace using pvae.distributions.WrappedNormal with hyperbolic_vae.distributions.WrappedNormal
# - refactor - replace hyperbolic_vae.manifolds.PoincareBallWithExtras with geoopt.PoincareBall
# - refactor - replace VAEHyperbolicEncoderOutput with direct use of WrappedNormal
# - change - add option for decoder_first_layer_module
# - change - add option for encoder_last_layer_module
# - change - add option for RelaxedBernoulli loss

# to change...
# - refactor - replace data_channels, width, height with data_shape

activation_functions = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


class ImageVAEHyperbolic(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        act_fn: nn.Module,
        image_shape: tuple,
        encoder_last_layer_module: Literal["linear", "mobius"],
        decoder_first_layer_module: Literal["linear", "pvae_geodesic", "pvae_mobius", "geoopt_gyroplane"],
        manifold_curvature: float,
        loss_recon: str,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_last_layer_module = encoder_last_layer_module
        self.decoder_first_layer_module = decoder_first_layer_module
        self.loss_recon = loss_recon
        self.example_input_array = torch.zeros(1, *image_shape)
        image_channels, width, height = image_shape
        self.manifold = geoopt.PoincareBall(c=manifold_curvature)
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, 3, 2, 1),
            act_fn(),
            nn.Conv2d(16, 32, 3, 2, 1),
            act_fn(),
            nn.Conv2d(32, 32, 3, 2, 1),
            act_fn(),
            nn.Flatten(),
        )
        encoder_out_channels = 32 * (width // 8) * (height // 8)
        logger.info("encoder_out_channels: %s", encoder_out_channels)
        # self.mu = MobiusLayer(encoder_out_channels, latent_dim, self.manifold)
        if encoder_last_layer_module == "linear":
            self.mu = nn.Linear(encoder_out_channels, latent_dim)
        elif encoder_last_layer_module == "pvae_mobius":
            self.mu = pvae.ops.manifold_layers.MobiusLayer(encoder_out_channels, latent_dim, self.manifold)
        else:
            raise ValueError(f"encoder_last_layer_module {encoder_last_layer_module} not supported")
        self.log_var = nn.Linear(encoder_out_channels, latent_dim)
        if decoder_first_layer_module == "linear":
            decoder_first_layer = nn.Linear(latent_dim, encoder_out_channels)
        elif decoder_first_layer_module == "pvae_geodesic":
            decoder_first_layer = pvae.ops.manifold_layers.GeodesicLayer(
                latent_dim, encoder_out_channels, self.manifold
            )
        elif decoder_first_layer_module == "pvae_mobius":
            decoder_first_layer = pvae.ops.manifold_layers.MobiusLayer(latent_dim, encoder_out_channels, self.manifold)
        elif decoder_first_layer_module == "geoopt_gyroplane":
            decoder_first_layer = geoopt.layers.stereographic.Distance2StereographicHyperplanes(
                latent_dim,
                encoder_out_channels,
                ball=self.manifold,
            )
        else:
            raise ValueError(f"decoder_first_layer {decoder_first_layer} not supported")
        logger.info("decoder_first_layer: %s", decoder_first_layer)
        decoder_layers = [
            decoder_first_layer,
            act_fn(),
            nn.Unflatten(-1, (32, width // 8, height // 8)),
            nn.ConvTranspose2d(32, 32, 3, 2, 1, output_padding=1),
            act_fn(),
            nn.Conv2d(32, 32, 3, 1, 1),
            act_fn(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1),
            act_fn(),
            nn.Conv2d(16, 16, 3, 1, 1),
            act_fn(),
            nn.ConvTranspose2d(16, image_channels, 3, 2, 1, output_padding=1),
            # nn.Sigmoid(),
        ]
        if self.loss_recon == "mse":
            decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e = self.encoder(x)
        mu_qz_x = self.mu(e)
        if self.loss_recon == "bernoulli":
            log_var_qz_x = torch.zeros_like(mu_qz_x)
        else:
            log_var_qz_x = self.log_var(e)
        # encoder_output = VAEHyperbolicEncoderOutput(self.manifold, mu_qz_x, log_var_qz_x)
        # samples_qz_x = encoder_output.qz_x.rsample(torch.Size([1])).squeeze(0)
        if self.encoder_last_layer_module == "linear":
            mu_qz_x_on_manifold = self.manifold.expmap0(mu_qz_x)
        elif self.encoder_last_layer_module == "pvae_mobius":
            mu_qz_x_on_manifold = mu_qz_x
        else:
            raise ValueError(f"encoder_last_layer_module {self.encoder_last_layer_module} not supported")
        scale_qz_x = torch.exp(0.5 * log_var_qz_x)
        qz_x = WrappedNormal(mu_qz_x_on_manifold, scale_qz_x, self.manifold)
        samples_qz_x = qz_x.rsample(torch.Size([1])).squeeze(0)
        x_hat = self.decoder(samples_qz_x)
        return mu_qz_x, log_var_qz_x, samples_qz_x, x_hat


class VAEHyperbolicExperiment(pl.LightningModule):
    def __init__(
        self,
        image_shape: tuple = (1, 32, 32),
        latent_dim: int = 2,
        manifold_curvature: float = 1.0,
        encoder_last_layer_module: Literal["linear", "pvae_mobius"] = "linear",
        decoder_first_layer_module: Literal["linear", "pvae_geodesic", "pvae_mobius", "geoopt_gyroplane"] = "linear",
        beta: float = 1.0,
        lr: float = 1e-3,
        loss_recon: Literal["mse", "bernoulli"] = "mse",
    ):
        super().__init__()
        self.save_hyperparameters(
            "latent_dim",
            "manifold_curvature",
            "encoder_last_layer_module",
            "decoder_first_layer_module",
            "beta",
            "lr",
            "loss_recon",
        )
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.manifold_curvature = manifold_curvature
        self.encoder_last_layer_module = encoder_last_layer_module
        self.decoder_first_layer_module = decoder_first_layer_module
        self.model = ImageVAEHyperbolic(
            latent_dim,
            nn.GELU,
            image_shape,
            encoder_last_layer_module,
            decoder_first_layer_module,
            manifold_curvature,
            loss_recon,
        )
        self.beta = beta
        self.lr = lr
        self.loss_recon = loss_recon
        self.example_input_array = torch.zeros(1, *image_shape)
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def loss(self, batch: torch.Tensor):
        x, _ = batch
        BATCH_SHAPE = x.size()[0:1]
        # print(BATCH_SHAPE)
        EVENT_SHAPE = torch.Size([self.model.latent_dim])
        mu_qz_x, log_var_qz_x, z, x_hat = self.model(x)
        # encoder_output = VAEHyperbolicEncoderOutput(self.vae.manifold, mu_pz_x, log_var_pz_x)
        assert z.shape == BATCH_SHAPE + EVENT_SHAPE, z.shape
        # qz_x = encoder_output.qz_x
        if self.model.encoder_last_layer_module == "linear":
            mu_qz_x_on_manifold = self.model.manifold.expmap0(mu_qz_x)
        elif self.model.encoder_last_layer_module == "pvae_mobius":
            mu_qz_x_on_manifold = mu_qz_x
        scale_qz_x = torch.exp(0.5 * log_var_qz_x)
        qz_x = WrappedNormal(mu_qz_x_on_manifold, scale_qz_x, self.model.manifold)
        assert qz_x.batch_shape == BATCH_SHAPE, qz_x.batch_shape
        assert qz_x.event_shape == EVENT_SHAPE, qz_x.event_shape
        manifold_origin = self.model.manifold.origin(EVENT_SHAPE, device=z.device)
        pz = WrappedNormal(
            manifold_origin,
            torch.ones_like(manifold_origin),
            self.model.manifold,
        )
        # print("pz loc and scale", pz.loc, pz.scale)
        # print("pz", pz, pz.loc.shape, pz.scale.shape, pz.batch_shape, pz.event_shape)
        # print("qz_x", qz_x, qz_x.loc.shape, qz_x.scale.shape, qz_x.batch_shape, qz_x.event_shape)
        # print("z", type(z), z.shape)
        # print("calling pz.log_prob(z)")
        # add dimension to z
        z = z.unsqueeze(0)
        # print("z after unsqueeze", type(z), z.shape)
        log_prob_pz = pz.log_prob(z)
        # print("log_prob_pz", type(log_prob_pz), log_prob_pz.shape)
        assert log_prob_pz.shape == torch.Size([1, x.shape[0], 1])
        # print("calling qz_x.log_prob(z)")
        # print("z before qz_x.log_prob(z)", type(z), z.shape)
        log_prob_qz_x = qz_x.log_prob(z)
        # print("log_prob_qz_x", type(log_prob_qz_x), log_prob_qz_x.shape)
        assert log_prob_qz_x.shape == torch.Size([1, *BATCH_SHAPE, 1])
        loss_kl = (log_prob_qz_x - log_prob_pz).sum()
        # print("loss_kl", type(loss_kl), loss_kl.shape)
        if self.loss_recon == "mse":
            loss_recon = nn.functional.mse_loss(x_hat, x, reduction="sum")
        elif self.loss_recon == "bernoulli":
            x_hat_flattened = x_hat.flatten(start_dim=1)
            x_flattened = x.flatten(start_dim=1)
            assert x_hat_flattened.shape == x_flattened.shape, f"{x_hat_flattened.shape} != {x_flattened.shape}"
            qx_z = torch.distributions.RelaxedBernoulli(temperature=torch.tensor(0.1), logits=x_hat_flattened)
            loss_recon = -qx_z.log_prob(x_flattened).mean()
        else:
            raise ValueError(f"loss_recon {self.loss_recon} not supported")
        # print("loss_recon", type(loss_recon), loss_recon.shape)
        return {
            "loss_total": loss_recon + self.beta * loss_kl,
            "loss_recon": loss_recon,
            "loss_kl": loss_kl,
        }

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

    def training_step(self, batch, batch_idx):
        loss_dict = self.loss(batch)
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()})
        loss_dict["loss"] = loss_dict["loss_total"]
        return loss_dict

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss_dict = self.loss(batch)
        self.log_dict({f"val/{k}": v for k, v in loss_dict.items()})
        return loss_dict

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss_dict = self.loss(batch)
        if self.loss_recon == "mse":
            loss_dict["mse"] = loss_dict["loss_recon"]
        else:
            x, _ = batch
            _, _, _, x_hat = self(x)
            loss_dict["mse"] = nn.functional.mse_loss(x_hat, x, reduction="sum")
        self.log_dict({f"test/{k}": v for k, v in loss_dict.items()})
        self.test_step_outputs.append(loss_dict)
        return loss_dict

    def on_test_epoch_end(self):
        # 'outputs' is a list of dictionaries returned from test_step for each batch
        # Compute the mean or aggregate metric over all batches
        outputs = self.test_step_outputs
        avg_loss = torch.stack([x["loss_total"] for x in outputs]).mean()
        avg_mse = torch.stack([x["mse"] for x in outputs]).mean()
        self.log("avg_test_mse", avg_mse)
        self.test_step_outputs.clear()
        return {"avg_test_loss": avg_loss, "avg_test_mse": avg_mse}

    def on_fit_start(self):
        # self.logger.experiment.add_hparams(hparam_dict=dict(vae_experiment.hparams), metric_dict={"val/accuracy": 0, "val/loss": 1})
        metric_placeholder = {"avg_test_mse": 0, "avg_test_loss": 1}
        self.logger.log_hyperparams(self.hparams, metrics=metric_placeholder)


class VisualizeVAEPoincareDiskValidationSetEncodings(VisualizeVAEEuclideanValidationSetEncodings):
    def get_encodings(self, images: torch.Tensor, vae_experiment: VAEHyperbolicExperiment) -> np.ndarray:
        images = images.to(vae_experiment.device)
        x = vae_experiment.model.encoder(images)
        mu = vae_experiment.model.mu(x)
        mu_on_manifold = vae_experiment.model.manifold.expmap0(mu)
        return mu_on_manifold.cpu().numpy()
