from dataclasses import dataclass

import geoopt
import geoopt.manifolds
import numpy as np
import pvae.distributions
import pvae.manifolds
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pvae.distributions import WrappedNormal
from pvae.ops.manifold_layers import GeodesicLayer, MobiusLayer

import hyperbolic_vae.manifolds
from hyperbolic_vae.models.vae_euclidean import VisualizeVAEEuclideanValidationSetEncodings


# Define a data class for your function's output
@dataclass
class VAEHyperbolicEncoderOutput:
    manifold: geoopt.Stereographic
    mu_qz_x: torch.Tensor
    log_var_qz_x: float

    @property
    def mu_qz_x_on_manifold(self) -> geoopt.ManifoldTensor:
        return self.manifold.expmap0(self.mu_qz_x)

    @property
    def scale_qz_x(self) -> torch.Tensor:
        return torch.exp(0.5 * self.log_var_qz_x)

    @property
    def qz_x(self) -> WrappedNormal:
        return WrappedNormal(self.mu_qz_x_on_manifold, self.scale_qz_x, self.manifold)


class VAEHyperbolic(nn.Module):
    def __init__(
        self,
        latent_dim: int = 2,
        act_fn: object = nn.GELU,
        data_channels: int = 1,
        width: int = 32,
        height: int = 32,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.data_channels = data_channels
        self.width = width
        self.height = height
        self.example_input_array = torch.zeros(torch.Size([1, data_channels, width, height]))

        self.manifold = hyperbolic_vae.manifolds.PoincareBallWithExtras(c=1.0)
        # self.manifold = geoopt.PoincareBall(c=1.0)
        # self.manifold = pvae.manifolds.PoincareBall(dim=latent_dim, c=1.0)
        self.encoder = nn.Sequential(
            nn.Conv2d(data_channels, 16, 3, 2, 1),
            act_fn(),
            nn.Conv2d(16, 32, 3, 2, 1),
            act_fn(),
            nn.Conv2d(32, 32, 3, 2, 1),
            act_fn(),
            nn.Flatten(),
        )
        encoder_out_channels = 32 * (width // 8) * (height // 8)
        # self.mu = MobiusLayer(encoder_out_channels, latent_dim, self.manifold)
        self.mu = nn.Linear(encoder_out_channels, latent_dim)
        self.log_var = nn.Linear(encoder_out_channels, latent_dim)
        self.decoder = nn.Sequential(
            # GeodesicLayer(latent_dim, encoder_out_channels, self.manifold),
            # MobiusLayer(latent_dim, encoder_out_channels, self.manifold),
            nn.Linear(latent_dim, encoder_out_channels),
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
            nn.ConvTranspose2d(16, data_channels, 3, 2, 1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e = self.encoder(x)
        mu_qz_x = self.mu(e)
        log_var_qz_x = self.log_var(e)
        encoder_output = VAEHyperbolicEncoderOutput(self.manifold, mu_qz_x, log_var_qz_x)
        samples_qz_x = encoder_output.qz_x.rsample(torch.Size([1])).squeeze(0)
        x_hat = self.decoder(samples_qz_x)
        return mu_qz_x, log_var_qz_x, samples_qz_x, x_hat

    def forward_1(
        self,
        x: torch.Tensor,
    ):
        # print("x", type(x), x.shape, x[:1, :1, :1, :5])
        e = self.encoder(x)
        # print("e", type(e), e.shape, e[:1, :5])
        mu, log_var = self.mu(e), self.log_var(e)
        # print("mu", type(mu), mu.shape, mu[:5])
        mu_on_manifold = self.manifold.expmap0(mu)  # todo - check this in pvae training code
        # print("mu_on_manifold", type(mu_on_manifold), mu_on_manifold.shape, mu_on_manifold[:5])
        scale = torch.exp(0.5 * log_var)
        # print("scale", type(scale), scale.shape, scale[:5])
        try:
            qz_x = WrappedNormal(mu_on_manifold, scale, self.manifold)
        except Exception as e:
            print("x", type(x), x.shape, x[:1, :1, :1, :5])
            # print value counts of elements of x
            uniques, counts = x.unique(return_counts=True)
            # top 5 values
            top5 = torch.argsort(counts)[-5:]
            print("top5 of x", top5)
            print("e", type(e), e)
            print("e info", type(e), e.shape, e[:1, :5])
            print("mu", type(mu), mu.shape, mu[:5])
            print("mu_on_manifold", type(mu_on_manifold), mu_on_manifold.shape, mu_on_manifold[:5])
            print("scale", type(scale), scale.shape, scale[:5])
            raise e
        samples_qz_x = qz_x.rsample(torch.Size([1])).squeeze(0)
        # print("samples_qz_x", type(samples_qz_x), samples_qz_x.shape, samples_qz_x[:5])
        samples_qz_x = self.manifold.logmap0(samples_qz_x)
        # print("samples_qz_x, logmap0 from manifold", type(samples_qz_x), samples_qz_x.shape)
        x_hat = self.decoder(samples_qz_x)
        # print("x_hat", x_hat.shape)
        # return mu, log_var, z, x_hat
        # px_z = torch.distributions.RelaxedBernoulli(
        #     temperature=torch.tensor([0.1], device=x_hat.device),
        #     logits=x_hat,
        #     # validate_args=False,
        # )
        return mu, log_var, mu_on_manifold, scale, samples_qz_x, x_hat

    def init_last_layer_bias(self, train_loader: torch.utils.data.DataLoader):
        if not hasattr(self.decoder.last_op, "bias"):
            print("no bias to init for last layer of decoder")
            return

        with torch.no_grad():
            from numpy import prod

            data_size = torch.Size([1, 32, 32])

            p = torch.zeros(prod(data_size[1:]), device=self._pz_mu.device)
            N = 0
            for i, (data, _) in enumerate(train_loader):
                data = data.to(self._pz_mu.device)
                B = data.size(0)
                N += B
                p += data.view(-1, prod(data_size[1:])).sum(0)
            p /= N
            p += 1e-4
            self.decoder.last_op.bias.set_(p.log() - (1 - p).log())


class VAEHyperbolicExperiment(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int = 2,
        act_fn: object = nn.GELU,
        data_channels: int = 1,
        width: int = 32,
        height: int = 32,
        beta: float = 1.0,
        lr: float = 1e-3,
        manifold_curvature: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vae = VAEHyperbolic(latent_dim, act_fn, data_channels, width, height)
        self.beta = beta
        self.lr = lr
        self.example_input_array = torch.zeros(torch.Size([1, data_channels, width, height]))

    def forward(self, x):
        return self.vae(x)

    def loss(self, batch: torch.Tensor):
        x, _ = batch
        BATCH_SHAPE = x.size()[0:1]
        # print(BATCH_SHAPE)
        EVENT_SHAPE = torch.Size([self.vae.latent_dim])
        # qz_x, z, x_hat = self.vae.forward(x)
        mu_pz_x, log_var_pz_x, z, x_hat = self.vae.forward(x)
        encoder_output = VAEHyperbolicEncoderOutput(self.vae.manifold, mu_pz_x, log_var_pz_x)
        assert z.shape == BATCH_SHAPE + EVENT_SHAPE, z.shape
        qz_x = encoder_output.qz_x
        assert qz_x.batch_shape == BATCH_SHAPE, qz_x.batch_shape
        assert qz_x.event_shape == EVENT_SHAPE, qz_x.event_shape
        manifold_origin = self.vae.manifold.origin(EVENT_SHAPE, device=z.device)
        pz = pvae.distributions.WrappedNormal(
            manifold_origin,
            torch.ones_like(manifold_origin),
            self.vae.manifold,
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
        loss_recon = nn.functional.mse_loss(x_hat, x, reduction="sum")
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
        return loss_dict["loss_total"]

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss_dict = self.loss(batch)
        self.log_dict({f"val/{k}": v for k, v in loss_dict.items()})
        return loss_dict["loss_total"]

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss_dict = self.loss(batch)
        self.log_dict({f"test/{k}": v for k, v in loss_dict.items()})
        return loss_dict["loss_total"]


class VisualizeVAEPoincareDiskValidationSetEncodings(VisualizeVAEEuclideanValidationSetEncodings):
    def get_encodings(self, images: torch.Tensor, vae_experiment: VAEHyperbolicExperiment) -> np.ndarray:
        images = images.to(vae_experiment.device)
        x = vae_experiment.vae.encoder(images)
        mu = vae_experiment.vae.mu(x)
        mu_on_manifold = vae_experiment.vae.manifold.expmap0(mu)
        return mu_on_manifold.cpu().numpy()
