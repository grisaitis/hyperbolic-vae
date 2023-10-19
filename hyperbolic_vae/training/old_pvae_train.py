import torch
import torch.nn as nn
from pvae.utils import has_analytic_kl, probe_infnan
from torch import optim
from torch.utils.data import DataLoader

from hyperbolic_vae.models.vae_hyperbolic_linear_wrapped import VAEHyperbolic


def loss_function(
    model: VAEHyperbolic,
    x: torch.Tensor,
    K: int = 1,
    beta: float = 1.0,
    components: bool = False,
    analytical_kl: bool = False,
):
    """Computes E_{p(x)}[ELBO]"""
    qz_x, px_z, zs = model(x, K)
    # add dimension to zs
    zs = zs.unsqueeze(0)
    print("zs in loss function", zs.shape)
    _, B, D = zs.size()
    print("B, D", B, D)
    # error here...
    # lpx_z = px_z.log_prob(x.expand(px_z.batch_shape)).view(flat_rest).sum(-1)
    x_for_log_prob = x.expand(px_z.batch_shape)
    print("x_for_log_prob", x_for_log_prob.shape, x_for_log_prob[:1, :1, :1, :5], x_for_log_prob.mean())
    unique_values, counts = torch.unique(x_for_log_prob, return_counts=True)
    print(
        "unique_values and counts",
        unique_values.shape,
        unique_values[:5],
        counts[:5],
        unique_values[-5:],
        counts[-5:],
    )

    lpx_z = px_z.log_prob(x_for_log_prob)  # error should happen here

    flat_rest = torch.Size([*px_z.batch_shape[:2], -1])
    lpx_z = lpx_z.view(flat_rest).sum(-1)
    pz = model.Q_Dist(
        model.manifold.zero,
        torch.ones_like(model.manifold.zero),
        model.manifold,
    )
    kld = (
        torch.distributions.kl_divergence(qz_x, pz).unsqueeze(0).sum(-1)
        if has_analytic_kl(type(qz_x), pz) and analytical_kl
        else qz_x.log_prob(zs).sum(-1) - pz.log_prob(zs).sum(-1)
    )
    obj = -lpx_z.mean(0).sum() + beta * kld.mean(0).sum()
    return (qz_x, px_z, lpx_z, kld, obj) if components else obj


def train(
    model: nn.Module,
    train_loader: DataLoader,
    epoch: int,
    agg: dict,
    device: torch.device,
    optimizer: optim.Optimizer,
    K: int = 1,
    beta: float = 1.0,
    analytical_kl: bool = False,
):
    model.train()
    b_loss, b_recon, b_kl = 0.0, 0.0, 0.0
    for i, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        qz_x, px_z, lik, kl, loss = loss_function(
            model, data, K=K, beta=beta, components=True, analytical_kl=analytical_kl
        )
        probe_infnan(loss, "Training loss:")
        loss.backward()
        optimizer.step()

        b_loss += loss.item()
        b_recon += -lik.mean(0).sum().item()
        b_kl += kl.sum(-1).mean(0).sum().item()

    agg["train_loss"].append(b_loss / len(train_loader.dataset))
    agg["train_recon"].append(b_recon / len(train_loader.dataset))
    agg["train_kl"].append(b_kl / len(train_loader.dataset))
    if epoch % 1 == 0:
        print(
            "====> Epoch: {:03d} Loss: {:.2f} Recon: {:.2f} KL: {:.2f}".format(
                epoch, agg["train_loss"][-1], agg["train_recon"][-1], agg["train_kl"][-1]
            )
        )
