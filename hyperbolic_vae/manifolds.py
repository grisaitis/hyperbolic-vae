import logging

import geoopt
import torch
from geoopt import ManifoldTensor
from geoopt.manifolds import PoincareBall
from geoopt.manifolds.stereographic.math import arsinh

logger = logging.getLogger(__name__)


class PoincareBallWithExtras(PoincareBall):
    pass

    # def logdetexp(self, x: ManifoldTensor, y: ManifoldTensor, is_vector=False, keepdim=False):
    #     logger.warning("calling PoincareBallWithExtras.logdetexp")
    #     if is_vector:
    #         d = self.norm(x, y, keepdim=keepdim)
    #     else:
    #         d = self.dist(x, y, keepdim=keepdim)
    #     dim = x.shape[-1]
    #     return (dim - 1) * (torch.sinh(self.c.sqrt() * d) / self.c.sqrt() / d).log()


def logdetexp(
    manifold: geoopt.manifolds.PoincareBall,
    x: ManifoldTensor,
    y: ManifoldTensor,
    keepdim=False,
) -> torch.Tensor:
    dist_x_y = manifold.dist(x, y, keepdim=keepdim)
    dim = x.shape[-1]
    c_sqrt = manifold.c.sqrt()
    # return (dim - 1) * (torch.sinh(c_sqrt * dist_x_y) / c_sqrt / dist_x_y).log()
    return (dim - 1) * (torch.sinh(c_sqrt * dist_x_y).log() - c_sqrt.log() - dist_x_y.log())


MIN_NORM = 1e-15


def normdist2plane(
    manifold_poincare: geoopt.PoincareBall,
    x,
    a,
    p,
    keepdim: bool = False,
    signed: bool = False,
    dim: int = -1,
    norm: bool = False,
):
    logger.warning("calling %s.normdist2plane (in hyperbolic_vae/manifolds.py)", __name__)
    c = manifold_poincare.c
    sqrt_c = c**0.5
    diff = manifold_poincare.mobius_add(-p, x, dim=dim)
    diff_norm2 = diff.pow(2).sum(dim=dim, keepdim=keepdim).clamp_min(MIN_NORM)
    sc_diff_a = (diff * a).sum(dim=dim, keepdim=keepdim)
    if not signed:
        sc_diff_a = sc_diff_a.abs()
    a_norm = a.norm(dim=dim, keepdim=keepdim, p=2).clamp_min(MIN_NORM)
    num = 2 * sqrt_c * sc_diff_a
    denom = (1 - c * diff_norm2) * a_norm
    res = arsinh(num / denom.clamp_min(MIN_NORM)) / sqrt_c
    if norm:
        res = res * a_norm  # * self.lambda_x(a, dim=dim, keepdim=keepdim)
    return res
