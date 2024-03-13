import logging
import math

import geoopt
import torch
from geoopt import ManifoldParameter
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

from hyperbolic_vae.manifolds import normdist2plane

logger = logging.getLogger(__name__)


# class GyroplaneLayer(nn.Module):
#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         manifold: geoopt.Manifold,
#     ):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.manifold = manifold

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         # input = input.unsqueeze(-2).expand(*input.shape[:-(len(input.shape) - 2)], self.out_features, self.in_features)
#         self.manifold.normdist2plane(input, self.bias, self.weight,
#                                                signed=True, norm=self.weight_norm)
#         return res


class RiemannianLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        manifold: geoopt.PoincareBall,
        over_param: bool,
        weight_norm: bool,
    ):
        super(RiemannianLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold

        self._weight = Parameter(torch.Tensor(out_features, in_features))
        self.over_param = over_param
        self.weight_norm = weight_norm
        if self.over_param:
            self._bias = ManifoldParameter(torch.Tensor(out_features, in_features), manifold=manifold)
        else:
            self._bias = Parameter(torch.Tensor(out_features, 1))
        self.reset_parameters()

    @property
    def weight(self):
        return self.manifold.transp0(self.bias, self._weight)  # weight \in T_0 => weight \in T_bias

    @property
    def bias(self):
        if self.over_param:
            return self._bias
        else:
            return self.manifold.expmap0(self._weight * self._bias)  # reparameterisation of a point on the manifold

    def reset_parameters(self):
        init.kaiming_normal_(self._weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self._weight)
        bound = 4 / math.sqrt(fan_in)
        init.uniform_(self._bias, -bound, bound)
        if self.over_param:
            with torch.no_grad():
                self._bias.set_(self.manifold.expmap0(self._bias))


class GeodesicLayer(RiemannianLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        manifold: geoopt.PoincareBall,
        over_param=False,
        weight_norm=False,
    ):
        super(GeodesicLayer, self).__init__(
            in_features,
            out_features,
            manifold,
            over_param,
            weight_norm,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        logger.debug("input shape: %s", input.shape)
        input = input.expand(
            *input.shape[:-1],
            self.out_features,
            input.shape[-1],
        )
        res = normdist2plane(
            manifold_poincare=self.manifold,
            x=input,
            a=self.bias,
            p=self.weight,
            signed=True,
            norm=self.weight_norm,
        )
        # res = geoopt.manifolds.stereographic.math.dist2plane(
        #     x=input,
        #     p=self.bias,
        #     a=self.weight,
        #     k=-1.0 * self.manifold.c,  # k = -c!!!
        #     keepdim=False,  # don't keep the last dimension
        #     signed=True,  # return the signed distance
        #     scaled=True,  # scale the result by the tangent norm
        #     dim=-1,
        # )
        return res


class ExpMap0(nn.Module):
    def __init__(self, manifold: geoopt.Stereographic):
        super().__init__()
        self.manifold = manifold

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.manifold.expmap0(input)


class MobiusLayer(RiemannianLayer):
    """Taken from https://github.com/emilemathieu/pvae/blob/c04ec2149fc4d37fd83946a366780816c0cbe3c0/pvae/ops/manifold_layers.py#L65"""

    def __init__(self, in_features, out_features, manifold, over_param=False, weight_norm=False):
        super(MobiusLayer, self).__init__(
            in_features,
            out_features,
            manifold,
            over_param,
            weight_norm,
        )

    def forward(self, input):
        res = self.manifold.mobius_matvec(self.weight, input)
        return res
