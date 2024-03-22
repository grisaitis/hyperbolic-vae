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

    def forward(self, input: torch.Tensor) -> geoopt.ManifoldTensor:
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


class Distance2PoincareHyperplanes(torch.nn.Module):
    """
    source:
    https://github.com/geoopt/geoopt/blob/master/examples/hyperbolic_multiclass_classification.ipynb
    """

    n = 0
    # 1D, 2D versions of this class ara available with a one line change
    # class Distance2PoincareHyperplanes2d(Distance2PoincareHyperplanes):
    #     n = 2

    def __init__(
        self,
        plane_shape: int,
        num_planes: int,
        signed=True,
        squared=False,
        *,
        ball: geoopt.PoincareBall,
        std=1.0,
    ):
        super().__init__()
        self.signed = signed
        self.squared = squared
        # Do not forget to save Manifold instance to the Module
        self.ball = ball
        self.plane_shape = geoopt.utils.size2shape(plane_shape)
        self.num_planes = num_planes

        # In a layer we create Manifold Parameters in the same way we do it for
        # regular pytorch Parameters, there is no difference. But geoopt optimizer
        # will recognize the manifold and adjust to it
        self.points = geoopt.ManifoldParameter(torch.empty(num_planes, plane_shape), manifold=self.ball)
        self.std = std
        # following best practives, a separate method to reset parameters
        self.reset_parameters()

    def forward(self, input):
        input_p = input.unsqueeze(-self.n - 1)
        points = self.points.permute(1, 0)
        points = points.view(points.shape + (1,) * self.n)

        distance = self.ball.dist2plane(x=input_p, p=points, a=points, signed=self.signed, dim=-self.n - 2)
        if self.squared and self.signed:
            sign = distance.sign()
            distance = distance**2 * sign
        elif self.squared:
            distance = distance**2
        return distance

    def extra_repr(self):
        return "plane_shape={plane_shape}, " "num_planes={num_planes}, ".format(**self.__dict__)

    @torch.no_grad()
    def reset_parameters(self):
        direction = torch.randn_like(self.points)
        direction /= direction.norm(dim=-1, keepdim=True)
        distance = torch.empty_like(self.points[..., 0]).normal_(std=self.std)
        self.points.set_(self.ball.expmap0(direction * distance.unsqueeze(-1)))