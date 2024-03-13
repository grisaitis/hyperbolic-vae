import logging
from numbers import Number

import geoopt
import torch
from torch.distributions import Independent, Normal
from torch.distributions.utils import _standard_normal, broadcast_all
from torch.nn import functional as F

from hyperbolic_vae.manifolds import logdetexp

logger = logging.getLogger(__name__)


class WrappedNormal(torch.distributions.Distribution):
    arg_constraints = {"loc": torch.distributions.constraints.real, "scale": torch.distributions.constraints.positive}
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        raise NotImplementedError

    @property
    def scale(self):
        return F.softplus(self._scale) if self.softplus else self._scale

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        manifold: geoopt.PoincareBall,
        validate_args=None,
        softplus=False,
    ):
        self.dtype = loc.dtype
        self.softplus = softplus
        self.loc, self._scale = broadcast_all(loc, scale)
        self.manifold = manifold
        try:
            self.manifold.assert_check_point_on_manifold(self.loc)
        except Exception as e:
            print(self.loc)
            raise e
        self.device = loc.device
        batch_shape = self.loc.shape[:-1]
        event_shape = self.loc.shape[-1:]
        super().__init__(
            batch_shape,
            event_shape,
            validate_args=validate_args,
        )

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        v = self.scale * _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        manifold_zero = self.manifold.origin(self.event_shape, dtype=self.loc.dtype, device=self.device)
        self.manifold.assert_check_vector_on_tangent(manifold_zero, v)
        v = v / self.manifold.lambda_x(manifold_zero, keepdim=True)
        u = self.manifold.transp(manifold_zero, self.loc, v)
        z = self.manifold.expmap(self.loc, u)
        return z

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        manifold_coord_dim = int(self.event_shape[0])
        manifold_zero = self.manifold.origin(self.event_shape, dtype=self.loc.dtype, device=self.device)
        loc = self.loc.unsqueeze(0).expand(x.shape[0], *self.batch_shape, manifold_coord_dim)
        if len(shape) < len(loc.shape):
            x = x.unsqueeze(1)
        v = self.manifold.logmap(loc, x)
        v = self.manifold.transp(loc, manifold_zero, v)
        u = v * self.manifold.lambda_x(manifold_zero, keepdim=True)
        norm_pdf = Normal(torch.zeros_like(self.scale), self.scale).log_prob(u).sum(-1, keepdim=True)
        logdetexp_value = logdetexp(self.manifold, loc, x, keepdim=True)
        result = norm_pdf - logdetexp_value
        return result
