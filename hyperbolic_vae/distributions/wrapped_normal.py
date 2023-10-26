import geoopt
import pvae.distributions
import torch
from torch.distributions import Normal
from torch.distributions.utils import broadcast_all

from hyperbolic_vae.manifolds import logdetexp

# work in progress :)
# see pvae.distributions.WrappedNormal
# https://github.com/emilemathieu/pvae/blob/master/pvae/distributions/wrapped_normal.py


class WrappedNormal(pvae.distributions.WrappedNormal):
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
        super(pvae.distributions.WrappedNormal, self).__init__(
            batch_shape,
            event_shape,
            validate_args=validate_args,
        )

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


class WrappedMultivariateNormal(torch.distributions.MultivariateNormal):
    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        validate_args: bool = True,
    ):
        self.loc = loc
        self.scale = scale
        super(torch.distributions.MultivariateNormal, self).__init__(
            batch_shape=loc.shape[:-1], event_shape=loc.shape[-1:], validate_args=validate_args
        )

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> geoopt.ManifoldTensor:
        # see:
        torch.distributions.MultivariateNormal.rsample
        pvae.distributions.WrappedNormal.rsample
        raise NotImplementedError

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # see:
        torch.distributions.MultivariateNormal.log_prob
        pvae.distributions.WrappedNormal.log_prob
        raise NotImplementedError

    def entropy(self) -> torch.Tensor:
        """
        Compute the entropy of the distribution.
        """
        raise NotImplementedError
