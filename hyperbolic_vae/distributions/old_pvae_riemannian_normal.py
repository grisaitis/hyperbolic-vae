import logging

import geoopt
import pvae.distributions
import torch
from pvae.distributions.hyperbolic_radius import HyperbolicRadius
from pvae.distributions.hyperspherical_uniform import HypersphericalUniform

logger = logging.getLogger(__name__)


class RiemannianNormal(pvae.distributions.RiemannianNormal):
    """doesn't work due to issues with HyperbolicRadius"""

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        manifold: geoopt.Stereographic,
        validate_args=None,
    ):
        assert not (torch.isnan(loc).any() or torch.isnan(scale).any())
        batch_shape = loc.shape[:-1]
        event_shape = loc.shape[-1:]
        logger.debug("batch_shape: %s", batch_shape)
        logger.debug("event_shape: %s", event_shape)
        self.manifold = manifold
        self.loc = loc
        self.manifold.assert_check_point_on_manifold(self.loc)
        self.scale = scale.clamp(min=0.1, max=7.0)
        self.radius = HyperbolicRadius(event_shape[0], manifold.c, self.scale)
        self.direction = HypersphericalUniform(event_shape[0] - 1, device=loc.device)
        logger.debug(
            "calling super init with batch_shape %s, event_shape %s",
            batch_shape,
            event_shape,
        )
        super(pvae.distributions.RiemannianNormal, self).__init__(
            batch_shape,
            event_shape,
            validate_args=validate_args,
        )

    def rsample(self, sample_shape=torch.Size()):
        shape_alpha = sample_shape + self.batch_shape
        logger.debug("shape_alpha: %s", shape_alpha)
        alpha = self.direction.sample(shape_alpha)
        radius = self.radius.rsample(sample_shape)
        # u = radius * alpha / self.manifold.lambda_x(self.loc, keepdim=True)
        # res = self.manifold.expmap(self.loc, u)
        res = self.manifold.expmap_polar(self.loc, alpha, radius)
        return res


if __name__ == "__main__":
    import geoopt
    import torch

    event_shape = torch.Size([2])
    manifold = geoopt.PoincareBall(c=1.0)
    scale = torch.Tensor([[1.0, 1.0]])
    sample_shape = torch.Size([1])

    radius = HyperbolicRadius(event_shape[0], manifold.c, scale)
    x = radius.sample(sample_shape)  # fails

    # fails
    HyperbolicRadius(dim=2, c=torch.Tensor(1.0), scale=torch.Tensor([[1.0, 1.0]])).sample(torch.Size([1]))
    # also fails
    HyperbolicRadius(dim=2, c=torch.tensor(2.0), scale=torch.Tensor([[1.0, 1.0]])).sample(torch.Size([1]))
