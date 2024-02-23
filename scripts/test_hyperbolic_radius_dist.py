import logging

import geoopt
import torch
from pvae.distributions.hyperbolic_radius import HyperbolicRadius

from hyperbolic_vae.util import ColoredFormatter

logging.getLogger("hyperbolic_vae").setLevel("DEBUG")
logging.getLogger("pvae").setLevel("DEBUG")
sh = logging.StreamHandler()
sh.setFormatter(ColoredFormatter("%(asctime)s %(name)s %(funcName)s %(levelname)s %(message)s"))
logging.getLogger().addHandler(sh)


event_shape = torch.Size([2])
manifold = geoopt.PoincareBall(c=1.0)
scale = torch.Tensor([[1.0, 1.0]])
sample_shape = torch.Size([1])

radius = HyperbolicRadius(event_shape[0], manifold.c, scale)

x = radius.sample(sample_shape)

print(x)
