import torch
from geoopt import ManifoldTensor
from geoopt.manifolds import PoincareBall


class PoincareBallWithExtras(PoincareBall):
    def logdetexp(self, x: ManifoldTensor, y: ManifoldTensor, is_vector=False, keepdim=False):
        if is_vector:
            d = self.norm(x, y, keepdim=keepdim)
        else:
            d = self.dist(x, y, keepdim=keepdim)
        dim = x.shape[-1]
        return (dim - 1) * (torch.sinh(self.c.sqrt() * d) / self.c.sqrt() / d).log()
