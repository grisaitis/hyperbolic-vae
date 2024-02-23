import geoopt
import torch
from geoopt.manifolds.stereographic.math import dist2plane
from networkx import out_degree_centrality

manifold = geoopt.PoincareBall(c=1.0)

batch_size = 5
dim = 2
shape = torch.Size([batch_size, dim])

x = torch.randn(shape)
x = manifold.expmap0(x)

print(x.shape)
print(x)

biases = torch.randn(dim)
biases = manifold.expmap0(biases)
print(biases.shape)

in_features = dim
out_features = 7
weights = torch.randn(in_features, out_features)
print(weights.shape)

y = dist2plane(
    x=x,
    p=biases,
    a=weights,
    k=-1.0 * manifold.c,
    signed=True,
)

print(y.shape)
