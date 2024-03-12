import pvae.manifolds
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n = 5
d = 2

zero = torch.zeros((n, d)).to(device)
print(zero)

# ball = geoopt.PoincareBall(c=1.0)
ball = pvae.manifolds.PoincareBall(dim=d, c=1.0)

ball.assert_check_point_on_manifold(zero)

ones = torch.ones_like(zero)
print(ones)
ones_ball = ball.expmap0(ones)
print(ones_ball)

ball.assert_check_point_on_manifold(ones_ball)

# n normally distributed 2d samples with mean 0 and std 1
# random number generator
rng = torch.Generator(device=device)
rng.manual_seed(42)
z = torch.randn(n, d, generator=rng, device=device)
print(z)
z_ball = ball.expmap0(z)
print(z_ball)
ball.assert_check_point_on_manifold(z_ball)
