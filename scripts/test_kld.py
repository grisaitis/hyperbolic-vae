import pvae.distributions
import torch

from hyperbolic_vae.manifolds import PoincareBallWithExtras

# manifold = geoopt.PoincareBall(c=1.0)
manifold = PoincareBallWithExtras(c=1.0)
d = 2

# define p(z)
mu_pz = manifold.origin(torch.Size([d]))
print("mu_pz", type(mu_pz), mu_pz)
log_var_pz = torch.exp(0.5 * torch.zeros_like(mu_pz))
print("log_var_pz", type(log_var_pz), log_var_pz)
pz = pvae.distributions.WrappedNormal(mu_pz, log_var_pz, manifold)
print("pz", type(pz), pz)

# define q(z|x)
# mu_qz_x = manifold.expmap0(torch.Tensor([2.0, 2.0]))
mu_qz_x = manifold.expmap0(torch.Tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]))
print("mu_qz_x", type(mu_qz_x), mu_qz_x)
log_var_qz_x = torch.exp(0.5 * torch.Tensor([[-2.0, -2.0], [-2.0, -2.0], [-2.0, -2.0]]))
print("log_var_qz_x", type(log_var_qz_x), log_var_qz_x)
qz_x = pvae.distributions.WrappedNormal(mu_qz_x, log_var_qz_x, manifold)
print("qz_x", type(qz_x), qz_x)

# generate samples
N = 1
samples_qz_x = qz_x.rsample(torch.Size([N]))
print("samples_qz_x", type(samples_qz_x), samples_qz_x.shape)
samples_pz = pz.rsample(torch.Size([N]))
print("samples_pz", type(samples_pz), samples_pz.shape)

# compute log_probs
log_prob_samples_qz_x = qz_x.log_prob(samples_qz_x)
print("log_prob_samples_qz_x", type(log_prob_samples_qz_x), log_prob_samples_qz_x)
log_prob_samples_pz = pz.log_prob(samples_pz)
print("log_prob_samples_pz", type(log_prob_samples_pz), log_prob_samples_pz)

kl_qz_x_from_pz = log_prob_samples_qz_x.exp() * (log_prob_samples_qz_x - log_prob_samples_pz)
print("kl_qz_x_from_pz", type(kl_qz_x_from_pz), kl_qz_x_from_pz)

kl_qz_x_from_pz_mean = kl_qz_x_from_pz.mean()
kl_qz_x_from_pz_std = kl_qz_x_from_pz.std()
print("kl_qz_x_from_pz_mean", type(kl_qz_x_from_pz_mean), kl_qz_x_from_pz_mean)
print("kl_qz_x_from_pz_std", type(kl_qz_x_from_pz_std), kl_qz_x_from_pz_std)
