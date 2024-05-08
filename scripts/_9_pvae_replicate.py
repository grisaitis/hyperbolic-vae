"""
MNIST pvae...
- batch

model: Mnist(
  (enc): EncWrapped(
    (manifold): R(ndim=1) manifold
    (enc): Sequential(
      (0): Sequential(
        (0): Linear(in_features=784, out_features=600, bias=True)
        (1): ReLU()
      )
    )
    (fc21): Linear(in_features=600, out_features=2, bias=True)
    (fc22): Linear(in_features=600, out_features=2, bias=True)
  )
  (dec): DecBernouilliWrapper(
    (dec): DecWrapped(
      (manifold): R(ndim=1) manifold
      (dec): Sequential(
        (0): Sequential(
          (0): Linear(in_features=2, out_features=600, bias=True)
          (1): ReLU()
        )
      )
      (fc31): Linear(in_features=600, out_features=784, bias=True)
    )
  )
  (manifold): R(ndim=1) manifold
"""

"""
approach for implementing this...
- i have a working implementation for older versions of pytorch, pytorch-lighting, and geoopt
- i need to update the code to work with the latest versions of the libraries
a few approaches for making this work for the latest versions of the libraries...
 
*******************************************************************************************
*********        LEFT OFF HERE         ****************************************************
*******************************************************************************************
"""






# pvae/models/mnist.py


def get_mean_param(params):
    """Return the parameter used to show reconstructions or generations.
    For example, the mean for Normal, or probs for Bernoulli.
    For Bernoulli, skip first parameter, as that's (scalar) temperature
    """
    if params[0].dim() == 0:
        return params[1]
    # elif len(params) == 3:
    #     return params[1]
    else:
        return params[0]


data_size = torch.Size([1, 28, 28])

class VAE(nn.Module):
    def __init__(self, prior_dist, posterior_dist, likelihood_dist, enc, dec, params):
        super(VAE, self).__init__()
        self.pz = prior_dist
        self.px_z = likelihood_dist
        self.qz_x = posterior_dist
        self.enc = enc
        self.dec = dec
        self.modelName = None
        self.params = params
        self.data_size = params.data_size
        self.prior_std = params.prior_std

        if self.px_z == dist.RelaxedBernoulli:
            self.px_z.log_prob = lambda self, value: \
                -F.binary_cross_entropy_with_logits(
                    self.probs if value.dim() <= self.probs.dim() else self.probs.expand_as(value),
                    value.expand(self.batch_shape) if value.dim() <= self.probs.dim() else value,
                    reduction='none'
                )

    def getDataLoaders(self, batch_size, shuffle, device, *args):
        raise NotImplementedError

    def generate(self, N, K):
        self.eval()
        with torch.no_grad():
            mean_pz = get_mean_param(self.pz_params)
            mean = get_mean_param(self.dec(mean_pz))
            px_z_params = self.dec(self.pz(*self.pz_params).sample(torch.Size([N])))
            means = get_mean_param(px_z_params)
            samples = self.px_z(*px_z_params).sample(torch.Size([K]))

        return mean, \
            means.view(-1, *means.size()[2:]), \
            samples.view(-1, *samples.size()[3:])

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(data))
            px_z_params = self.dec(qz_x.rsample(torch.Size([1])).squeeze(0))

        return get_mean_param(px_z_params)

    def forward(self, x, K=1):
        qz_x = self.qz_x(*self.enc(x))
        zs = qz_x.rsample(torch.Size([K]))
        px_z = self.px_z(*self.dec(zs))
        return qz_x, px_z, zs

    @property
    def pz_params(self):
        return self._pz_mu.mul(1), F.softplus(self._pz_logvar).div(math.log(2)).mul(self.prior_std_scale)

    def init_last_layer_bias(self, dataset): pass


class Mnist(VAE):
    def __init__(self, params):
        c = nn.Parameter(params.c * torch.ones(1), requires_grad=False)
        manifold = getattr(manifolds, params.manifold)(params.latent_dim, c)
        super(Mnist, self).__init__(
            eval(params.prior),   # prior distribution
            eval(params.posterior),   # posterior distribution
            dist.RelaxedBernoulli,        # likelihood distribution
            eval('Enc' + params.enc)(manifold, data_size, getattr(nn, params.nl)(), params.num_hidden_layers, params.hidden_dim, params.prior_iso),
            DecBernouilliWrapper(eval('Dec' + params.dec)(manifold, data_size, getattr(nn, params.nl)(), params.num_hidden_layers, params.hidden_dim)),
            params
        )
        self.manifold = manifold
        self.c = c
        self._pz_mu = nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False)
        self._pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=params.learn_prior_std)
        self.modelName = 'Mnist'

    def init_last_layer_bias(self, train_loader):
        if not hasattr(self.dec.dec.fc31, 'bias'): return
        with torch.no_grad():
            p = torch.zeros(prod(data_size[1:]), device=self._pz_mu.device)
            N = 0
            for i, (data, _) in enumerate(train_loader):
                data = data.to(self._pz_mu.device)
                B = data.size(0)
                N += B
                p += data.view(-1, prod(data_size[1:])).sum(0)
            p /= N
            p += 1e-4
            self.dec.dec.fc31.bias.set_(p.log() - (1 - p).log())

    @property
    def pz_params(self):
        return self._pz_mu.mul(1), F.softplus(self._pz_logvar).div(math.log(2)).mul(self.prior_std), self.manifold

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        # this is required if using the relaxedBernoulli because it doesn't
        # handle scoring values that are actually 0. or 1.
        tx = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda p: p.clamp(Constants.eta, 1 - Constants.eta))
        ])
        train_loader = DataLoader(
            datasets.MNIST('data', train=True, download=True, transform=tx),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(
            datasets.MNIST('data', train=False, download=True, transform=tx),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train_loader, test_loader

    def generate(self, runPath, epoch):
        N, K = 64, 9
        mean, means, samples = super(Mnist, self).generate(N, K)
        save_image(mean.data.cpu(), '{}/gen_mean_{:03d}.png'.format(runPath, epoch))
        save_image(means.data.cpu(), '{}/gen_means_{:03d}.png'.format(runPath, epoch))

    def reconstruct(self, data, runPath, epoch):
        recon = super(Mnist, self).reconstruct(data[:8])
        comp = torch.cat([data[:8], recon])
        save_image(comp.data.cpu(), '{}/recon_{:03d}.png'.format(runPath, epoch))



def train(
    dataset: Dataset,
    batch_size: int,
    latent_dim: int,
    manifold_curvature: float,
    hidden_layer_dim: int,
    learning_rate: float,
    beta: float,
):

if __name__ == "__main__":
    train(
        dataset=mnist_digits,
        batch_size=128,
        latent_dim=2,
        manifold_curvature=0.0,
        hidden_layer_dim=600,
        learning_rate=5e-4,
        beta=1.0,
        iwae_samples=5000,
        n_epochs=80,
    )