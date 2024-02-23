import torch

locs = torch.Tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
covariance_matrices = torch.Tensor(
    [
        [
            [0.1, 0.0],
            [0.0, 0.1],
        ],
        [
            [0.1, 0.0],
            [0.0, 0.1],
        ],
        [
            [0.1, 0.0],
            [0.0, 0.1],
        ],
    ]
)
print(covariance_matrices)

normal_dist = torch.distributions.MultivariateNormal(
    loc=locs, covariance_matrix=covariance_matrices
)  # , batch_shape=batch_shape, event_shape=event_shape)
samples = normal_dist.rsample()
print(samples.shape)
# print(samples)

log_prob_samples = normal_dist.log_prob(samples)
print(log_prob_samples.shape)
# print(log_prob_samples)
