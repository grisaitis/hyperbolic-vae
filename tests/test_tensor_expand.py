import torch

shape = torch.Size([5, 2])

x = torch.randn(shape)
print(x.shape)
print(x)

x = x.unsqueeze(-2)
print(x.shape)
print(x)

x = x.expand(
    *x.shape[:-1],
    3,
    2,
)

print(x.shape)
print(x)
