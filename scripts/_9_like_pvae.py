'''
data loader...
- see https://github.com/emilemathieu/pvae/blob/c04ec2149fc4d37fd83946a366780816c0cbe3c0/pvae/models/mnist.py#L60
'''

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 64
device = "cuda"
shuffle = True

tx = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda p: p.clamp(Constants.eta, 1 - Constants.eta))
])
kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
train_loader = DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=tx),
    batch_size=batch_size, shuffle=shuffle, **kwargs)
test_loader = DataLoader(
    datasets.MNIST('data', train=False, download=True, transform=tx),
    batch_size=batch_size, shuffle=shuffle, **kwargs)

data_size = torch.Size([1, 28, 28])

'''
then...

model.init_last_layer_bias(train_loader)
model.train()
for epoch in range(1, args.epochs + 1):
    for i, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        qz_x, px_z, lik, kl, loss = loss_function(model, data, K=args.K, beta=args.beta, components=True, analytical_kl=args.analytical_kl)
        probe_infnan(loss, "Training loss:")
        loss.backward()
        optimizer.step()
'''
