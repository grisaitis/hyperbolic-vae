import pytorch_lightning as pl
import torch.utils.data
from torchvision import transforms
from torchvision.datasets import MNIST

from hyperbolic_vae.config import DATA_PATH

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Pad(2),
        # transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
dataset_train, dataset_val = torch.utils.data.random_split(
    MNIST(DATA_PATH, train=True, download=True, transform=transform),
    lengths=[0.9, 0.1],
    generator=torch.Generator().manual_seed(42),
)

dataset_test = MNIST(DATA_PATH, train=False, download=True, transform=transform)
mnist_data_module = pl.LightningDataModule.from_datasets(
    train_dataset=dataset_train,
    val_dataset=dataset_val,
    test_dataset=dataset_test,
    batch_size=256,
    num_workers=4,
)
