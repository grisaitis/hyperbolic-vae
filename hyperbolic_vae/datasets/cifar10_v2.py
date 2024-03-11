import pytorch_lightning as pl
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

from hyperbolic_vae.config import DATA_PATH

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset_train, dataset_val = torch.utils.data.random_split(
    CIFAR10(DATA_PATH, train=True, download=True, transform=transform),
    [45000, 5000],
)
dataset_test = CIFAR10(DATA_PATH, train=False, download=True, transform=transform)
cifar10_data_module = pl.LightningDataModule.from_datasets(
    train_dataset=dataset_train,
    val_dataset=dataset_val,
    test_dataset=dataset_test,
    batch_size=256,
    num_workers=4,
)
