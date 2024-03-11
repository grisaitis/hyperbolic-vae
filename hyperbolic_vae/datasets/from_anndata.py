import anndata as a
import pytorch_lightning as pl
import torch_adata
from torchvision import transforms

adata = a.read_h5ad("/path/to/data/pbmcs.h5ad")
dataset = torch_adata.AnnDataset(adata, use_key="X_pca", groupby="time", obs_keys=["affinity"])

dataset_train, dataset_val, dataset_test = dataset.train_val_test_split(
    train_size=0.8, val_size=0.1, test_size=0.1, stratify="time"
)

scrnaseq_data_module = pl.LightningDataModule.from_datasets(
    train_dataset=dataset_train,
    val_dataset=dataset_val,
    test_dataset=dataset_test,
    batch_size=256,
    num_workers=4,
)
