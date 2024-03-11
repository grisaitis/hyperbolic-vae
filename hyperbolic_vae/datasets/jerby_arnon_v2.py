from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from .jerby_arnon import (
    ANNOTATIONS_CSV_GZ_URL,
    TPM_CSV_GZ_URL,
    _download_and_extract_csv_gz,
    _filter_gene_symbols,
    _read_annotations,
    _read_tpm,
)

ANNOTATIONS_CSV_PATH = Path("/home/jupyter/hyperbolic-vae/data/jerby_arnon/manual/GSE115978_cell.annotations.csv")
TPM_CSV_PATH = Path("/home/jupyter/hyperbolic-vae/data/jerby_arnon/manual/GSE115978_tpm.csv")

df_annotations = _read_annotations(ANNOTATIONS_CSV_PATH)
df_tpm = _read_tpm(TPM_CSV_PATH)
df_tpm = _filter_gene_symbols(df_tpm)
rng = np.random.default_rng(seed=42)


def make_dataset_from_dataframe(df: pd.DataFrame) -> torch.utils.data.Dataset:
    class JerbyArnonCSVDataset(torch.utils.data.Dataset):
        def __init__(self, df: pd.DataFrame):
            self.df = df

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {
                "x": torch.tensor(self.df.iloc[idx].values, dtype=torch.float32),
            }

    return JerbyArnonCSVDataset(df)


df_tpm_train, df_tpm_val, df_tpm_test = np.split(
    df_tpm.sample(frac=1, random_state=rng), [int(0.6 * len(df_tpm)), int(0.8 * len(df_tpm))]
)

dataset_train, dataset_val, dataset_test = map(make_dataset_from_dataframe, [df_tpm_train, df_tpm_val, df_tpm_test])

cifar10_data_module = pl.LightningDataModule.from_datasets(
    train_dataset=dataset_train,
    val_dataset=dataset_val,
    test_dataset=dataset_test,
    batch_size=256,
    num_workers=4,
)