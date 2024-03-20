import gzip
import io
import itertools
import types
import urllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset

from hyperbolic_vae.config import DATA_PATH
from hyperbolic_vae.datasets.jerby_arnon import (
    ANNOTATIONS_CSV_GZ_URL,
    TPM_CSV_GZ_URL,
    _filter_gene_symbols,
    _read_annotations,
    _read_tpm,
)


def _download_and_extract_csv_gz(url: str, save_path: Path) -> None:
    with urllib.request.urlopen(url) as response:
        compressed_file = io.BytesIO(response.read())
    with gzip.open(compressed_file, "rb") as gz:
        decompressed_content = gz.read()
    with open(save_path, "wb") as f_out:
        f_out.write(decompressed_content)


def _save_split_parquet_datasets(
    df_annotations: pd.DataFrame,
    df_tpm: pd.DataFrame,
    rng: np.random.Generator,
    parquet_datasets_path: Path,
) -> None:
    # sample 0.6 train, 0.2 val, 0.2 test
    parquet_datasets_path.mkdir(exist_ok=True)
    all_cell_ids = list(df_annotations.index)
    all_cell_ids = rng.permutation(all_cell_ids)
    n_train, n_val = int(0.6 * len(all_cell_ids)), int(0.2 * len(all_cell_ids))
    n_test = len(all_cell_ids) - n_train - n_val
    split_cell_ids = {
        "train": all_cell_ids[:n_train],
        "val": all_cell_ids[n_train : n_train + n_val],
        "test": all_cell_ids[-n_test:],
    }
    for split, cell_ids in split_cell_ids.items():
        df_annotations_split = df_annotations.loc[cell_ids]
        df_tpm_split = df_tpm.loc[cell_ids]
        df_annotations_split.to_parquet(parquet_datasets_path / f"{split}_annotations.parquet")
        df_tpm_split.to_parquet(parquet_datasets_path / f"{split}_rnaseq.parquet")


class JerbyArnonCSVDataModule(pl.LightningDataModule):
    def __init__(self, data_path: Path, batch_size: int, num_workers: int):
        super().__init__()
        self.data_path = data_path
        assert self.data_path.exists(), f"Path {self.data_path} does not exist."
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        """
        Prepare data by downloading, extracting, cleaning, saving splits. Called only once globally.
        """
        for url, csv_filename in [
            (ANNOTATIONS_CSV_GZ_URL, "annotations.csv"),
            (TPM_CSV_GZ_URL, "tpm.csv"),
        ]:
            csv_path = self.data_path / csv_filename
            if not csv_path.exists():
                _download_and_extract_csv_gz(url, csv_path)
        parquet_datasets_path = self.data_path / "parquet_datasets"
        if not parquet_datasets_path.exists():
            rng = np.random.default_rng(seed=42)
            df_annotations = _read_annotations(self.data_path / "annotations.csv")
            df_tpm = _read_tpm(self.data_path / "tpm.csv")
            df_tpm = _filter_gene_symbols(df_tpm)
            _save_split_parquet_datasets(df_annotations, df_tpm, rng, parquet_datasets_path)

    def setup(self, stage: str) -> None:
        """
        Set state that is specific to each process, such as splitting data into train/validation/test sets.
        """
        parquet_datasets_path = self.data_path / "parquet_datasets"
        if stage == "fit" or stage is None:
            self.df_train_rnaseq = pd.read_parquet(parquet_datasets_path / "train_rnaseq.parquet")
            self.df_train_annotations = pd.read_parquet(parquet_datasets_path / "train_annotations.parquet")
            self.df_val_rnaseq = pd.read_parquet(parquet_datasets_path / "val_rnaseq.parquet")
            self.df_val_annotations = pd.read_parquet(parquet_datasets_path / "val_annotations.parquet")
        if stage == "test" or stage is None:
            self.df_test_rnaseq = pd.read_parquet(parquet_datasets_path / "test_rnaseq.parquet")
            self.df_test_annotations = pd.read_parquet(parquet_datasets_path / "test_annotations.parquet")


# class JerbyArnonCSVDataset(torch.utils.data.Dataset):
#     def __init__(self):
#         """
#         Initialize the JerbyArnonCSVDataset.

#         Args:
#             annotations (str): The URL of the gzip compressed CSV file containing annotations for each cell.
#             counts (str): The URL of the gzip compressed CSV file containing counts for each gene symbol.
#             tpm (str): The URL of the gzip compressed CSV file containing TPM values for each gene symbol.

#         """
#         self.annotations = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE115978&format=file&file=GSE115978%5Fcell%2Eannotations%2Ecsv%2Egz"
#         self.counts = (
#             "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE115978&format=file&file=GSE115978%5Fcounts%2Ecsv%2Egz"
#         )
#         self.tpm = (
#             "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE115978&format=file&file=GSE115978%5Ftpm%2Ecsv%2Egz"
#         )

#         # Load the data
#         self.annotations_data = self._load_data(self.annotations)
#         self.counts_data = self._load_data(self.counts)
#         self.tpm_data = self._load_data(self.tpm)

#     def _load_data(self, url: str) -> pd.DataFrame:
#         """
#         This method is used to load data from a given URL. The data is expected to be a gzip compressed CSV file.

#         Parameters:
#         url (str): The URL of the gzip compressed CSV file.

#         Returns:
#         pd.DataFrame: The loaded data as a pandas DataFrame.
#         """
#         # Download and load the gzip compressed CSV file
#         with gzip.open(url, "rt") as file:
#             data = pd.read_csv(file, engine="pyarrow")

#         return data

#     def train_val_test_split(self, train_size, val_size, test_size, stratify):
#         # Split the data into train, validation, and test sets
#         # The split is stratified based on the given column
#         return


# dataset = JerbyArnonCSVDataset(
#     )

# dataset_train, dataset_val, dataset_test = dataset.train_val_test_split(
#     train_size=0.8, val_size=0.1, test_size=0.1, stratify="time"
# )

# scrnaseq_data_module = pl.LightningDataModule.from_datasets(
#     train_dataset=dataset_train,
#     val_dataset=dataset_val,
#     test_dataset=dataset_test,
#     batch_size=256,
#     num_workers=4,
# )


if __name__ == "__main__":
    path_jerby_arnon = DATA_PATH / "jerby_arnon"
    path_jerby_arnon.mkdir(exist_ok=True)
    _download_and_extract_csv_gz(
        ANNOTATIONS_CSV_GZ_URL,
        path_jerby_arnon / "annotations.csv",
    )
    _download_and_extract_csv_gz(
        TPM_CSV_GZ_URL,
        path_jerby_arnon / "tpm.csv",
    )
