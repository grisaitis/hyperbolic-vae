import gzip
import io
import itertools
import logging
import types
import urllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.stats
import torch
from torch.utils.data import Dataset

from hyperbolic_vae.config import DATA_PATH
from hyperbolic_vae.util import ColoredFormatter

logger = logging.getLogger(__name__)

ANNOTATIONS_CSV_GZ_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115978/suppl/GSE115978%5Fcell.annotations.csv.gz"
)
COUNTS_CSV_GZ_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115978/suppl/GSE115978%5Fcounts.csv.gz"
TPM_CSV_GZ_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115978/suppl/GSE115978%5Ftpm.csv.gz"
JERBY_ARNON_DATA_PATH = DATA_PATH / "jerby_arnon"
ANNOTATIONS_CSV_PATH = JERBY_ARNON_DATA_PATH / "annotations.csv"
COUNTS_CSV_PATH = JERBY_ARNON_DATA_PATH / "counts.csv"
TPM_CSV_PATH = JERBY_ARNON_DATA_PATH / "tpm.csv"


columns = types.SimpleNamespace(
    CELL_TYPE="cell_type",
    GENE_SYMBOL="gene_symbol",
    SAMPLE_ID="sample_id",
    SINGLE_CELL_ID="single_cell_id",
)
nice_to_weirds = {
    "Malignant": ["Malignant.cell", "Mal", "Malignant cell"],
    "Endothelial": [
        "Endothelial.cell",
        "Endothelial cells",
        "Endo.",
        "Endothelial cell",
    ],
    "CAF": [],
    "T CD8": ["T.CD8", "T cells CD8", "TCD8"],
    "NK": ["NK cells"],
    "Macrophage": ["Macrophages"],
    "T CD4": ["T.CD4", "T cells CD4", "TCD4"],
    "B": ["B.cell", "B cells", "B cell"],
    "T": ["T.cell", "T cell"],
}
weird_to_nice = {weird: nice for nice in nice_to_weirds for weird in nice_to_weirds[nice]}


class RNASeqAnnotatedDataset(Dataset):
    def __init__(
        self,
        df_rnaseq: pd.DataFrame,
        df_annotations: pd.DataFrame,
        rnaseq_normalize_method: str | None,
    ):
        """
        Initialize the RNASeqAnnotatedDataset object.

        Args:
            df_rnaseq (pd.DataFrame): A DataFrame of shape (n samples, n genes) with RNA-seq counts or transcripts.
            df_annotations (pd.DataFrame): A DataFrame of shape (n samples, annotations) with cell_type and other columns.
            rnaseq_normalize_method (str | None): The method to normalize the gene expression values.
                If "sum_to_one", the gene expression values are normalized to sum to 1.0.
                If "sum_to_million", the gene expression values are normalized to sum to 1,000,000.
                If None, the gene expression values are not normalized.
        """
        assert df_rnaseq.index.name == columns.SINGLE_CELL_ID
        assert df_rnaseq.columns.name == columns.GENE_SYMBOL
        assert df_annotations.index.name == columns.SINGLE_CELL_ID
        assert df_rnaseq.index.equals(df_annotations.index)
        self.df_rnaseq = df_rnaseq
        if rnaseq_normalize_method:
            self.df_rnaseq = normalize_rnaseq(self.df_rnaseq, rnaseq_normalize_method)
        self.df_annotations = df_annotations
        logger.debug("df_rnaseq.iloc[:10, :10]:\n%s", self.df_rnaseq.iloc[:10, :10])
        logger.debug("df_rnaseq.sum(axis=0):\n%s", self.df_rnaseq.sum(axis=0))
        logger.debug("df_rnaseq.sum(axis=1):\n%s", self.df_rnaseq.sum(axis=1))
        logger.debug("df_annotations.head():\n%s", self.df_annotations.head())

    def __len__(self):
        return len(self.df_annotations)

    def __getitem__(self, idx) -> tuple[torch.Tensor, pd.Series]:
        rnaseq = torch.tensor(self.df_rnaseq.iloc[idx].values, dtype=torch.float)
        cell_type_series = self.df_annotations[columns.CELL_TYPE].iloc[idx]
        return rnaseq, cell_type_series


def normalize_rnaseq(df_rnaseq: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "sum_to_one":
        return df_rnaseq.div(df_rnaseq.sum(axis=1), axis=0)
    elif method == "sum_to_million":
        return df_rnaseq.div(df_rnaseq.sum(axis=1), axis=0) * 1_000_000
    elif method == "z_score":
        # see https://stackoverflow.com/a/41713622/781938
        return df_rnaseq.apply(scipy.stats.zscore)
    else:
        raise ValueError(f"rnaseq_normalize_method {method} not recognized")


def _read_annotations(path_csv: Path) -> pd.DataFrame:
    """Returns a pd.DataFrame with cells as rows, annotations as columns."""
    logger.info(f"Reading annotations from {path_csv}")
    df = pd.read_csv(
        path_csv,
        na_values={"cell.types": "?"},
    )
    logger.info(f"Read annotations from {path_csv}")
    df = df.rename(
        columns={
            "cells": columns.SINGLE_CELL_ID,
            "cell.types": columns.CELL_TYPE,
            "samples": columns.SAMPLE_ID,
        }
    )
    df[columns.CELL_TYPE] = df[columns.CELL_TYPE].fillna("Unknown")
    logger.info("Renamed columns")
    df = df.replace({columns.CELL_TYPE: weird_to_nice})
    df = df.rename_axis(index=columns.SINGLE_CELL_ID)
    df = df.set_index(columns.SINGLE_CELL_ID, drop=False)
    df = df.sort_index()
    logger.info("Sorted index")
    return df


def _read_tpm(path_csv: Path, skiprows=None) -> pd.DataFrame:
    """Returns a pd.DataFrame with cells as rows, genes as columns."""
    logger.info("reading TPM from %s", path_csv)
    if skiprows is None:
        logger.debug("reading tpm csv with pyarrow")
        df = pd.read_csv(path_csv, engine="pyarrow", index_col=0)
    else:
        logger.debug("reading tpm csv with skiprows")
        df = pd.read_csv(path_csv, index_col=0, skiprows=skiprows)
    logger.info("renaming and sorting index")
    df = df.rename_axis(index=columns.GENE_SYMBOL, columns=columns.SINGLE_CELL_ID)
    df = df.sort_index(axis="columns")
    df = df.sort_index(axis="rows")
    logger.info("transposing")
    df = df.T
    logger.info("shape: %s", df.shape)
    return df


def _filter_gene_symbols(df_tpm: pd.DataFrame) -> pd.DataFrame:
    # remove mitochondrial gene symbols
    mitochondrial_gene_symbols = df_tpm.columns[df_tpm.columns.str.startswith("MT")]
    logger.debug("dropping %s mitochondrial gene symbols", len(mitochondrial_gene_symbols))
    df_tpm = df_tpm.drop(columns=mitochondrial_gene_symbols)
    # identify and remove gene symbols which are zero more than 90% of the time
    logger.debug("gene zero rates with eq: %s", df_tpm.eq(0).mean())
    logger.debug("gene zero rates with (df == 0): %s", (df_tpm == 0).mean())
    zero_genes = df_tpm.columns[df_tpm.eq(0).mean() > 0.9]
    logger.debug("dropping %s gene symbols which are zero more than 90%% of the time", len(zero_genes))
    df_tpm = df_tpm.drop(columns=zero_genes)
    return df_tpm


def _filter_single_cells(df_rnaseq: pd.DataFrame) -> pd.DataFrame:
    # remove single cells with more than 90% zero gene expression values
    very_sparse_single_cells = df_rnaseq.index[df_rnaseq.eq(0).mean(axis=1) > 0.9]
    logger.debug(
        "dropping %s single cells with more than 90%% zero gene expression values", len(very_sparse_single_cells)
    )
    df_rnaseq = df_rnaseq.drop(index=very_sparse_single_cells)
    return df_rnaseq


def get_pytorch_dataset(rnaseq_normalize_method: str | None) -> RNASeqAnnotatedDataset:
    df_annotations = _read_annotations(ANNOTATIONS_CSV_PATH)
    df_rnaseq = _read_tpm(TPM_CSV_PATH)
    df_rnaseq = _filter_gene_symbols(df_rnaseq)
    # df_rnaseq = _filter_single_cells(df_rnaseq)
    return RNASeqAnnotatedDataset(df_rnaseq, df_annotations, rnaseq_normalize_method)


def get_subset_jerby_arnon_dataset(
    n_samples: int = 10,
    genes_keep_one_in: int = 100,
    rnaseq_normalize_method: str | None = "sum_to_one",
) -> RNASeqAnnotatedDataset:
    df_annotations = _read_annotations(ANNOTATIONS_CSV_PATH)
    df_rnaseq = _read_tpm(TPM_CSV_PATH, skiprows=lambda i: i % genes_keep_one_in)
    df_rnaseq = _filter_gene_symbols(df_rnaseq)
    samples_to_keep = df_annotations.index[:n_samples]
    df_rnaseq = df_rnaseq.loc[samples_to_keep]
    df_annotations = df_annotations.loc[samples_to_keep]
    return RNASeqAnnotatedDataset(df_rnaseq, df_annotations, rnaseq_normalize_method)


def make_fake_dataframes(
    n_samples: int = 1000,
    n_genes: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    single_cell_id_index = pd.Index([f"cell_{i}" for i in range(n_samples)], name=columns.SINGLE_CELL_ID)
    gene_symbol_index = pd.Index([f"gene_{i:05d}" for i in range(n_genes)], name=columns.GENE_SYMBOL)
    rng = np.random.default_rng(42)
    rnaseq_tpm = rng.poisson(100, size=(n_samples, n_genes))
    df_rnaseq = pd.DataFrame(rnaseq_tpm, index=single_cell_id_index, columns=gene_symbol_index)
    cell_types = rng.choice(list(nice_to_weirds), size=n_samples)
    df_annotations = pd.DataFrame({columns.CELL_TYPE: cell_types}, index=single_cell_id_index)
    return df_rnaseq, df_annotations


def get_fake_dataset(
    n_samples: int = 1000,
    n_genes: int = 2000,
    rnaseq_normalize_method: str | None = "sum_to_one",
) -> RNASeqAnnotatedDataset:
    df_rnaseq, df_annotations = make_fake_dataframes(n_samples, n_genes)
    return RNASeqAnnotatedDataset(df_rnaseq, df_annotations, rnaseq_normalize_method)


def _download_and_extract_csv_gz(url: str, save_path: Path) -> None:
    with urllib.request.urlopen(url) as response:
        compressed_file = io.BytesIO(response.read())
    with gzip.open(compressed_file, "rb") as gz:
        decompressed_content = gz.read()
    with open(save_path, "wb") as f_out:
        f_out.write(decompressed_content)


def make_rnaseq_data_module(
    dataset: RNASeqAnnotatedDataset, batch_size: int, num_workers: int
) -> pl.LightningDataModule:
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    data_module = pl.LightningDataModule.from_datasets(
        train_dataset=dataset_train,
        val_dataset=dataset_val,
        test_dataset=dataset_test,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return data_module


if __name__ == "__main__":
    # logging.getLogger("hyperbolic_vae").setLevel("INFO")
    # logging.getLogger("hyperbolic_vae.models.vae_hyperbolic_rnaseq").setLevel("DEBUG")
    logging.getLogger().setLevel("INFO")
    sh = logging.StreamHandler()
    sh.setFormatter(ColoredFormatter("%(asctime)s %(name)s %(funcName)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(sh)

    # dataset = get_fake_dataset(5, 10)
    dataset = get_subset_jerby_arnon_dataset(100, 100, "sum_to_one")
    print(dataset.df_rnaseq)
    print("type", type(dataset.df_rnaseq))
    print("index names", dataset.df_rnaseq.index.name, dataset.df_rnaseq.columns.name)
    print(dataset.df_annotations)
    print("type", type(dataset.df_annotations))
    print("index names", dataset.df_annotations.index.name)
