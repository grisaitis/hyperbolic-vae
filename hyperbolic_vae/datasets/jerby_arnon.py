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
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

ANNOTATIONS_CSV_GZ_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115978/suppl/GSE115978%5Fcell.annotations.csv.gz"
)
COUNTS_CSV_GZ_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115978/suppl/GSE115978%5Fcounts.csv.gz"
TPM_CSV_GZ_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE115nnn/GSE115978/suppl/GSE115978%5Ftpm.csv.gz"
ANNOTATIONS_CSV_PATH = Path("/home/jupyter/hyperbolic-vae/data/jerby_arnon/manual/GSE115978_cell.annotations.csv")
TPM_CSV_PATH = Path("/home/jupyter/hyperbolic-vae/data/jerby_arnon/manual/GSE115978_tpm.csv")


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
    ):
        """
        Initialize the RNASeqAnnotatedDataset object.

        Args:
            df_rnaseq (pd.DataFrame): A DataFrame of shape (n samples, n genes) with RNA-seq counts or transcripts.
            df_annotations (pd.DataFrame): A DataFrame of shape (n samples, annotations) with cell_type and other columns.
        """
        assert df_rnaseq.index.name == columns.SINGLE_CELL_ID
        assert df_rnaseq.columns.name == columns.GENE_SYMBOL
        assert df_annotations.index.name == columns.SINGLE_CELL_ID
        assert df_rnaseq.index.equals(df_annotations.index)
        self.df_rnqseq = df_rnaseq
        self.df_annotations = df_annotations

    def __len__(self):
        return len(self.df_annotations)

    def __getitem__(self, idx):
        return {
            "rnaseq": torch.tensor(self.df_rnqseq.iloc[idx], dtype=torch.float),
            "cell_type_series": self.df_annotations[columns.CELL_TYPE].iloc[idx],
        }


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
    logger.info("Renamed columns")
    df = df.replace({columns.CELL_TYPE: weird_to_nice})
    df = df.rename_axis(index=columns.SINGLE_CELL_ID)
    df = df.set_index(columns.SINGLE_CELL_ID, drop=False)
    df = df.sort_index()
    logger.info("Sorted index")
    return df


def _read_tpm(path_csv: Path) -> pd.DataFrame:
    """Returns a pd.DataFrame with cells as rows, genes as columns."""
    logger.info("reading TPM from %s", path_csv)
    df = pd.read_csv(path_csv, engine="pyarrow", index_col=0)
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
    mitochondrial_gene_symbols = df_tpm.columns[df_tpm.columns.str.startswith("MT-")]
    logger.info("dropping %s mitochondrial gene symbols", len(mitochondrial_gene_symbols))
    df_tpm = df_tpm.drop(columns=mitochondrial_gene_symbols)
    # identify and remove gene symbols which are zero more than 90% of the time
    zero_genes = df_tpm.columns[df_tpm.eq(0).mean() > 0.9]
    logger.info("dropping %s gene symbols which are zero more than 90%% of the time", len(zero_genes))
    df_tpm = df_tpm.drop(columns=zero_genes)
    return df_tpm


def get_pytorch_dataset() -> RNASeqAnnotatedDataset:
    df_annotations = _read_annotations(ANNOTATIONS_CSV_PATH)
    df_rnaseq = _read_tpm(TPM_CSV_PATH)
    df_rnaseq = _filter_gene_symbols(df_rnaseq)
    return RNASeqAnnotatedDataset(df_rnaseq, df_annotations)


def get_fake_dataset() -> RNASeqAnnotatedDataset:
    n_samples = 1000
    n_genes = 2000
    single_cell_id_index = pd.Index([f"cell_{i}" for i in range(n_samples)], name=columns.SINGLE_CELL_ID)
    gene_symbol_index = pd.Index([f"gene_{i:05d}" for i in range(n_genes)], name=columns.GENE_SYMBOL)
    rng = np.random.default_rng(42)
    rnaseq_tpm = rng.poisson(100, size=(n_samples, n_genes))
    df_rnaseq = pd.DataFrame(rnaseq_tpm, index=single_cell_id_index, columns=gene_symbol_index)
    cell_types = rng.choice(list(nice_to_weirds), size=n_samples)
    df_annotations = pd.DataFrame({columns.CELL_TYPE: cell_types}, index=single_cell_id_index)
    return RNASeqAnnotatedDataset(df_rnaseq, df_annotations)
