from pathlib import Path

import numpy as np
import pandas as pd

from neurotd.app_ephys_gene.core import build_sample_maps, load_sample_maps, save_sample_maps
from neurotd.app_ephys_gene.paths import DATA_PATH, DATA_PROCESSED_PATH

# %% helpers


def log2_transform(df: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
    """
    Log2-transform normalized expression values, adding pseudocount to avoid log2(0).

    Returns
    -------
    log_expr : DataFrame (genes × cells)
    """
    return np.log2(df + pseudocount)


def length_normalize(counts: pd.DataFrame, lengths_bp: pd.Series) -> pd.DataFrame:
    """
    Normalize raw counts by gene length (in base pairs) → counts per kb.

    Parameters
    ----------
    counts : DataFrame (genes × cells)
    gene_lengths : DataFrame indexed by gene, with a column 'length_col' giving gene length in bp
    length_col : which column of gene_lengths to use. Default 'gene_bp'.

    Returns
    -------
    rpk : DataFrame (genes × cells) — reads per kilobase
    """
    lengths_kb = lengths_bp / 1000.0
    # In the dataset, genes with zero-length annotation always have zero raw counts, so divide by 1 is safe.
    lengths_kb = lengths_kb.replace(0, 1)  # Avoid division by zero, lengths_kb = lengths_kb.replace(0, np.nan)
    return counts.div(lengths_kb, axis=0)


def cpm_normalize(rpk: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize RPK (reads per kb) matrix to CPM (counts per million) per cell.

    This controls for sequencing / library depth per cell.

    Returns
    -------
    cpm : DataFrame (genes × cells)
    """
    per_cell_sums = rpk.sum(axis=0)
    scaling = per_cell_sums / 1e6
    return rpk.div(scaling, axis=1)


def normalize_by_length_and_cpm(counts: pd.DataFrame, lengths_bp: pd.Series) -> pd.DataFrame:
    """
    Normalize raw counts by feature length (bp → kb) and library size (per cell).

    Steps:
      1. Divide each gene's counts by its length in kilobases → RPK
      2. For each cell (column), scale RPK so that total counts sum to 1e6 → CPM

    Parameters
    ----------
    counts : DataFrame (genes × cells)
    lengths_bp : Series indexed by gene, giving length in base-pairs

    Returns
    -------
    DataFrame of normalized counts (genes × cells)
    """
    rpk = length_normalize(counts, lengths_bp)
    cpm = cpm_normalize(rpk)
    return cpm


def read_patchseq_counts(
    exon_csv: Path, intron_csv: Path, lengths_csv: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read exon and intron count matrices and gene lengths.

    Returns
    -------
    exon_counts : DataFrame (genes × cells)
    intron_counts : DataFrame (genes × cells)
    gene_lengths : DataFrame with columns including exon_bp, intron_bp, gene_bp (or similar)
    """
    exon_df = pd.read_csv(exon_csv, index_col=0)
    intron_df = pd.read_csv(intron_csv, index_col=0)
    gene_lengths_df = pd.read_csv(lengths_csv, index_col=0)
    return exon_df, intron_df, gene_lengths_df


def preprocess_patchseq_separate(
    exon_df: pd.DataFrame,
    intron_df: pd.DataFrame,
    gene_lengths: pd.DataFrame,
    exon_length_col: str = "exon_bp",
    intron_length_col: str = "intron_bp",
    pseudocount: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full patch-seq normalization pipeline, treating exon and intron counts separately,
    then optionally producing a combined normalized expression matrix.

    Returns
    -------
    expr_exon   : DataFrame (genes × cells) — normalized & log2-transformed exon counts
    expr_intron : DataFrame (genes × cells) — normalized & log2-transformed intron counts
    expr_total  : DataFrame (genes × cells) — combined normalized expression (exon + intron)
    """
    # Exon-only normalization
    lin_exon = normalize_by_length_and_cpm(exon_df, gene_lengths[exon_length_col])
    expr_exon = log2_transform(lin_exon, pseudocount=pseudocount)

    # Intron-only normalization
    lin_intron = normalize_by_length_and_cpm(intron_df, gene_lengths[intron_length_col])
    expr_intron = log2_transform(lin_intron, pseudocount=pseudocount)

    # Combine normalized (linear scale), then re-log-transform
    length_normalized_exon = length_normalize(exon_df, gene_lengths[exon_length_col])
    length_normalized_intron = length_normalize(intron_df, gene_lengths[intron_length_col])
    # counts_combined = exon_count/exon_length_kb + intron_count/intron_length_kb
    combined_lin = cpm_normalize(length_normalized_exon + length_normalized_intron)
    expr_total = log2_transform(combined_lin, pseudocount=pseudocount)

    return expr_exon, expr_intron, expr_total


def qc_zero_length_genes(
    counts_df: pd.DataFrame, lengths_bp: pd.Series, length_type: str = "length_bp"
) -> pd.DataFrame:
    """
    Identify genes where annotated length is zero but raw counts are non-zero.

    Returns
    -------
    DataFrame of genes that violate the assumption.
    Columns: ['GeneID', 'total_raw_counts']
    """
    # genes with zero annotated length
    zero_len = lengths_bp[lengths_bp == 0].index
    # restrict counts to those genes (if present in counts_df)
    sub = counts_df.loc[counts_df.index.intersection(zero_len)]
    # sum raw counts across cells for each gene
    total_counts = sub.sum(axis=1)
    # keep genes with any non-zero count
    bad = total_counts[total_counts > 0]
    if bad.empty:
        print(f"All genes with {length_type}=0 have zero raw counts.")
        return pd.DataFrame(columns=["GeneID", "total_raw_counts"])
    df_bad = bad.reset_index().rename(columns={"index": "GeneID", 0: "total_raw_counts"})
    print("Genes with zero annotated length but non-zero raw counts:")
    print(df_bad.to_string(index=False))
    return df_bad


def filter_zero_genes(expr: pd.DataFrame) -> pd.DataFrame:
    """
    Remove genes (rows) whose sum across all cells is zero (no expression anywhere).

    Parameters
    ----------
    expr : DataFrame (genes × cells)

    Returns
    -------
    filtered : DataFrame with only genes having non-zero total expression.
    """
    mask = (expr.sum(axis=1) != 0).values
    return expr.loc[mask, :]


def filter_protein_coding(
    expr: pd.DataFrame, gene_list: pd.DataFrame, gene_name_col: str = "Gene name"
) -> pd.DataFrame:
    """
    Filter expression matrix to only genes whose name is in the provided protein-coding list.
    gene_list may be a DataFrame (one-column) or a Series / Index-like.
    """
    # If gene_list is a DataFrame with only one column, extract that column
    if isinstance(gene_list, pd.DataFrame) and gene_list.shape[1] == 1:
        names = gene_list.iloc[:, 0].values
    elif isinstance(gene_list, pd.Series) or isinstance(gene_list, (list, tuple, np.ndarray)):
        names = gene_list
    else:
        raise ValueError("gene_list must be DataFrame, Series, or array-like")

    mask = expr.index.isin(names)
    return expr.loc[mask, :]


def clean_and_save(expr: pd.DataFrame, protein_coding_list: pd.Series, out_path: Path) -> None:
    """
    Filter expression matrix to protein-coding genes, remove zero-expressed genes, and save to CSV.
    """
    expr_pc = filter_protein_coding(expr, protein_coding_list)
    expr_filt = filter_zero_genes(expr_pc)
    expr_filt.to_csv(out_path)


def rename_columns_sample_to_cell(expr: pd.DataFrame, sample_to_cell_id: dict) -> pd.DataFrame:
    """
    Rename columns of expr (genes × cells), mapping sample_id → cell_id.
    Columns with sample_id not in map are dropped (or could be kept as NaN).
    """
    new_cols = [sample_to_cell_id.get(sid) for sid in expr.columns]
    # drop samples where mapping is missing (None)
    mask = [c is not None for c in new_cols]
    expr = expr.iloc[:, mask]
    expr.columns = [c for c in new_cols if c is not None]
    return expr


# %% dictionary map from  sample_id to cell_id & nwb filename
raw_ephys_root = DATA_PATH / "raw" / "raw_data_ephys"

missing_sample_id = "20190211_sample_10"  # this sample has gene expression data but no ephys data

sample_to_cell_id, sample_to_nwb = build_sample_maps(raw_ephys_root)

csv_path = DATA_PROCESSED_PATH / "sample_to_cell_id_map.csv"
save_sample_maps(sample_to_cell_id, sample_to_nwb, csv_path)
loaded_sample_to_cell, loaded_sample_to_nwb = load_sample_maps(csv_path)


# %%
###############################################################################
# load data
###############################################################################
# 42466 genes x 1329 cells for both m1_patchseq_exon_counts and m1_patchseq_intron_counts
DATA = DATA_PATH / "raw" / "patchseq-Scala" / "data"
# 2026.01.05_Qiang: better use exon counts, asked KP/GaoYu, GaoYu did alignment, lamp5 is only for Int makes little
# sense.
exon_csv = DATA / "m1_patchseq_exon_counts.csv"
intron_csv = DATA / "m1_patchseq_intron_counts.csv"
lengths_csv = DATA / "gene_lengths.txt"


exon_df, intron_df, gene_lengths_df = read_patchseq_counts(
    exon_csv=exon_csv,
    intron_csv=intron_csv,
    lengths_csv=lengths_csv,
)

# `expr` is a genes × cells matrix, normalized and log-transformed,
# including both mature (exonic) and nascent (intronic) transcripts?
expr_exon, expr_intron, expr_total = preprocess_patchseq_separate(
    exon_df=exon_df,
    intron_df=intron_df,
    gene_lengths=gene_lengths_df,
    exon_length_col="exon_bp",
    intron_length_col="intron_bp",
    pseudocount=1.0,
)


qc_zero_length_genes(exon_df, gene_lengths_df["exon_bp"], "exon_bp")
qc_zero_length_genes(intron_df, gene_lengths_df["intron_bp"], "intron_bp")
# All genes with exon_bp=0 have zero raw counts.
# All genes with intron_bp=0 have zero raw counts.


# %% rename columns from sample_id to cell_id
# assuming expr_exon, expr_intron, expr_total are normalized matrices with sample_id columns
expr_exon = rename_columns_sample_to_cell(expr_exon, sample_to_cell_id)
expr_intron = rename_columns_sample_to_cell(expr_intron, sample_to_cell_id)
expr_total = rename_columns_sample_to_cell(expr_total, sample_to_cell_id)
# expr_exon.to_csv(DATA_PROCESSED_PATH / "gxp_exon_log2.csv")
# %% filter
# 21691 genes
protein_coding_mouse = pd.read_table(DATA_PATH / "raw" / "mouse_39_protein_coding_genes.txt", index_col=0)
for name, expr in [("exon", expr_exon), ("intron", expr_intron), ("total", expr_total)]:
    out_csv = DATA_PROCESSED_PATH / f"gxp_{name}_log2_filtered_pc.csv"
    clean_and_save(expr, protein_coding_mouse, out_csv)

# %%
