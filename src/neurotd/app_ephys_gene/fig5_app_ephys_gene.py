"""
Figure 5. Gene expression correlations with electrophysiological time-delay features in mouse motor cortex

Files required (note: GitHub file size limit is ~100 MB):

time_delays_all_cells.csv
Computed from ephys_data_processing.py and uploaded to GitHub for convenience.

cell_name_to_cell_id_and_sample_id.csv
Computed from ephys_data_processing.py and uploaded to GitHub.

gxp_total_log2_filtered_pc.csv
Computed from gene_expr_data_processing_XH.py, or alternatively uploaded as a compressed file:
gxp_total_log2_filtered_pc.zip (zipped to save space and to comply with the GitHub file size limit of 100 MB).

m1_patchseq_meta_data.csv
Provided by the original paper and uploaded to GitHub.
"""

import datetime

# %%
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from neurotd.app_ephys_gene.core import (
    get_cell_ids_by_rna_family,
    load_cell_name_to_cell_id,
    load_sample_maps,
    parse_vector_columns,
)
from neurotd.app_ephys_gene.paths import (
    CVS_MAP_PATH,
    DATA_PATH,
    DATA_PROCESSED_PATH,
    FIGURE_PATH,
    META_FILE_PATH,
    RESULT_PATH,
)
from neurotd.app_ephys_gene.plotting import plot_all_sweeps_for_cell

plt.ion()  # enable interactive mode for command-line script execution (non-blocking figure display)

SHOW_EXTRA_PLOTS = False  # set to True to see extra plots for debugging

# %% functions
def truncated_mean(data, percentage=10):
    """
    Calculate the truncated mean of a 1D numpy array by excluding the smallest and largest given percentage of values.

    Parameters:
    - data (np.array): The input 1D numpy array.
    - percentage (int): The percentage of data points to exclude from each end of the sorted array.

    Returns:
    - float: The truncated mean of the array.
    """
    data = np.asarray(data)
    sorted_data = np.sort(data)
    n = len(sorted_data)
    exclusion_count = int(np.ceil(percentage / 100.0 * n))

    if n - exclusion_count * 2 <= 0:
        return np.mean(sorted_data)  # original mean
    else:
        trimmed_data = sorted_data[exclusion_count:-exclusion_count]
        return np.mean(trimmed_data)


def compute_sweep_coefs(td_index, tdvs, cell_ids, degree=1, normalized=False, plot=False):
    """
    Fit a polynomial of given degree to each sweep (time-delay vector) and return an array of coefficients for each row
    in td_index.

    This function optionally normalizes each time-delay vector by its first valid inter-spike interval before
    polynomial fitting. When normalized=True, the within-sweep adaptation slope reflects relative change from the
    first interval.

    Parameters
    ----------
    td_index : array-like
        Sequence of sweep identifiers corresponding to rows in tdvs.
    tdvs : sequence of array-like
        Collection of time-delay vectors, one per sweep.
    cell_ids : array-like
        Identifier of the cell for each sweep, same length as td_index.
    degree : int
        Polynomial degree to fit to each sweep vector (default 1).
    normalized : bool
        If True, normalize each time-delay vector by its first non-NaN value before
        fitting (default False).
    plot : bool
        If True, show a per-cell plot of fitted polynomials overlaying data
        (default False).

    Returns
    -------
    coefs : np.ndarray, shape (len(td_index), degree+1)
        Polynomial coefficients for each sweep. Each row corresponds to the fit
        for td_index[i]. Rows with insufficient data or failed fits contain NaN.

    Notes
    -----
    Normalization is computed as:

        y_norm = y / y[0]

    where y[0] is the first element of the time-delay vector after casting to float. If y[0] is NaN or zero,
    normalization proceeds only if a valid baseline can be determined; otherwise the fit is skipped.

    The polynomial is fitted against x = 0,1,...,len(y)-1.

    Plots show raw or normalized points with fitted polynomial curves.
    """
    td_index = np.asarray(td_index)
    tdvs = np.asarray(tdvs, dtype=object)
    cell_ids = np.asarray(cell_ids)

    n = len(td_index)
    coefs = np.full((n, degree + 1), np.nan, dtype=float)

    for cell_id in np.unique(cell_ids):
        mask = cell_ids == cell_id
        if plot:
            plt.figure()
        for idx, tdv in zip(td_index[mask], tdvs[mask]):
            i = np.where(td_index == idx)[0][0]
            y = np.asarray(tdv, dtype=float)
            # skip if insufficient points or all NaN
            if y.size < 2 or np.isnan(y).all():
                continue

            if normalized:
                # determine baseline (first non-NaN)
                valid = ~np.isnan(y)
                if not np.any(valid):
                    continue
                first_idx = np.argmax(valid)
                baseline = y[first_idx]
                # skip if baseline is zero (prevents division by zero)
                if baseline == 0:
                    continue
                # normalize relative to baseline
                y = y / baseline

            x = np.arange(len(y))
            try:
                coef = np.polyfit(x, y, degree)
                coefs[i, :] = coef
            except Exception:
                coefs[i, :] = np.nan
                continue

            if plot:
                poly = np.poly1d(coef)
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = poly(x_fit)
                plt.scatter(x, y, color="blue", s=20, alpha=0.7)
                plt.plot(x_fit, y_fit, color="red", lw=1.5)
        if plot:
            plt.xlabel("Spike index within sweep")
            ylabel = "Normalized delay" if normalized else "Raw delay"
            plt.ylabel(ylabel)
            plt.title(f"Polynomial fits (degree={degree}, normalized={normalized}) — cell {cell_id}")
            plt.show()

    return coefs


def compute_istd_from_coefs(coefs, cell_ids, percentage=20):
    """
    Compute ISTD (ft_scalars) per unique cell by taking the truncated mean of the per-sweep slope (first coefficient)
    across sweeps belonging to each cell.

    Parameters:
    - coefs: array-like, shape (n_sweeps, degree+1) each row is polynomial coeffs
    - cell_ids: array-like, length n_sweeps, cell identifier for each sweep
    - percentage: int, percent to trim from each end when computing truncated mean

    Returns:
    - ft_scalars: np.ndarray, shape (n_unique_cells,), truncated-mean slope per cell
    - unique_cells: np.ndarray of unique cell ids in the returned order
    """
    coefs = np.asarray(coefs, dtype=object)
    cell_ids = np.asarray(cell_ids)
    unique_cells = np.unique(cell_ids)
    ft_scalars = np.full((len(unique_cells),), np.nan, dtype=float)

    for i, cell_id in enumerate(unique_cells):
        mask = cell_ids == cell_id
        # take the slope (first element) from each sweep's coef array
        col1 = np.array([arr[0] for arr in coefs[mask]], dtype=float)
        # drop NaNs before computing truncated mean
        col1 = col1[~np.isnan(col1)]
        if col1.size == 0:
            ft_scalars[i] = np.nan
        else:
            ft_scalars[i] = truncated_mean(col1, percentage=percentage)

    return ft_scalars, unique_cells


# refactor to function with inputs of gxp_std and ft_scalars_std
def compute_spearman_per_gene(gxp_std, ft_scalars_std, min_cells=10):
    corr_scores = np.zeros((gxp_std.shape[0],))
    pvals = np.zeros((gxp_std.shape[0],))
    for i, gene in enumerate(gxp_std.index.values):
        # masking out those cells that have 0 expression for the gene
        mask = gxp.loc[gene].values != 0
        # filtering out those genes with less than min_cells expressing the gene
        if sum(mask) < min_cells:
            continue
        # spearmanr. using standardized gxp and ft_scalars_std
        res = spearmanr(
            gxp_std.loc[gene].values[mask],
            ft_scalars_std[mask],
        )
        corr_scores[i] = res[0]
        pvals[i] = res[1]
    return corr_scores, pvals


def compute_sweep_coefs_per_cell(
    td_pd: pd.DataFrame, degree: int = 1, normalized: bool = True, plot: bool = False
) -> pd.DataFrame:
    """
        Fit a polynomial of given degree to all (normalized) intra-sweep delays
        across all sweeps of each cell, returning one set of coefficients per cell.

        Normalization is by each sweep’s first valid inter-spike interval,
        i.e., for sweep i and spike index j:
            y_norm_ij = d_ij / d_i0

        Parameters
        ----------
        td_pd : pd.DataFrame
            DataFrame with parsed 'time_delays(ms)' for each sweep and a 'cell_id'
            identifying the cell of each sweep. Each element of
            'time_delays(ms)' should be a numeric list/array of delays.
        degree : int
            Degree of polynomial to fit (default 1).
        normalized : bool
            If True (default), fit using normalized delays
            y_norm = d_ij / d_i0. If False, fit raw delays.
        plot : bool
            If True, generate a plot per cell showing the data points and fitted
            polynomial curve.

        Returns
        -------
        coef_df : pd.DataFrame
            DataFrame with one row per cell containing:
                'cell_id' : identifier of the cell
                'coefs'   : numpy array of polynomial coefficients (degree+1 length)
                'n_points': total number of (j,i) points used for fit
        Usage
        -------
    # future explore multiple degrees, though seems the current degree = 1 is most interpretable
    # cell_coefs_df = compute_sweep_coefs_per_cell(td_pd, degree=1, normalized=True)
    # coefs_slope_intercept = np.vstack(cell_coefs_df["coefs"].values)
    # ft_scalars = coefs_slope_intercept[:, 0]  # use slope only
    """
    results = []

    # iterate through unique cells
    for cell in td_pd["cell_id"].unique():
        sub = td_pd[td_pd["cell_id"] == cell]

        # collect all normalized x,y pairs across sweeps
        X = []  # independent variable (spike index within sweep)
        Y = []  # dependent variable (normalized or raw delay)
        for _, row in sub.iterrows():
            delays = np.asarray(row["time_delays(ms)"], dtype=float)
            # skip empty or all-NaN
            if delays.size < 1 or np.isnan(delays).all():
                continue

            # determine baseline for normalization if needed
            if normalized:
                valid = ~np.isnan(delays)
                if not np.any(valid):
                    continue
                first_valid_idx = np.argmax(valid)
                baseline = delays[first_valid_idx]
                # skip if baseline is zero or NaN
                if baseline == 0 or np.isnan(baseline):
                    continue

            # iterate through all indices and collect
            for j, d in enumerate(delays):
                if np.isnan(d):
                    continue
                if normalized:
                    y_val = d / baseline
                else:
                    y_val = d
                X.append(j)
                Y.append(y_val)

        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        # if too few points to fit
        if Y.size < (degree + 1):
            coefs = np.full(degree + 1, np.nan, dtype=float)
            n_pts = Y.size
        else:
            # fit polynomial
            coefs = np.polyfit(X, Y, degree)
            n_pts = Y.size

            if plot:
                poly = np.poly1d(coefs)
                x_fit = np.linspace(X.min(), X.max(), 200)
                y_fit = poly(x_fit)

                plt.figure()
                plt.scatter(X, Y, color="blue", s=10, alpha=0.6)
                plt.plot(x_fit, y_fit, color="red", lw=1.5)
                title = f"cell {cell} poly(degree={degree}) fit, " f"normalized={normalized}, n_points={n_pts}"
                plt.title(title)
                plt.xlabel("Within-sweep spike index")
                ylabel = "Normalized delay" if normalized else "Raw delay"
                plt.ylabel(ylabel)
                plt.show()

        results.append({"cell_id": cell, "coefs": coefs, "n_points": n_pts})

    coef_df = pd.DataFrame(results)
    return coef_df


def plot_topN_corr_genes(corr_scores, gxp_index, N=20):
    """ show top N = 20 genes as bar chart """
    ranking = np.argsort(np.abs(corr_scores))[::-1]

    # plot top N genes
    indices = ranking[:N]
    x = gxp_index[indices]
    y = corr_scores[indices]
    # Define colors based on positive or negative values of y
    colors = ['#155ed4' if value < 0 else '#9e2134' for value in y]

    plt.figure(figsize=(10, 6))
    plt.bar(x, y, color=colors)
    # plt.ylim(-0.45, 0.55)
    plt.xticks(rotation=60, fontsize=16)
    plt.yticks(ticks=[-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=17)
    plt.ylabel("Expression correlation \nwith ISTD score", fontsize=17)
    plt.xlabel("Gene names", fontsize=13)


    legend_elements = [
        Patch(facecolor='#9e2134', edgecolor='#9e2134', label='Positive correlation'),
        Patch(facecolor='#155ed4', edgecolor='#155ed4', label='Negative correlation'),
    ]
    plt.legend(handles=legend_elements, fontsize=14)
    plt.subplots_adjust(bottom=0.3, left=0.15)
    plt.savefig(f"top_{N}_correlated_genes.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.show()


# %% specify files and metadata
time_delay_file = DATA_PROCESSED_PATH / "time_delays_all_cells.csv"
cell_name_to_cell_id = load_cell_name_to_cell_id(CVS_MAP_PATH)  # cell_name_to_cell_id_and_sample_id.csv
sample_id_to_cell_id, sample_to_nwb = load_sample_maps(CVS_MAP_PATH)

try:
    meta = pd.read_csv(META_FILE_PATH, sep="\t")  # m1_patchseq_meta_data.csv
except (FileNotFoundError, OSError):
    meta = pd.read_csv(DATA_PROCESSED_PATH / "m1_patchseq_meta_data.csv", sep="\t")
meta["RNA family"].value_counts()
# meta_give_cell_type = meta.loc[meta["RNA family"] == cell_type]

USE_ALL_CELLS = True  # if True, use all 1000+ cells; if False, use 51 cells for debugging
if USE_ALL_CELLS:
    time_delay_file = DATA_PROCESSED_PATH / "time_delays_all_cells.csv"
    gxp_file = DATA_PROCESSED_PATH / "gxp_total_log2_filtered_pc.csv"  # genes x cells  use total counts
else:
    time_delay_file = DATA_PROCESSED_PATH / "time_delays_51cells.csv"
    gxp_file = DATA_PROCESSED_PATH / "gxp_log2_51cells.csv"  # genes x cells

# toggle saving of ISTD scores table
SAVE_ISTD = False  # False to avoid overwriting existing file
istd_file = RESULT_PATH / "ISTD_scores.csv"

# %%
TOL = 1e-6  # numeric tolerance for exact-match checks (can relax to 1e-3 if needed)
SAVE_RECOMPUTED = True

# 1) load inputs
td = pd.read_csv(time_delay_file, index_col=0)  # 10639 rows: sweeps, columns: 'peak_idx', 'time_delays(ms)'
td["cell_id"] = td.index.str.rsplit("-", n=1).str[0]
gxp = pd.read_csv(gxp_file, index_col=0)  # genes x cells = 18646 x 1328 cells

if not USE_ALL_CELLS:
    # fix the gxp which was computed as gxp = log2(raw + 1)
    gxp_raw = np.exp2(gxp) - 1
    # check if gxp_raw are all integers (original counts)
    assert np.all(np.abs(gxp_raw - np.round(gxp_raw)) < TOL), "gxp_raw contains non-integer values"
    gxp_raw = np.round(gxp_raw).astype(int)
    # normalize sequencing depth back to counts per 10,000, so that each cell sums to 10,000
    gxp_raw_norm = gxp_raw * 10000 / gxp_raw.sum(axis=0)
    gxp = np.log2(gxp_raw_norm + 1)

#  Analysis for a given cell type only, use the largest excitatory cell type "IT" (ExN), avoid mixing cell types
filtering_by_cell_type = True
if filtering_by_cell_type:
    cell_type = (
        "IT"  # "IT" (ExN) = 254 -> 237, "Pvalb" (InN) = 289 -> 286 which is inhibitory also works but less interesting
    )
    cell_ids = get_cell_ids_by_rna_family(cell_type, sample_id_to_cell_id, meta_file=META_FILE_PATH)
    print(f"Use only {cell_type} cell_type with {len(cell_ids)} cells")
    td = td.loc[td["cell_id"].isin(cell_ids)]
    gxp = gxp[gxp.columns.intersection(cell_ids)]

cell_ids_final_sorted = sorted(td["cell_id"].unique().tolist())
N_cells = len(cell_ids_final_sorted)  # 51 unique cells  # 237 for IT cells
print(f"Number of unique cells in td: {N_cells}")

# only keep gxp columns that are in np.unique(cell_ids)
if USE_ALL_CELLS:
    overlap_cells = gxp.columns.intersection(cell_ids_final_sorted)  # 1298
    if len(overlap_cells) == 0:
        raise RuntimeError("No overlapping cells between td and gxp.")
    gxp = gxp.loc[:, overlap_cells]
    print(f"Using all cells: gxp shape after aligning with td: {gxp.shape}")

# make sure sorted for alignment
# td = td.loc[cell_ids_final_sorted]  # td already sorted
gxp = gxp.loc[:, cell_ids_final_sorted]


# Follow common practice in single-cell analysis to remove inhibitory andglial markers when defining excitatory neuron
# expression profiles, list of inhibitory/glial marker genes to remove:
# # major interneuron markers, GABA synthesis, oligodendrocyte markers, microglial markers, add others as needed
inhibitory_markers = ["Lamp5", "Pvalb", "Sst", "Vip", "Gad1", "Gad2", "Mbp", "Plp1", "Mag", "C1qa", "C1qb", "Cx3cr1"]
marker_mask = ~gxp.index.isin(inhibitory_markers)
print(f"After removing inhibitory/glial markers: {gxp.shape[0]} -> {marker_mask.sum()}")  # 18668 -> 18656


threshold_zero = 0.0  # expression threshold in log2p1 scale
threshold_percent = 0.50  # require gene expressed in at least threshold_percent of cells, 50% by default
gxp_expression_level = (gxp > threshold_zero).sum(axis=1) / gxp.shape[1]
gene_mask = gxp_expression_level >= threshold_percent
print(f"Filtering genes: {marker_mask.sum()} -> {gene_mask.sum()} (expressed in >={threshold_percent*100}% cells)")
gene_mask = gene_mask & marker_mask  # final mask used later
# 51 cells: Filtering genes: 14783 -> 4333 (expressed in >=50% cells)
# full: Filtering genes: 18656 -> 7993 ( (expressed in >=50% cells)


td_pd = parse_vector_columns(td, vec_cols=["peak_idx", "time_delays(ms)"])
tdvs = td_pd["time_delays(ms)"].values

# compute sweep-wise polynomial coefficients
coefs = compute_sweep_coefs(td.index, tdvs, td["cell_id"], degree=1, normalized=False, plot=False)
# compute ISTD scores from coefs
ft_scalars, ft_cells = compute_istd_from_coefs(coefs, td["cell_id"], percentage=20)


istd_std = ft_scalars_std = (ft_scalars - ft_scalars.mean()) / ft_scalars.std()
istd_z = pd.Series(istd_std, index=cell_ids_final_sorted, name="ISTD")

#  standardize expression per gene (row-wise)
gxp_std = gxp.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
corr_scores, pvals = compute_spearman_per_gene(gxp_std, ft_scalars_std, min_cells=10)
corr_scores_df = pd.DataFrame(corr_scores, index=gxp.index, columns=['corr_score'])

# %% sort corr_scores by absolute value
# only consider gene_masked genes
corr_scores_df_expressed = corr_scores_df.loc[gene_mask, :]
gxp_expressed = gxp.loc[gene_mask, :]

corr_scores_abs_sorted = corr_scores_df_expressed.reindex(
    corr_scores_df_expressed['corr_score'].abs().sort_values(ascending=False).index
)


plot_topN_corr_genes(corr_scores_df_expressed['corr_score'].values, gxp_expressed.index, N=20)


###############################################################################
# Figure: Enrichment of top 200 correlated genes
###############################################################################
# datastr = datetime.datetime.now().strftime("%Y%m%d")
# filename = f"genes_top200_{datastr}"
filename = "genes_top200"
top200 = corr_scores_abs_sorted.iloc[:200]

# Save CSV (full table)
csv_file = RESULT_PATH / f"{filename}.csv"
top200.to_csv(csv_file)

# Save TXT (gene names only, one per line)
txt_file = RESULT_PATH / f"{filename}.txt"
top200.index.to_series().to_csv(txt_file, index=False, header=False)
# use metascape express analysis to get the enriched terms from the gene list txt file
# https://metascape.org/gp/index.html#/main/step1
# Step 2: dataset is from the Mouse Motor Cortex.  choose M. musculus (199)" for Both Input and Analysis
# Step 3: Choose "Express Analysis"


# %% Extra plots for debugging-----------------------------------------
if SHOW_EXTRA_PLOTS:
    plt.plot(corr_scores_abs_sorted['corr_score'].abs().values)
    plt.xlabel("Genes sorted by |Spearman ρ|")
    plt.ylabel("|Spearman ρ|")
    plt.title("Spearman correlation scores (absolute values) between gene expression and ISTD")

    # for top N= 20 genes, show the expression level in the sorted order as a secondary y-axis
    N = 20
    plt.axvline(x=N, color='red', linestyle='--', alpha=0.7)
    plt.plot(corr_scores_abs_sorted['corr_score'].abs().values[:N], color='blue')
    plt.xlabel("Genes sorted by |Spearman ρ|")
    plt.ylabel("|Spearman ρ|", color='blue')
    ax2 = plt.gca().twinx()
    expression_levels_sorted = gxp_expression_level.reindex(corr_scores_abs_sorted.index)
    ax2.plot(expression_levels_sorted.values[:N], color='orange', alpha=0.5)
    ax2.set_ylabel("Expression level (fraction of cells expressed)", color='orange')
    plt.title(f"Top {N} genes by |Spearman ρ| with expression levels")
    plt.show()

    # % draw fit line
    # make sure cell order matches
    cells = gxp_std.columns
    x = pd.Series(ft_scalars_std, index=cells, name="ISTD")

    best_gene = corr_scores_abs_sorted.index[0]  # 'Cdca4' -> 'Fut9' for 51 cells
    y = gxp_std.loc[best_gene, cells]

    # compute Spearman correlation
    rho, p = spearmanr(x, y, nan_policy="omit")
    print(f"{best_gene}: Spearman ρ = {rho:.3f}, p = {p:.2e}, n = {len(x)}")

    # scatter + regression line
    plt.figure(figsize=(7, 6))
    sns.regplot(x=x, y=y, scatter_kws=dict(s=25, alpha=0.8), line_kws=dict(color="black", lw=2))
    plt.xlabel("ISTD score (ft_scalars_std)", fontsize=14)
    plt.ylabel(f"{best_gene} expression (z-scored log2)", fontsize=14)
    plt.title(f"{best_gene} vs ISTD  (ρ={rho:.2f}, p={p:.1e})", fontsize=13)
    plt.tight_layout()
    # plt.savefig(FIGURE_PATH + f"scatter_{best_gene}_expr_vs_ISTD.png", dpi=300)
    plt.show()

plt.show(block=True)  # keep the figures open for scripts run outside of an interactive environment

# %%
