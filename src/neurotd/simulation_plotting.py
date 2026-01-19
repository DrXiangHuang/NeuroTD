import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import lines

from neurotd.time_varying_shift import sample_to_time_ms

# %%
# Define colors based on the provided plotting code
colors = {
    "Ground Truth": "black",
    "NeuroTD": "#D55E00",  # Dark Orange, NeuroTD default is quality
    "NeuroTD_quality": "#D55E00",  # Dark Orange
    "NeuroTD_oracle": "#000000",  # Black
    "DTW": "#CC79A7",  # Purple
    "CTW": "#009E73",  # Green
    "FFT": "#0072B2",  # Blue
}

markers = {
    "Ground Truth": "o",
    "NeuroTD": "s",  # Square, NeuroTD default is quality
    "NeuroTD_quality": "s",  # Square
    "NeuroTD_oracle": "*",  # Star
    "DTW": "D",  # Diamond
    "CTW": "^",  # Triangle Up
    "FFT": "v",  # Triangle Down
}


def plot_rmse_vs_noise_oracle_vs_quality(
    df,
    markers=markers,
    colors=colors,
    fs=5120,
    fig_name="time_varying_shift_noise_all_levels",
    figsize=(4, 6),
    linewidth=2,
    marker_size=10,
    NeuroTD_with_oracle=True,  # True: continuous, False: discrete as it has RMSE=0.0 for noise=0.0 in oracle  bad for log scale
):
    """
    Plot RMSE versus noise level under two evaluation regimes:
    (i) oracle window-size selection with ground-truth delay access (all methods);
    (ii) quality-based window-size selection without ground-truth access (NeuroTD only).
    df is pandas DataFrame with columns:
        ['noise_std', 'window_size', 'hop_size', 'rmse_neurotd', 'quality_rmse_neurotd',
        'quality_rmse_per_window_neurotd', 'rmse_ctw', 'rmse_dtw', 'rmse_fft']
        Note: the rmse_* columns are in sample unit instead of ms unit.
    returns the matplotlib Figure object.

    """
    # Oracle selection: minimum RMSE across window sizes (ground-truth access)
    rmse_cols = [c for c in df.columns if c.startswith("rmse_")]
    min_rmse_per_noise = df.groupby("noise_std")[rmse_cols].min().sort_index()

    # NeuroTD quality-based selection: no ground-truth access
    best_by_quality = df.loc[df.groupby("noise_std")["quality_rmse_neurotd"].idxmin()].sort_values("noise_std")

    method_style = {
        "rmse_neurotd": dict(
            label="NeuroTD (oracle window)",
            linestyle=":",
            marker=markers["NeuroTD_oracle"],
            color=colors["NeuroTD_oracle"],
        ),
        "rmse_dtw": dict(
            label="DTW",
            linestyle=":",
            marker=markers["DTW"],
            color=colors["DTW"],
        ),
        "rmse_ctw": dict(
            label="CTW",
            linestyle=":",
            marker=markers["CTW"],
            color=colors["CTW"],
        ),
        "rmse_fft": dict(
            label="FFT",
            linestyle=":",
            marker=markers["FFT"],
            color=colors["FFT"],
        ),
    }

    neurotd_quality_style = dict(
        label="NeuroTD (quality-selected window)" if NeuroTD_with_oracle else "NeuroTD",
        linestyle="--",
        marker=markers["NeuroTD_quality"],
        color=colors["NeuroTD_quality"],
    )

    fig = plt.figure(figsize=figsize)

    # NeuroTD without oracle tuning
    plt.plot(
        best_by_quality["noise_std"],
        sample_to_time_ms(best_by_quality["rmse_neurotd"], fs),
        linewidth=linewidth,
        markersize=marker_size,
        **neurotd_quality_style,
    )

    # Oracle-selected curves for all methods
    for col, style in method_style.items():
        if not NeuroTD_with_oracle and col == "rmse_neurotd":
            continue  # skip NeuroTD oracle curve in discrete case
        plt.plot(
            min_rmse_per_noise.index,
            sample_to_time_ms(min_rmse_per_noise[col], fs),
            linewidth=linewidth,
            markersize=marker_size,
            **style,
        )

    plt.xlabel("Noise Std")
    plt.ylabel("RMSE (ms)")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    fig.savefig(fig_name + ".png", dpi=300, bbox_inches="tight")


def format_noise_sig3(x: float) -> str:
    """Format noise_std with ~2 significant digits while keeping 0.000 exact."""
    if abs(x) < 1e-12:
        return "0.000"
    return f"{x:.3g}"


def generate_table_full_latex(
    csv_path: str,
    sampling_rate_hz: float = 5120.0,
    noise_col: str = "noise_std",
    window_col: str = "window_size",
    quality_col: str = "quality_rmse_neurotd",
    rmse_neurotd_col: str = "rmse_neurotd",
    rmse_dtw_col: str = "rmse_dtw",
    rmse_ctw_col: str = "rmse_ctw",
    rmse_fft_col: str = "rmse_fft",
    sort_windows_desc: bool = False,
    quality_decimals: int = 3,
    rmse_decimals: int = 3,
) -> str:
    """
    Generate LaTeX source for a full supplementary benchmarking table.

    Formatting rules (per noise level):
    - Bold: NeuroTD row with minimum quality score.
    - Underline: minimum RMSE across window sizes for each method.
    - Bold + underline: NeuroTD row that is both quality-selected and oracle RMSE.
    - DTW / CTW / FFT: underline oracle RMSE only (no bold).

    Notes:
    - All RMSE values are converted from samples to milliseconds.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    required = {
        noise_col,
        window_col,
        quality_col,
        rmse_neurotd_col,
        rmse_dtw_col,
        rmse_ctw_col,
        rmse_fft_col,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Window size in ms
    df["window_ms"] = df[window_col].apply(lambda x: sample_to_time_ms(x, sampling_rate_hz))

    # RMSEs in ms
    for col in (
        rmse_neurotd_col,
        rmse_dtw_col,
        rmse_ctw_col,
        rmse_fft_col,
    ):
        df[col + "_ms"] = df[col].apply(lambda x: sample_to_time_ms(x, sampling_rate_hz))

    # Stable ordering
    df = df.sort_values(
        by=[noise_col, window_col],
        ascending=[True, not sort_windows_desc],
        kind="mergesort",
    )

    # Indices for formatting
    idx_best_quality = df.groupby(noise_col)[quality_col].idxmin()

    idx_min_rmse = {
        "neurotd": df.groupby(noise_col)[rmse_neurotd_col + "_ms"].idxmin(),
        "dtw": df.groupby(noise_col)[rmse_dtw_col + "_ms"].idxmin(),
        "ctw": df.groupby(noise_col)[rmse_ctw_col + "_ms"].idxmin(),
        "fft": df.groupby(noise_col)[rmse_fft_col + "_ms"].idxmin(),
    }

    best_quality_set = set(idx_best_quality.tolist())
    min_rmse_sets = {k: set(v.tolist()) for k, v in idx_min_rmse.items()}

    def fmt_quality(v: float) -> str:
        return f"{v:.{quality_decimals}f}"

    def fmt_rmse(v: float) -> str:
        return f"{v:.{rmse_decimals}f}"

    lines: list[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{c c c c c c c}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Noise} & "
        r"\textbf{Window (samples, ms)} & "
        r"\textbf{Quality} & "
        r"\textbf{NeuroTD} & "
        r"\textbf{DTW} & "
        r"\textbf{CTW} & "
        r"\textbf{FFT} \\"
    )
    lines.append(r"\hline")

    for noise, dfn in df.groupby(noise_col, sort=True):
        for idx, row in dfn.iterrows():
            noise_str = format_noise_sig3(float(row[noise_col]))

            w_samp = int(row[window_col])
            w_ms = float(row["window_ms"])

            # Quality
            q_str = fmt_quality(float(row[quality_col]))
            is_best_q = idx in best_quality_set
            if is_best_q:
                q_str = rf"\textbf{{{q_str}}}"

            # NeuroTD RMSE
            r_neuro = fmt_rmse(float(row[rmse_neurotd_col + "_ms"]))
            is_neuro_oracle = idx in min_rmse_sets["neurotd"]
            if is_neuro_oracle:
                r_neuro = rf"\underline{{{r_neuro}}}"
            if is_best_q:
                if r_neuro.startswith(r"\underline{"):
                    r_neuro = r_neuro.replace(r"\underline{", r"\textbf{\underline{", 1) + "}"
                else:
                    r_neuro = rf"\textbf{{{r_neuro}}}"

            # DTW RMSE
            r_dtw = fmt_rmse(float(row[rmse_dtw_col + "_ms"]))
            if idx in min_rmse_sets["dtw"]:
                r_dtw = rf"\underline{{{r_dtw}}}"

            # CTW RMSE
            r_ctw = fmt_rmse(float(row[rmse_ctw_col + "_ms"]))
            if idx in min_rmse_sets["ctw"]:
                r_ctw = rf"\underline{{{r_ctw}}}"

            # FFT RMSE
            r_fft = fmt_rmse(float(row[rmse_fft_col + "_ms"]))
            if idx in min_rmse_sets["fft"]:
                r_fft = rf"\underline{{{r_fft}}}"

            # Window string
            if is_best_q:
                window_str = rf"\textbf{{{w_samp} ({w_ms:.1f})}}"
            else:
                window_str = f"{w_samp} ({w_ms:.1f})"

            lines.append(f"{noise_str} & {window_str} & {q_str} & " f"{r_neuro} & {r_dtw} & {r_ctw} & {r_fft} \\\\")

        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_table_s1_latex(
    csv_path: str,
    sampling_rate_hz: float = 5120.0,
    quality_col: str = "quality_rmse_neurotd",
    rmse_col: str = "rmse_neurotd",
    noise_col: str = "noise_std",
    window_col: str = "window_size",
    sort_windows_desc: bool = False,
    quality_decimals: int = 3,
    rmse_decimals: int = 3,
) -> str:
    """
    Generate LaTeX source for Supplementary Table S1 from NeuroTD simulation CSV.

    Formatting rules:
    - Per noise level: bold the row with minimum quality score (quality + window + RMSE).
    - Per noise level: underline the minimum RMSE across all windows.

    Notes:
    - RMSE values are converted from samples to milliseconds before reporting.
    """
    df = pd.read_csv(csv_path)

    required = {noise_col, window_col, quality_col, rmse_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Window size in ms
    df["window_ms"] = df[window_col].apply(lambda x: sample_to_time_ms(x, sampling_rate_hz))

    # RMSE in ms (converted from samples)
    df["rmse_ms"] = df[rmse_col].apply(lambda x: sample_to_time_ms(x, sampling_rate_hz))

    # Stable ordering
    df = df.sort_values(
        by=[noise_col, window_col],
        ascending=[True, not sort_windows_desc],
        kind="mergesort",
    )

    # Identify best rows per noise (after conversion)
    idx_best_quality = df.groupby(noise_col)[quality_col].idxmin()
    idx_best_rmse = df.groupby(noise_col)["rmse_ms"].idxmin()
    best_quality_set = set(idx_best_quality.tolist())
    best_rmse_set = set(idx_best_rmse.tolist())

    def fmt_quality(v: float) -> str:
        return f"{v:.{quality_decimals}f}"

    def fmt_rmse(v: float) -> str:
        return f"{v:.{rmse_decimals}f}"

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{c c c c}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Noise std} & "
        r"\textbf{Window size: samples (ms)} & "
        r"\textbf{Quality score} & "
        r"\textbf{RMSE (ms)} \\"
    )
    lines.append(r"\hline")

    # Body
    for noise, dfn in df.groupby(noise_col, sort=True):
        for idx, row in dfn.iterrows():
            noise_str = format_noise_sig3(float(row[noise_col]))

            w_samp = int(row[window_col])
            w_ms = float(row["window_ms"])

            q_str = fmt_quality(float(row[quality_col]))
            r_str = fmt_rmse(float(row["rmse_ms"]))

            is_best_q = idx in best_quality_set
            is_best_r = idx in best_rmse_set

            if is_best_q:
                q_str = rf"\textbf{{{q_str}}}"
            if is_best_r:
                r_str = rf"\underline{{{r_str}}}"

            if is_best_q:
                window_str = rf"\textbf{{{w_samp} ({w_ms:.1f})}}"
                if not r_str.startswith(r"\underline{"):
                    r_str = rf"\textbf{{{r_str}}}"
                else:
                    r_str = r_str.replace(r"\underline{", r"\textbf{\underline{", 1) + "}"
            else:
                window_str = f"{w_samp} ({w_ms:.1f})"

            lines.append(f"{noise_str} & {window_str} & {q_str} & {r_str} \\\\")
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_table_s2_latex(
    csv_path: str,
    sampling_rate_hz: float = 5120.0,
    noise_col: str = "noise_std",
    window_col: str = "window_size",
    rmse_neurotd_col: str = "rmse_neurotd",
    quality_col: str = "quality_rmse_neurotd",
    rmse_cols_baseline: tuple[str, ...] = ("rmse_dtw", "rmse_ctw", "rmse_fft"),
    rmse_decimals: int = 2,
) -> str:
    """
    Generate LaTeX source for Supplementary Table S2.

    Definitions (all RMSE reported in ms):
    - NeuroTD (quality-selected): RMSE at the window selected by the quality criterion.
    - NeuroTD (oracle): minimum NeuroTD RMSE across all evaluated window sizes.
    - DTW / CTW / FFT-based: oracle minimum RMSE across window sizes.
    """
    df = pd.read_csv(csv_path)

    required = {noise_col, window_col, rmse_neurotd_col, quality_col, *rmse_cols_baseline}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert all RMSE columns from samples to ms
    rmse_all_cols = (rmse_neurotd_col,) + rmse_cols_baseline
    for col in rmse_all_cols:
        df[col + "_ms"] = df[col].apply(lambda x: sample_to_time_ms(x, sampling_rate_hz))

    # Baselines: oracle min RMSE across window sizes (ms)
    min_rmse_baseline = (
        df.groupby(noise_col, sort=True)[[c + "_ms" for c in rmse_cols_baseline]]
        .min()
        .rename(
            columns={
                "rmse_dtw_ms": "rmse_dtw",
                "rmse_ctw_ms": "rmse_ctw",
                "rmse_fft_ms": "rmse_fft",
            }
        )
    )

    # NeuroTD oracle: min RMSE across window sizes (ms)
    neurotd_oracle = df.groupby(noise_col, sort=True)["rmse_neurotd_ms"].min().rename("rmse_neurotd_oracle")

    # NeuroTD quality-selected RMSE (ms)
    idx_best_quality = df.groupby(noise_col, sort=True)[quality_col].idxmin()
    neurotd_quality = (
        df.loc[idx_best_quality].set_index(noise_col)["rmse_neurotd_ms"].rename("rmse_neurotd_quality").sort_index()
    )

    # Merge all results
    out = pd.concat([neurotd_quality, neurotd_oracle, min_rmse_baseline], axis=1)

    def fmt_rmse(v: float) -> str:
        return f"{v:.{rmse_decimals}f}"

    # Build LaTeX
    lines: list[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"    \centering")
    lines.append(r"    \begin{tabular}{c|ccccc}")
    lines.append(r"        \hline")
    lines.append(
        r"        \textbf{Noise std} & "
        r"\textbf{NeuroTD (quality-selected)} & "
        r"\textbf{NeuroTD (oracle)} & "
        r"\textbf{DTW} & "
        r"\textbf{CTW} & "
        r"\textbf{FFT-based} \\"
    )
    lines.append(r"        \hline")

    for noise, row in out.iterrows():
        noise_str = format_noise_sig3(float(noise))
        line = (
            f"        {noise_str} & "
            f"{fmt_rmse(float(row['rmse_neurotd_quality']))} & "
            f"{fmt_rmse(float(row['rmse_neurotd_oracle']))} & "
            f"{fmt_rmse(float(row['rmse_dtw']))} & "
            f"{fmt_rmse(float(row['rmse_ctw']))} & "
            f"{fmt_rmse(float(row['rmse_fft']))} \\\\"
        )
        lines.append(line)

    lines.append(r"        \hline")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)
