"""
codes for generate:
(1) fig 2: simulation of discrete time-varying shifts
(2) supplementary table S1 and S2
2026.01.15_Frozen exactly the same as the paper submission
"""

# %%
import datetime
from dataclasses import dataclass
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from neurotd.rcParams import rcParams
from neurotd.neuro_signal import SegmentedSignal, my_ricker
from neurotd.simulation_plotting import (
    colors,
    generate_table_full_latex,
    markers,
    plot_rmse_vs_noise_oracle_vs_quality,
)
from neurotd.time_varying_shift import (  # below are simulation specific, keep in neurotd.time_varying_shift if multiple simulation, otherwise move here
    CTWShift,
    DTWShift,
    FFTShift,
    MSEShift,
    SignalParameters,
    compare_signals,
    compute_shifts_sliding_window_gt,
    compute_windowed_mse_with_hop,
    evaluate_alignment_quality_sliding,
    reconstruct_from_shifts,
    sample_to_time_ms,
    simulate_and_plot_shifted_signals,
)
from neurotd.utils import InterpolatedVector, SlidingWindowPair, ensure_odd_window_size, rmse

# %%
plt.ion()  # enable interactive mode for command-line script execution (non-blocking figure display)

def configure_plot():
    plt.rcParams["figure.figsize"] = [12, 8]
    plt.rcParams.update({"font.size": 18})


SHOW_TRACE_OF_ONE_NOISE_LEVEL = True  # True, False, True for one noise level and show subfigure b
RANDOM_SEED = 0
fs = sampling_rate = 5120
# debug_print = print if debug else lambda x: x

# simulate a noiseless signal and its shifted version
paras = SignalParameters()
# paras.noise_stds = np.linspace(0, 0.2, 10)  # legacy used in paper submission [0.0, 0.0222, 0.0444, ..., 0.2]
paras.noise_stds = np.linspace(0, 0.2, 11)  #  [0.0, 0.02, 0.04, ..., 0.2] better than [0.0, 0.0222, 0.0444, ..., 0.2]
seg_signal = SegmentedSignal(duration=1.0, sampling_rate=fs, n_segments=10, shape_func=my_ricker)
seg_signal_shifted = seg_signal.generate_shifted(paras.shifts)
configure_plot()

# % plot the base signal and its shifted noisy version
rng = np.random.default_rng(RANDOM_SEED)
simulate_and_plot_shifted_signals(seg_signal, paras.shifts, noise_std=0.2, rng=rng, fig_size=(8, 5.5))

# %% add noise to base signal and its shifted version, and compute alignment error
fig_size = (12, 8)
shifts_err_rmse_all = np.zeros(len(paras.noise_stds))
shifts_err_rmse_dtw_all = np.zeros(len(paras.noise_stds))
signal_name = "Mexican hat"

marker_size = 10
linewidth = 2

# %% Simulate viarious noise levels and window sizes: # [0.4, 1, 2, 3, 1 / 0.23, 10]
# window_sizes = np.array([51, 119, 171, 257, 513, 1281])  # legacy values used before 2026.01.16
# paras.noise_stds = np.linspace(0, 0.2, 10)  # legacy used before 2026.01.16 [0.0, 0.0222, 0.0444, ..., 0.2]
window_sizes = np.array([32, 64, 128, 256, 512, 1024]) + 1  # ensure odd, 2^n + 1
# better more 'integer' noise levels from 0 to 0.2  [0.0, 0.02, 0.04, ..., 0.2]
paras.noise_stds = np.linspace(0, 0.2, 11)

# for simplicity, we use odd window size, so that the center is an integer
window_sizes = ensure_odd_window_size(window_sizes)
window_sizes_ms = sample_to_time_ms(window_sizes)
print(f"Window sizes (ms): window_sizes_ms")  # in ms

RANDOM_SEED = 0
DEBUG_ONE_NOISE = False  # only for debug of one noise level
if DEBUG_ONE_NOISE:  # only for debug of one noise level
    # single noise level, we tested that with all segment_to_window_ratios we get similar results as it finds optimal
    # window size using quality metric
    param_grid = [(paras.noise_stds[1], 513)]
else:
    param_grid = list(product(paras.noise_stds, window_sizes))

results = []
rng = np.random.default_rng(RANDOM_SEED)
oracle_trace_bank = {}  # key: (noise_std, window_size), value: (shifts and indices for each method)
for noise_std, window_size in tqdm(param_grid, desc="Running simulations"):
    # noise_std = 0.0  # debug
    # noise_std = paras.noise_stds[0]
    # sliding window of the signal with shape (num_windows, window_size)
    # hop_size = window_size = seg_signal.n_samples_per_seg  # for debug

    hop_size = window_size // 2  # we always use half of the window size in discrete shift simulation

    print(f"------Begin: noise_std={noise_std:.3g}")
    rng = np.random.default_rng(RANDOM_SEED)
    seg_signal_noisy = seg_signal.generate_noisy(rng, noise_std)
    seg_signal_shifted_noisy = seg_signal_shifted.generate_noisy(rng, noise_std)

    x, y, t = seg_signal_noisy.data, seg_signal_shifted_noisy.data, seg_signal_noisy.time

    sliding_window_pair = SlidingWindowPair(x, y, window_size, hop_size)
    original_segments_center_idx = seg_signal.n_samples_per_seg // 2 + seg_signal.n_samples_per_seg * np.arange(
        seg_signal.n_segments
    )
    # DONE?: 2025.09.12_compare original_segments_center_idx with sliding_window_pair.center_indices

    errors_rec = np.zeros(sliding_window_pair.num_windows)
    diffphase_rec = np.zeros(sliding_window_pair.num_windows)

    dtw_method = DTWShift(sliding_window_pair)
    shifts_dtw = dtw_method.shifts  # Computation triggered automatically here

    ctw_method = CTWShift(sliding_window_pair)
    shifts_ctw = ctw_method.shifts  # Computation triggered automatically here

    mse_method = MSEShift(sliding_window_pair)
    shifts_mse = mse_method.shifts  # Computation triggered automatically here

    fft_method = FFTShift(sliding_window_pair, is_filter=False)
    shifts_fft = fft_method.shifts  # Computation triggered automatically here

    # DONE?: sliding_window_center_idx_neuroTD has 83 points, while sliding_window_pair.center_indices has 85 points, we
    # ignored boundary points. may already fixed
    # DONE?: debug above when window_size = hope_size = seg_signal.n_samples_per_seg, the
    # compute_windowed_mse_with_hop is wrong? may already fixed

    sliding_window_center_idx_neuroTD, shifts_sliding_window = compute_windowed_mse_with_hop(
        x, y, window_size, hop_size
    )

    shifts_sliding_window_gt_neuroTD = compute_shifts_sliding_window_gt(
        sliding_window_center_idx_neuroTD, original_segments_center_idx, paras.shifts
    )
    shifts_sliding_window_gt = compute_shifts_sliding_window_gt(
        sliding_window_pair.center_indices, original_segments_center_idx, paras.shifts
    )

    #  DONE?: 2025.05.21_sample to time conversion immediately after compute_windowed_mse_with_hop
    # shifts_sliding_window_gt_neuroTD = sample_to_time_ms(shifts_sliding_window_gt_neuroTD)
    shifts_err_rmse_neuroTD = rmse(shifts_sliding_window_gt_neuroTD, shifts_sliding_window)
    shifts_err_rmse_ctw = rmse(shifts_sliding_window_gt, shifts_ctw)
    shifts_err_rmse_dtw = rmse(shifts_sliding_window_gt, shifts_dtw)
    shifts_err_rmse_fft = rmse(shifts_sliding_window_gt, shifts_fft)

    # alignment_quality

    # DONE?: fix the start and end of the sliding window, we are not using the first and last window
    # For SlidingWindowPair(window_size=51, hop_size=25, num_windows=203, x.shape=(203, 51), y.shape=(203, 51))
    # sliding_window_center_idx_neuroTD is 201 points

    quality_rmse_neurotd, quality_rmse_per_window_neurotd = evaluate_alignment_quality_sliding(
        x, y, shifts_sliding_window, sliding_window_center_idx_neuroTD, window_size
    )

    # print(
    #     f"Max diff per window: {np.max(np.abs(quality_rmse_per_window_neurotd_old - quality_rmse_per_window_neurotd_old)):.4e}"
    # )

    # # -- Optional: Strict equality check --
    # if np.allclose(quality_rmse_per_window_neurotd, quality_rmse_per_window_neurotd_old, atol=1e-8):
    #     print("The results match!")
    # else:
    #     print("The results differ—investigate why.")

    results.append(
        {
            "noise_std": noise_std,
            "n_samples_per_seg": seg_signal.n_samples_per_seg,
            "window_fraction": window_size / seg_signal.n_samples_per_seg,
            "windows_per_segment": seg_signal.n_samples_per_seg / window_size,
            "window_size": window_size,
            "hop_size": hop_size,
            "rmse_neurotd": shifts_err_rmse_neuroTD,
            "quality_rmse_neurotd": quality_rmse_neurotd,
            "quality_rmse_per_window_neurotd": quality_rmse_per_window_neurotd,
            "rmse_ctw": shifts_err_rmse_ctw,
            "rmse_dtw": shifts_err_rmse_dtw,
            "rmse_fft": shifts_err_rmse_fft,
        }
    )

    if SHOW_TRACE_OF_ONE_NOISE_LEVEL and noise_std == paras.noise_stds[1] and window_size == 513:
        # Define result objects
        results_plot = [
            {
                "label": "Ground Truth",
                "shifts": shifts_sliding_window_gt_neuroTD,
                "indices": sliding_window_center_idx_neuroTD,
                "style": f"-{markers['Ground Truth']}",
            },
            {
                "label": "NeuroTD",
                "shifts": shifts_sliding_window,
                "indices": sliding_window_center_idx_neuroTD,
                "style": f":{markers['NeuroTD']}",
                "rmse": shifts_err_rmse_neuroTD,
            },
            {
                "label": "DTW",
                "shifts": shifts_dtw,
                "indices": sliding_window_pair.center_indices,
                "style": f":{markers['DTW']}",
                "rmse": shifts_err_rmse_dtw,
            },
            {
                "label": "CTW",
                "shifts": shifts_ctw,
                "indices": sliding_window_pair.center_indices,
                "style": f":{markers['CTW']}",
                "rmse": shifts_err_rmse_ctw,
            },
            {
                "label": "FFT",
                "shifts": shifts_fft,
                "indices": sliding_window_pair.center_indices,
                "style": f":{markers['FFT']}",
                "rmse": shifts_err_rmse_fft,
            },
        ]

        broken_y_axis = True
        if broken_y_axis:
            # ax_low: low values for GT, NeuroTD, DTW, CTW, ax_high: large values for FFT
            fig, (ax_low, ax_high) = plt.subplots(
                2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
            )

            ax_high.set(xlabel="time (s)")
            fig.text(0.07, 0.5, "delay (ms)", va="center", rotation="vertical")
            # Y limits: adjust based on observed ranges
            ax_low.set_ylim(-1.5, 8)  # GT + NeuroTD + DTW + CTW
            ax_high.set_ylim(-50, 55)  # FFT range  # ax_high.set_yticks([-50, -25, 0, 25, 50])
        else:
            fig, ax = plt.subplots()
            ax.set(xlabel="time (s)", ylabel="delay (ms)")

        # Plot using a loop
        time = InterpolatedVector(t)
        for res in results_plot:
            rmse_text = f" (RMSE={sample_to_time_ms(res['rmse']):.3g} ms)" if "rmse" in res else ""
            if broken_y_axis:
                target_ax = ax_high if res["label"] == "FFT" else ax_low
            else:
                target_ax = ax
            target_ax.plot(
                time[res["indices"]],
                sample_to_time_ms(res["shifts"]),
                res["style"],
                label=f"{res['label']}{rmse_text}",
                color=colors[res["label"]],
                linewidth=linewidth,
                markersize=marker_size,
            )

        title = f"Time-varying shift: noise std={noise_std:.3f}."
        if broken_y_axis:
            # Diagonal break markers
            d = 0.015
            kwargs = dict(transform=ax_low.transAxes, color="k", clip_on=False)
            ax_low.plot((-d, +d), (-d, +d), **kwargs)
            ax_low.plot((1 - d, 1 + d), (-d, +d), **kwargs)

            kwargs.update(transform=ax_high.transAxes)
            ax_high.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax_high.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

            ax_low.set_title(title)
            # fig.suptitle(title, y=0.98)

            # Legend intentionally on ax_low; FFT plotted only on ax_high
            ax_low.legend(loc="upper left")
            ax_high.legend(loc="upper left")
            ax_low.grid("minor")
            ax_high.grid("minor")
        else:
            ax.set(title=title)
            ax.legend()
            ax.grid("minor")
        fig_name = f"time_varying_shift_noise_hopvswindow_ratio={hop_size/window_size:.2f}_noise={noise_std:.3f}"

        fig.savefig(fig_name + ".png", dpi=300, bbox_inches="tight")


df_results = pd.DataFrame(results)
df_results.to_csv(f"simulation_results_discrete.csv", index=False)
df_results


# %% Paper figures and supplementary tables
df = df_results.copy()
plot_rmse_vs_noise_oracle_vs_quality(
    df,
    markers,
    colors,
    fs=5120,
    fig_name="fig2c_discrete_rmse_vs_noise",
    figsize=(10, 6.5),
    linewidth=linewidth,
    marker_size=marker_size,
    NeuroTD_with_oracle=False,  # True, False
)

# %% generate the full table including RMSE for all methods
# Additional benchmarking results, including NeuroTD quality scores, evaluated and selected window sizes at each noise level, RMSE values across all window sizes, and oracle versus quality-selected RMSE comparisons across methods, are provided in \textbf{Supplementary Tables S1}.
latex = generate_table_full_latex(
    csv_path=f"simulation_results_discrete.csv",
    sampling_rate_hz=5120.0,
    quality_decimals=3,
    rmse_decimals=3,
)
print(latex)

# --------------------End of paper figure and supplementary table generation-------------------- #
plt.show(block=True)  # keep the figures open for scripts run outside of an interactive environment
# %%
