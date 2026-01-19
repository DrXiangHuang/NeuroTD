# %% continuous time-varying shift simulation + compare NeuroTD/DTW/CTW/FFT
import datetime
import pickle
from dataclasses import dataclass
from itertools import product
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from tqdm import tqdm

from neurotd.simulation_plotting import (
    colors,
    generate_table_full_latex,
    markers,
    plot_rmse_vs_noise_oracle_vs_quality,
)
from neurotd.time_varying_shift import (
    CTWShift,
    DTWShift,
    FFTShift,
    MSEShift,
    compute_windowed_mse_with_hop,
    evaluate_alignment_quality_sliding,
    sample_to_time_ms,
)
from neurotd.utils import InterpolatedVector, SlidingWindowPair, ensure_odd_window_size, rmse


# ------------------------ utilities ------------------------
@dataclass
class SpikeSpec:
    centers: np.ndarray  # shape (N_spike,)
    widths: np.ndarray  # shape (N_spike,)
    amp: float = 0.35


def make_time(duration: float = 1.0, sampling_rate: int = 5120) -> np.ndarray:
    """Time grid on [0, duration) with given sampling rate."""
    n = int(round(duration * sampling_rate))
    return np.linspace(0.0, duration, n, endpoint=False)


def make_spikes(t: np.ndarray, spec: SpikeSpec) -> np.ndarray:
    """Sum of Gaussians with centers and widths specified on the same [0,1) domain as t."""
    # If duration != 1, interpret centers in absolute seconds
    x = np.zeros_like(t, dtype=float)
    for c, w in zip(spec.centers, spec.widths):
        x += spec.amp * np.exp(-((t - c) ** 2) / (w**2))
    return x


def normalize_peak(x: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(x))
    return x / (m if m > 0 else 1.0)


def wrap01(z: np.ndarray) -> np.ndarray:
    """Wrap real values to [0,1) modulo 1."""
    return (z % 1.0 + 1.0) % 1.0


def interp1_periodic(x: np.ndarray, y: np.ndarray, tq: np.ndarray) -> np.ndarray:
    """
    Periodic linear interpolation on [0,1).
    x: sample locations in [0,1), strictly increasing, length N.
    y: values at x, length N.
    tq: query locations in [0,1).
    """
    # Append one point at x=1 with y=y[0] for periodicity
    x_ext = np.concatenate([x, [1.0]])
    y_ext = np.concatenate([y, [y[0]]])
    return np.interp(tq, x_ext, y_ext)


# ------------------------ core generators ------------------------
def generate_reference_signal(
    duration: float = 1.0,
    sampling_rate: int = 5120,
    N_spike: int = 10,
    base_freq_hz: float = 4.0,
    base_amp: float = 0.1,
    spike_amp: float = 0.5,
    center_jitter: float = 0.015,
    width_range: Tuple[float, float] = (0.005, 0.01),
    rng: Optional[np.random.Generator] = None,
):
    """
    Build x(t) = base sinusoid + N_spike Gaussians, normalized to unit peak.
    Centers are approximately uniform over [0,1) with a small jitter.
    Widths are drawn U[width_range].
    """
    if rng is None:
        rng = np.random.default_rng(0)

    t = make_time(duration, sampling_rate)
    t01 = t / duration  # normalize to [0,1)

    # approximate uniform centers + jitter, clipped to [0,1)
    centers_nominal = (np.arange(N_spike) + 0.5) / N_spike
    centers = np.clip(centers_nominal + rng.uniform(-center_jitter, center_jitter, size=N_spike), 0, 1)
    widths = rng.uniform(*width_range, size=N_spike)  # absolute seconds on [0,1) scale

    x_base = base_amp * np.sin(2 * np.pi * base_freq_hz * t)  # base sinusoid in seconds
    spikes = make_spikes(t01, SpikeSpec(centers=centers, widths=widths, amp=spike_amp))
    x = normalize_peak(x_base + spikes)

    return t, x, centers, widths


def build_delay_curve(
    t: np.ndarray,
    centers: np.ndarray,
    duration: float = 1.0,
    Delta0: float = 0.00,
    Delta1: float = 0.005,
    fDelta: float = 1.2,
    spike_jitter_std: float = 0.01,
    boundary_mode: str = "clamped",
    rng: Optional[np.random.Generator] = None,
):
    """
    Construct a smooth, time-varying delay curve \tilde{Δ}(t) for continuous simulations.

    The delay curve starts from a sinusoidal baseline Δ_base(t), applies random
    perturbations at spike centers, and fits a cubic spline through boundary
    points and perturbed spike constraints to yield a smooth function.

    Parameters
    ----------
    t : np.ndarray
        Time array in seconds, typically sampled uniformly on [0, duration).
    centers : np.ndarray
        Spike center positions, normalized to [0, 1). Each c_i is a location
        where the delay curve is perturbed.
    duration : float, default=1.0
        Duration of the signal in seconds. Used to normalize t to t01 = t/duration.
    Delta0 : float, default=0.06
        Baseline constant component of the delay curve (in seconds).
    Delta1 : float, default=0.035
        Amplitude of the sinusoidal modulation of the baseline delay.
    fDelta : float, default=1.2
        Frequency (Hz) of the sinusoidal modulation in the baseline delay curve.
    spike_jitter_std : float, default=0.005
        Standard deviation of Gaussian perturbations applied at each spike center,
        modeling per-event variability in delay.
    boundary_mode : str, default="clamped"
        Boundary condition for cubic spline interpolation:
        - "clamped": zero slope at endpoints,
        - "not-a-knot": standard cubic spline natural condition.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. If None, a default RNG is created.

    Returns
    -------
    t01 : np.ndarray
        Normalized time array on [0,1).
    delays : np.ndarray
        Smooth delay curve \tilde{Δ}(t) evaluated at each t01.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    t01 = t / duration
    Delta_base = Delta0 + Delta1 * np.sin(2 * np.pi * fDelta * t01)

    # per-spike edits at the spike centers
    eps = rng.normal(loc=0.0, scale=spike_jitter_std, size=len(centers))
    y_pts = (Delta0 + Delta1 * np.sin(2 * np.pi * fDelta * centers)) + eps

    # spline knots: 0, (centers sorted), 1
    order = np.argsort(centers)
    c_sorted = centers[order]
    y_sorted = y_pts[order]
    x_knots = np.concatenate([[0.0], c_sorted, [1.0]])
    y_knots = np.concatenate(
        [[Delta0 + Delta1 * np.sin(0.0)], y_sorted, [Delta0 + Delta1 * np.sin(2 * np.pi * fDelta)]]
    )

    cs = CubicSpline(x_knots, y_knots, bc_type=boundary_mode)
    delays = cs(t01)

    return t01, delays


def apply_continuous_delay(t01: np.ndarray, x: np.ndarray, delays: np.ndarray) -> np.ndarray:
    """
    Generate y(t) = x( (t - Delta(t)) mod 1 ) using periodic linear interpolation.
    """
    tq = wrap01(t01 - delays)  # warped time in [0,1)
    # x is sampled at uniform t01 grid; build an x-grid for interpolation:
    return interp1_periodic(t01, x, tq)


def plot_signals_and_delays(t, x, y, delays, title_prefix="", figsize=(10, 8)):
    """
    Plot reference signal x(t), delayed signal y(t), and delay curve Δ(t).

    Parameters
    ----------
    t : np.ndarray
        Time axis.
    x : np.ndarray
        Reference signal (possibly noisy).
    y : np.ndarray
        Delayed signal (possibly noisy).
    delays : np.ndarray
        Delay curve values Δ(t).
    title_prefix : str, optional
        Prefix for figure titles, e.g., 'Simulation 1'.
    """
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # --- Top subplot: signals ---
    axs[0].plot(t, x, label="x (reference)", color="red", alpha=0.8)
    axs[0].plot(t, y, label="y (delayed)", color="blue", alpha=0.8)
    axs[0].set_ylabel("amplitude")
    axs[0].set_title(f"{title_prefix} two time-series data")
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.6)

    # --- Bottom subplot: delays ---
    axs[1].plot(t, delays * 1000, label="Δ(t)", color="black")  # s -> ms
    axs[1].set_xlabel("time (s)")
    axs[1].set_ylabel("delay (ms)")
    axs[1].set_title(f"{title_prefix} corresponding time-varying delays")
    axs[1].legend()
    axs[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()
    return fig


# %%
# ------------------------ simulation and compare performance ------------------------
debug = False
RANDOM_SEED = 2025  # 0, 2025
debug_print = print if debug else lambda x: x

rng = np.random.default_rng(RANDOM_SEED)

# (A) reference with N_spike=10
duration, fs, N_spike = 1.0, 5120, 10
t, x0, centers, widths = generate_reference_signal(duration=duration, sampling_rate=fs, N_spike=N_spike, rng=rng)
t01 = t / duration

# (B) build smooth delay curve with per-spike random edits
_, delays = build_delay_curve(
    t=t,
    centers=centers,
    duration=duration,
    Delta0=0.00,
    Delta1=0.01 * 0.2,
    fDelta=0.1,
    spike_jitter_std=0.001,  # s, so 1 mns
    boundary_mode="clamped",
    rng=rng,
)

# (C) apply continuous delay
# delays = delays / 5  # scale down delays for easier alignment
y0 = apply_continuous_delay(t01, x0, delays)

# (D) add measurement noise
noise_std = 0.020
rng = np.random.default_rng(RANDOM_SEED)
x_noisy = x0 + rng.normal(0, noise_std, size=x0.size)
y_noisy = y0 + rng.normal(0, noise_std, size=y0.size)


fig = plot_signals_and_delays(
    t, x_noisy, y_noisy, delays, title_prefix="", figsize=(10, 7)
)  # title_prefix=Continuous delay simulation"
fig.savefig("fig3_continuous_delay_signals_and_delays.png", dpi=300, bbox_inches="tight")

# (D) segments like the discrete case, reuse  SlidingWindowPair downstream.
# From here, any estimator (NeuroTD/DTW/CTW/FFT) can be run on (x_noisy, y_noisy).


# %%  one simulation + compare NeuroTD/DTW/CTW/FFT
# ---------------- D) Compare NeuroTD/DTW/CTW/FFT in sliding windows ----------------
# Choose window/hop (match your discrete setup style)
# Example: about 2 windows per "segment" worth of samples; tune as you like.
approx_windows_per_second = 50
window_size = ensure_odd_window_size(int(round(fs / approx_windows_per_second)))  # 103
window_size = 513  # 257, 513, 1025
window_fun = 'Hann'  # 'rectangle' or 'Hann'
continuous = True  # True or False
# hop_size = window_size // 2
hop_size = 1

# Set up windowed pairs
sliding_window_pair = SlidingWindowPair(x_noisy, y_noisy, window_size=window_size, hop_size=hop_size)

# Ground-truth shifts (in samples) at window centers:
# - Your sign convention: positive shift = y delayed vs x
# - Our continuous truth: delays(t) is in seconds (duration=1s => delays already in s)
# - Convert center times to indices, then samples = delays * fs
center_idx = sliding_window_pair.center_indices.astype(int)
gt_shift_samples = delays[center_idx] * fs

# --- NeuroTD (windowed MSE baseline from your file) ---
centers_neurotd, shifts_neurotd = compute_windowed_mse_with_hop(
    x_noisy, y_noisy, window_size=window_size, hop_size=hop_size, window_fun=window_fun, continuous=continuous
)

# centers_neurotd are indices; align GT at those centers
gt_neurotd_samples = delays[centers_neurotd.astype(int)] * fs

# --- Other baselines (DTW/CTW/FFT) from your file ---
dtw_method = DTWShift(sliding_window_pair)  # returns shifts (samples) at swp.center_indices
ctw_method = CTWShift(sliding_window_pair)
# mse_method = MSEShift(swp)2
fft_method = FFTShift(sliding_window_pair, is_filter=False)

shifts_dtw = dtw_method.shifts
shifts_ctw = ctw_method.shifts
# shifts_mse = mse_method.shifts
shifts_fft = fft_method.shifts

# ---------------- E) RMSE (samples -> ms for readability) ----------------
rmse_neurotd_samp = rmse(gt_neurotd_samples, shifts_neurotd)
rmse_dtw_samp = rmse(gt_shift_samples, shifts_dtw)
rmse_ctw_samp = rmse(gt_shift_samples, shifts_ctw)
# rmse_mse_samp     = rmse(gt_shift_samples,   shifts_mse)
rmse_fft_samp = rmse(gt_shift_samples, shifts_fft)

print("RMSE (ms):")
print(f"  NeuroTD: {sample_to_time_ms(rmse_neurotd_samp, sampling_rate=fs):.3f}")
print(f"  DTW:     {sample_to_time_ms(rmse_dtw_samp,     sampling_rate=fs):.3f}")
print(f"  CTW:     {sample_to_time_ms(rmse_ctw_samp,     sampling_rate=fs):.3f}")
# print(f"  MSE:     {sample_to_time_ms(rmse_mse_samp,     sampling_rate=fs):.3f}")
print(f"  FFT:     {sample_to_time_ms(rmse_fft_samp,     sampling_rate=fs):.3f}")

# ---------------- F) Visualization of estimated shifts vs truth ----------------
# Plot on a common time axis (convert indices to seconds)
#  DTW and CTW, do it for the whole signal, rather than per window

marker_size = 10
linewidth = 2

time_axis = t  # seconds

plt.figure(figsize=(8, 6))
plt.plot(time_axis[center_idx], sample_to_time_ms(gt_shift_samples, fs), '-', label="Ground Truth", linewidth=2)

plt.plot(
    time_axis[centers_neurotd.astype(int)],
    sample_to_time_ms(shifts_neurotd, fs),
    ':',
    label=f"NeuroTD (RMSE={sample_to_time_ms(rmse_neurotd_samp, fs):.2f} ms)",
)
compare_DTW = True  # True or False
if compare_DTW:
    plt.plot(
        time_axis[center_idx],
        sample_to_time_ms(shifts_dtw, fs),
        ':',
        label=f"DTW (RMSE={sample_to_time_ms(rmse_dtw_samp, fs):.2f} ms)",
    )

compare_others = True  # True or False
if compare_others:
    plt.plot(
        time_axis[center_idx],
        sample_to_time_ms(shifts_ctw, fs),
        ':',
        label=f"CTW (RMSE={sample_to_time_ms(rmse_ctw_samp, fs):.2f} ms)",
    )
    # plt.plot(time_axis[center_idx], sample_to_time_ms(shifts_mse, fs),
    #          ':v', label=f"MSE (RMSE={sample_to_time_ms(rmse_mse_samp, fs):.2f} ms)")
    plt.plot(
        time_axis[center_idx],
        sample_to_time_ms(shifts_fft, fs),
        ':',
        label=f"FFT (RMSE={sample_to_time_ms(rmse_fft_samp, fs):.2f} ms)",
    )

plt.xlabel("time (s)")
plt.ylabel("delay (ms)")
plt.title(f"Continuous Time-varying shift: window={window_size} samples, hop={hop_size} samples")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("fig3_continuous_delay_estimated_shifts.png", dpi=300, bbox_inches="tight")
# Optional: alignment-quality score
q_rmse_global, q_rmse_per_win = evaluate_alignment_quality_sliding(
    x_noisy, y_noisy, np.round(shifts_neurotd).astype(int), centers_neurotd, window_size
)
print(
    f"{continuous=}, {noise_std=:.3f}, NeuroTD: RMSE={sample_to_time_ms(rmse_neurotd_samp, fs):.2f} ms, alignment quality (normalized RMSE): {q_rmse_global:.4g}, {window_size=}, {window_fun=}"
)
"""
noise_std=0.010, NeuroTD: RMSE=1.28 ms, alignment quality (normalized RMSE): 0.7466, window_size=257, window_fun='rectangle'
noise_std=0.010, NeuroTD: RMSE=1.19 ms, alignment quality (normalized RMSE): 0.8241, window_size=257, window_fun='Hann'
noise_std=0.010, NeuroTD: RMSE=2.15 ms, alignment quality (normalized RMSE): 0.7484, window_size=513, window_fun='rectangle'
noise_std=0.010, NeuroTD: RMSE=1.92 ms, alignment quality (normalized RMSE): 0.7764, window_size=513, window_fun='Hann'

noise_std=0.020, NeuroTD: RMSE=0.42 ms, alignment quality (normalized RMSE): 0.6796, window_size=513, window_fun='rectangle'
noise_std=0.020, NeuroTD: RMSE=0.46 ms, alignment quality (normalized RMSE): 0.6896, window_size=513, window_fun='Hann'
noise_std=0.020, NeuroTD: RMSE=0.89 ms, alignment quality (normalized RMSE): 0.9479, window_size=257, window_fun='rectangle'
noise_std=0.020, NeuroTD: RMSE=0.46 ms, alignment quality (normalized RMSE): 0.6896, window_size=513, window_fun='Hann'
----------above continuous = False -----------

continuous=True, noise_std=0.020, NeuroTD: RMSE=0.31 ms, alignment quality (normalized RMSE): 0.6896, window_size=513, window_fun='Hann'
"""


# %% broken-axis plot
linewidth = 1.5


fig, (ax_low, ax_high) = plt.subplots(
    2, 1, sharex=True, figsize=(6, 8), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
)

# -----------------
# Low axis: GT + NeuroTD + DTW + CTW
# -----------------
ax_low.plot(
    time_axis[center_idx],
    sample_to_time_ms(gt_shift_samples, fs),
    "-",
    color=colors["Ground Truth"],
    linewidth=linewidth,
    label="Ground Truth",
)

ax_low.plot(
    time_axis[centers_neurotd.astype(int)],
    sample_to_time_ms(shifts_neurotd, fs),
    "--",
    color=colors["NeuroTD"],
    linewidth=linewidth,
    label=f"NeuroTD (RMSE={sample_to_time_ms(rmse_neurotd_samp, fs):.2f} ms)",
)

ax_low.plot(
    time_axis[center_idx],
    sample_to_time_ms(shifts_dtw, fs),
    "-.",
    color=colors["DTW"],
    linewidth=linewidth,
    label=f"DTW (RMSE={sample_to_time_ms(rmse_dtw_samp, fs):.2f} ms)",
)

ax_low.plot(
    time_axis[center_idx],
    sample_to_time_ms(shifts_ctw, fs),
    ":",
    color=colors["CTW"],
    linewidth=linewidth,
    label=f"CTW (RMSE={sample_to_time_ms(rmse_ctw_samp, fs):.2f} ms)",
)

# -----------------
# High axis: FFT only (+ GT for reference)
# -----------------
ax_high.plot(
    time_axis[center_idx],
    sample_to_time_ms(gt_shift_samples, fs),
    "-",
    color=colors["Ground Truth"],
    linewidth=linewidth,
    label="Ground Truth",
)

ax_high.plot(
    time_axis[center_idx],
    sample_to_time_ms(shifts_fft, fs),
    "--",
    color=colors["FFT"],
    linewidth=linewidth,
    label=f"FFT (RMSE={sample_to_time_ms(rmse_fft_samp, fs):.2f} ms)",
)

# -----------------
# Axis limits (adjust if needed)
# -----------------
ax_low.set_ylim(-2, 3)
ax_high.set_ylim(-55, 55)

# -----------------
# Labels
# -----------------
ax_high.set_xlabel("time (s)")
fig.text(0.06, 0.5, "delay (ms)", va="center", rotation="vertical")

# -----------------
# Broken-axis diagonals
# -----------------
d = 0.015
kwargs = dict(color="k", clip_on=False)

ax_low.plot((-d, +d), (-d, +d), transform=ax_low.transAxes, **kwargs)
ax_low.plot((1 - d, 1 + d), (-d, +d), transform=ax_low.transAxes, **kwargs)

ax_high.plot((-d, +d), (1 - d, +d), transform=ax_high.transAxes, **kwargs)
ax_high.plot((1 - d, 1 + d), (1 - d, +d), transform=ax_high.transAxes, **kwargs)

# -----------------
# Legend & grid
# -----------------
ax_low.legend(loc="lower right")
ax_high.legend(loc="upper left")

ax_low.grid(True, linestyle="--", alpha=0.4)
ax_high.grid(True, linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig("fig_continuous_broken_axis_rmse.png", dpi=300, bbox_inches="tight")


# %% multiple simulations + compare NeuroTD/DTW/CTW/FFT

# find optimal window size for NeuroTD
# window_sizes = np.array([32, 64, 128, 256, 512]) + 1  # ensure odd, 2^n + 1, before 2026.01.16
window_sizes = np.array([32, 64, 128, 256, 512, 1024]) + 1  # ensure odd, 2^n + 1
window_sizes = ensure_odd_window_size(window_sizes)
print(f"Window sizes (ms): {window_sizes / fs * 1000}")  # in ms
hop_size = 1
# noise_stds = np.linspace(0, 0.1, 6)  # old before 2026.01.16
noise_stds = np.linspace(0, 0.2, 11)  #  [0.0, 0.02, 0.04, ..., 0.2] better than [0.0, 0.0222, 0.0444, ..., 0.2]

debug = False  # True or False, Set to False for full run
if debug:
    param_grid = [(0.020, 513)]  # single combination (noise, window_size_candidates)
    param_grid = list(product(noise_stds, [513]))  # single combination (noise, window_size_candidates)
else:
    param_grid = list(product(noise_stds, window_sizes))

results = []

for noise_std, window_size in tqdm(param_grid, desc="Running simulations"):
    # print(f"------Begin: noise_std={noise_std:.3f}")
    print(f"Noise std: {noise_std:.3f}, window size: {window_size}")
    #
    rng = np.random.default_rng(RANDOM_SEED)
    x_noisy = x0 + rng.normal(0, noise_std, size=x0.size)
    y_noisy = y0 + rng.normal(0, noise_std, size=y0.size)

    # ---------------- D) Compare NeuroTD/DTW/CTW/FFT in sliding windows ----------------

    # Set up windowed pairs
    sliding_window_pair = SlidingWindowPair(x_noisy, y_noisy, window_size=window_size, hop_size=hop_size)

    # gt_shift_samples: Ground-truth shifts (in samples) at window centers:
    # - sign convention: positive shift = y delayed vs x
    # - Our continuous truth: delays(t) is in seconds (duration=1s => delays already in s)
    # - Convert center times to indices, then samples = delays * fs
    center_idx = sliding_window_pair.center_indices.astype(int)
    gt_shift_samples = delays[center_idx] * fs

    # --- NeuroTD (windowed MSE baseline from your file) ---
    centers_neurotd, shifts_neurotd = compute_windowed_mse_with_hop(
        x_noisy, y_noisy, window_size=window_size, hop_size=hop_size, window_fun=window_fun, continuous=continuous
    )
    # centers_neurotd are indices; align GT at those centers
    gt_neurotd_samples = delays[centers_neurotd.astype(int)] * fs

    # --- Other baselines (DTW/CTW/FFT) from your file ---
    dtw_method = DTWShift(sliding_window_pair)  # returns shifts (samples) at swp.center_indices
    ctw_method = CTWShift(sliding_window_pair)
    # mse_method = MSEShift(swp)
    fft_method = FFTShift(sliding_window_pair, is_filter=False)

    shifts_dtw = dtw_method.shifts
    shifts_ctw = ctw_method.shifts
    # shifts_mse = mse_method.shifts
    shifts_fft = fft_method.shifts

    # ---------------- E) RMSE (samples -> ms for readability) ----------------
    rmse_neurotd_samp = rmse(gt_neurotd_samples, shifts_neurotd)
    rmse_dtw_samp = rmse(gt_shift_samples, shifts_dtw)
    rmse_ctw_samp = rmse(gt_shift_samples, shifts_ctw)
    # rmse_mse_samp     = rmse(gt_shift_samples,   shifts_mse)
    rmse_fft_samp = rmse(gt_shift_samples, shifts_fft)

    print("RMSE (ms):")
    print(f"  NeuroTD: {sample_to_time_ms(rmse_neurotd_samp, sampling_rate=fs):.3f}")
    print(f"  DTW:     {sample_to_time_ms(rmse_dtw_samp,     sampling_rate=fs):.3f}")
    print(f"  CTW:     {sample_to_time_ms(rmse_ctw_samp,     sampling_rate=fs):.3f}")
    # print(f"  MSE:     {sample_to_time_ms(rmse_mse_samp,     sampling_rate=fs):.3f}")
    print(f"  FFT:     {sample_to_time_ms(rmse_fft_samp,     sampling_rate=fs):.3f}")

    q_rmse_global, q_rmse_per_win = evaluate_alignment_quality_sliding(
        x_noisy, y_noisy, np.round(shifts_neurotd).astype(int), centers_neurotd, window_size
    )
    print(f"NeuroTD alignment quality (normalized RMSE): {q_rmse_global:.4g}")

    results.append(
        {
            "noise_std": noise_std,
            "window_size": window_size,
            "hop_size": hop_size,
            "rmse_neurotd": rmse_neurotd_samp,
            "quality_rmse_neurotd": q_rmse_global,
            "quality_rmse_per_window_neurotd": q_rmse_per_win,
            "rmse_ctw": rmse_ctw_samp,
            "rmse_dtw": rmse_dtw_samp,
            "rmse_fft": rmse_fft_samp,
        }
    )

# rmse_neurotd, rmse_ctw, etc are in samples, convert to ms using sample_to_time_ms when reporting/plotting.
df_results = pd.DataFrame(results)
# datestr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# df_results.to_csv(f"simulation_results_continuous_{datestr}.csv", index=False)
df_results.to_csv(f"simulation_results_continuous.csv", index=False)
with open(f"simulation_results_continuous.pkl", "wb") as f:
    pickle.dump(results, f)


# %% ---------------- F) Visualization of estimated shifts vs truth ----------------
df = df_results.copy()
plot_rmse_vs_noise_oracle_vs_quality(
    df, fs=fs, fig_name="fig3c_continuous_rmse_vs_noise", figsize=(5, 6), linewidth=linewidth, marker_size=marker_size
)

# %% generate the new one big full supplmentary table including RMSE for all methods
latex = generate_table_full_latex(
    csv_path=f"simulation_results_continuous.csv",
    sampling_rate_hz=5120.0,
    quality_decimals=3,
    rmse_decimals=3,
)
print(latex)
# %%
