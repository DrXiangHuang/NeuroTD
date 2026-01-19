from app1 import plot_region_shifts_with_stats
# %% Test compute shift by correlation
def demo_correlation_comparison():
    """compute best shift via cross-correlation, both linear and circular. Both fail for boundary segments."""

    def compute_best_shift_via_circular_cross_correlation(x_seg: np.ndarray, y_seg: np.ndarray) -> int:
        """
        Find the best shift between x_seg and y_seg using circular cross-correlation.

        Parameters:
        x_seg (np.ndarray): The first segment of the signal.
        y_seg (np.ndarray): The second segment of the signal.

        Returns:
        int: The best shift that aligns y_seg with x_seg.
        """
        n = len(x_seg)
        correlation = np.correlate(x_seg, y_seg, mode="full")
        best_shift = -np.argmax(correlation) - (n - 1)
        return best_shift

    def compute_best_shift_via_linear_cross_correlation(x_seg: np.ndarray, y_seg: np.ndarray) -> int:
        """
        Find the best shift between x_seg and y_seg using linear cross-correlation.

        Parameters:
        x_seg (np.ndarray): The first segment of the signal.
        y_seg (np.ndarray): The second segment of the signal.

        Returns:
        int: The best shift that aligns y_seg with x_seg.
        """
        correlation = np.correlate(x_seg, y_seg, mode="valid")
        best_shift = -np.argmax(correlation)
        return best_shift

    # Example signals (not circular shifts)
    x = np.array([0, 0, 1, 2, 3])
    y = np.array([0, 1, 2, 3, 4])

    # Compute cross-correlation using full mode
    # correlation = np.correlate(x, y, mode='full')
    x_padded = np.pad(x, (len(y) - 1, len(y) - 1))
    correlation = np.correlate(x_padded, y, mode="valid")

    # Find the shift (lag where correlation is maximized)
    lags = np.arange(-(len(y) - 1), len(x))
    shift = lags[np.argmax(correlation)]

    print("Cross-correlation:", correlation)
    print("Shift (lag):", shift)

    # Example signals
    x = np.array([0, 0, 1, 2, 3])
    y = np.array([0, 1, 2, 3, 0])

    # Compute full cross-correlation
    correlation = np.correlate(x, y, mode="full")

    # Generate lags corresponding to the "full" mode
    lags = np.arange(-(len(y) - 1), len(x))
    shift = lags[np.argmax(correlation)]

    print("CCor:", correlation)
    print("Lags:", lags)
    print("Shift (lag):", shift)


demo_correlation_comparison()


# %% naive mean squared error
x = np.array([0, 0, 1, 2, 3])
y = np.array([0, 1, 2, 3, 4])
# x, y = y, x


shifts, mse = compute_shifted_mse(x, y)
best_shift = shifts[np.argmin(mse)]

compute_best_shift_via_mse(x, y)
print(f"Shifts: {shifts}")
print(f"Errors: {mse}")
print(f"Best shift: {best_shift}")

fig, ax = plt.subplots()
ax.plot(shifts, mse, "rx-")
ax.set(xlabel="Shift", ylabel="Mean Squared Error")

correlation = np.correlate(x, y, mode="full")
mse_from_correlation = (np.sum(x**2) + np.sum(y**2) - 2 * correlation) / len(x)
ax.plot(shifts, correlation, "bx-")
ax.plot(shifts, mse_from_correlation, "ko-")
ax.legend(["MSE restricted to Mask", "Correlation", "MSE from Correlation"])

# %%


# %%
import numpy as np
from scipy.signal import istft, stft

# Parameters
fs = 1.0  # Sampling frequency
n_fft = 256  # FFT size
hop_length = 128  # Hop size, must be <= nperseg
window = "hann"

# Ensure valid noverlap
nperseg = n_fft  # Set segment size to FFT size
noverlap = nperseg - hop_length  # Overlap size
if noverlap >= nperseg:
    raise ValueError("Hop size must be smaller than the segment size (nperseg).")

# Compute STFT of both signals
f, t, X = stft(x, fs, window=window, nperseg=nperseg, noverlap=noverlap)
_, _, Y = stft(y, fs, window=window, nperseg=nperseg, noverlap=noverlap)

# Cross-spectral density
C = X * np.conj(Y)

# Cross-correlation via IFFT
cross_corr = np.fft.ifft(C, axis=-1).real

# Best alignment shifts
best_shifts = np.argmax(cross_corr, axis=-1) - n_fft // 2


import matplotlib.pyplot as plt

# %% ShortTimeFFT example
import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

T_x, N = 1 / 20, 1000  # 20 Hz sampling rate for 50 s signal
t_x = np.arange(N) * T_x  # time indexes for signal
f_i = 1 * np.arctan((t_x - t_x[N // 2]) / 2) + 5  # varying frequency
x = np.sin(2 * np.pi * np.cumsum(f_i) * T_x)  # the signal

g_std = 8  # standard deviation for Gaussian window in samples
w = gaussian(50, std=g_std, sym=True)  # symmetric Gaussian window
SFT = ShortTimeFFT(w, hop=10, fs=1 / T_x, mfft=200, scale_to="magnitude")
Sx = SFT.stft(x)  # perform the STFT


def compute_windowed_mse_with_hop_test():
    # Example usage
    x = np.sin(np.linspace(0, 2 * np.pi, 100))  # Example signal
    rng = np.random.default_rng(0)
    y = np.roll(x, 5) + 0.1 * rng.normal(0, 0.5, len(x))  # Shifted and noisy version
    window_size = 20

    # Default hop_size (half of window_size)
    window_centers, shifts = compute_windowed_mse_with_hop(x, y, window_size)
    print("Window center indices:", window_centers)
    print("Time-varying optimal shifts:", shifts)

    plt.plot(x, label="Reference signal")
    plt.plot(y, label="Shifted and noisy signal")


# %%

import numpy as np
from scipy.signal import istft, stft

# References:
# # sliding window:  https://docs.scipy.org/doc/scipy/tutorial/signal.html#tutorial-stft


def compute_stft_cross_correlation(x, y, window_size, hop_size):
    """
    Compute the cross-correlation for a sliding window using STFT.

    Parameters
    ----------
    x : np.ndarray
        The first signal.
    y : np.ndarray
        The second signal.
    window_size : int
        Size of the sliding window.
    hop_size : int
        Step size for the sliding window.

    Returns
    -------
    cross_corr : list of np.ndarray
        Cross-correlation values for each window.
    """
    assert len(x) == len(y), "Signals must have the same length."

    # Compute STFT for both signals
    _, _, X_stft = stft(x, nperseg=window_size, noverlap=window_size - hop_size, return_onesided=False)
    _, _, Y_stft = stft(y, nperseg=window_size, noverlap=window_size - hop_size, return_onesided=False)

    # Initialize list to store cross-correlations for each window
    cross_corr_list = []

    # Compute cross-correlation for each window in the STFT
    for i in range(X_stft.shape[1]):
        # Get the FFT of the windowed segments
        X_w = X_stft[:, i]
        Y_w = Y_stft[:, i]

        # Cross-correlation in the frequency domain
        Z = X_w * np.conj(Y_w)

        # Compute inverse FFT to get cross-correlation in time domain
        r_w = np.fft.ifft(Z).real

        # Append the valid part of the cross-correlation
        cross_corr_list.append(r_w[:window_size])

    return cross_corr_list


# Example usage
x = np.sin(np.linspace(0, 2 * np.pi, 100))  # Example signal
y = np.roll(x, 5) + 0.1 * np.random.randn(100)  # Shifted and noisy version
window_size = 20
hop_size = 10

cross_corr = compute_stft_cross_correlation(x, y, window_size, hop_size)
for i, r_w in enumerate(cross_corr):
    print(f"Window {i}, Cross-correlation:", r_w)

# %% Dummy figure D
import matplotlib.pyplot as plt
import numpy as np

# Given noise levels
noise_levels = np.array([0.00, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20])

# Given RMSE values for NeuroTD and DTW
rmse_neurotd = np.array([0.01, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25])
rmse_dtw = np.array([0.01, 0.95, 1.02, 1.08, 1.15, 1.18, 1.25])

# Generate RMSE values for CTW and FFT
rmse_ctw = np.array([0.02, 0.5, 0.7, 1.5, 2.0, 2.5, 3.0])
# seed
np.random.seed(0)
rmse_fft = rmse_ctw + np.random.uniform(0.1, 0.3, size=rmse_ctw.shape)  # FFT should be worse than CTW


# Re-plot with matching colors
# plt.figure(figsize=(5, 3))
# Re-plot with bold lines
linewidth = 4
fig, ax = plt.subplots()
ax.plot(noise_levels, rmse_neurotd, "x-", label="NeuroTD", color=colors["NeuroTD"], linewidth=linewidth)
ax.plot(noise_levels, rmse_dtw, "x-", label="DTW", color=colors["DTW"], linewidth=linewidth)
ax.plot(noise_levels, rmse_ctw, "x-", label="CTW", color=colors["CTW"], linewidth=linewidth)
ax.plot(noise_levels, rmse_fft, "x-", label="FFT", color=colors["FFT"], linewidth=linewidth)

# Labels and legend
ax.set_xlabel("noise (std)")
ax.set_ylabel("alignment RMSE (ms)")
ax.set_title("Alignment performance vs. noise level")  # Added title
ax.legend()
ax.grid(True)

# Show the plot
plt.show()

fig_name = f"time_varying_shift_noise_hopvswindow_ratio={hop_size/window_size:.2f}"
fig.savefig(fig_name + ".png", dpi=300)

# %% dummy figure 3C
# 1. Hippocampus - Learning and memory
# 2. Superior Temporal Gyrus - Sound processing and speech comprehension
# 3. Middle Temporal Gyrus - Language and memory
# 4. Insular Gyrus - Decision-making, emotions
# 5. Amygdala - Emotional responses
#  Hippocampus > Middle Temporal Gyrus > Insular Gyrus > Superior Temporal Gyrus > Amygdala?
regions = ["Hippocampus", "Superior Temporal Gyrus", "Middle Temporal Gyrus", "Insular Gyrus", "Amygdala"]
# create dummy data and box plots showing the time-varying shifts for each region
# Create dummy data

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.stats.multitest as smm
from scipy.stats import ttest_ind

# Define brain regions
regions = ["Hippocampus", "Superior Temporal Gyrus", "Middle Temporal Gyrus", "Insular Gyrus", "Amygdala"]


# Function to generate data with a mixture of normal distributions and additional outliers
def generate_mixture_data(mean1, std1, mean2, std2, size1, size2, outlier_mean, outlier_std, outlier_size):
    data_main = np.concatenate(
        [np.random.normal(loc=mean1, scale=std1, size=size1), np.random.normal(loc=mean2, scale=std2, size=size2)]
    )
    outliers = np.random.normal(loc=outlier_mean, scale=outlier_std, size=outlier_size)
    return np.concatenate([data_main, outliers])


# Generate data
delays_final_subtle = {
    "Hippocampus": generate_mixture_data(250, 45, 270, 30, 80, 20, 320, 35, 5) / 10,
    "Superior Temporal Gyrus": generate_mixture_data(210, 40, 230, 25, 80, 20, 270, 30, 5) / 10,
    "Middle Temporal Gyrus": generate_mixture_data(240, 45, 260, 30, 80, 20, 300, 35, 5) / 10,
    "Insular Gyrus": generate_mixture_data(220, 50, 240, 30, 80, 20, 290, 40, 5) / 10,
    "Amygdala": generate_mixture_data(215, 50, 235, 30, 80, 20, 280, 40, 5) / 10,
}


def plot_region_shifts_with_stats_old(delays_final_subtle):
    """
    Generate violin plots and perform pairwise t-tests with Bonferroni correction
    for time-varying shifts across brain regions.

    Parameters
    ----------
    delays_final_subtle : dict
        Dictionary mapping region names to arrays of shift values.
    """
    # Create DataFrame
    data_final_subtle = []
    regions = list(delays_final_subtle.keys())
    for region in regions:
        data_final_subtle.extend([(region, val) for val in delays_final_subtle[region]])

    df_final_subtle = pd.DataFrame(data_final_subtle, columns=["Region", "Delay (ms)"])
    df_final_subtle["Region"] = pd.Categorical(df_final_subtle["Region"], categories=regions, ordered=True)

    # Perform pairwise T-tests
    pairwise_tests = []
    for region1, region2 in combinations(regions, 2):
        t_stat, p_value = ttest_ind(
            df_final_subtle[df_final_subtle["Region"] == region1]["Delay (ms)"],
            df_final_subtle[df_final_subtle["Region"] == region2]["Delay (ms)"],
            equal_var=False,
        )
        pairwise_tests.append((region1, region2, p_value))

    # Convert to DataFrame
    t_test_results = pd.DataFrame(pairwise_tests, columns=["Region 1", "Region 2", "Raw p-value"])

    # Apply Bonferroni correction
    t_test_results["Corrected p-value"] = smm.multipletests(t_test_results["Raw p-value"], method="bonferroni")[1]

    # Determine significance levels
    def significance_marker(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"

    t_test_results["Significance"] = t_test_results["Corrected p-value"].apply(significance_marker)

    # Print results
    print("\nPairwise T-Test Results with Bonferroni Correction:")
    print(t_test_results.to_string(index=False))

    # Create the violin plot
    plt.figure(figsize=(8, 6))
    ax = sns.violinplot(data=df_final_subtle, x="Region", y="Delay (ms)", palette="Blues", inner="box", cut=0)

    # Overlay significant comparisons without overlapping lines
    y_max = df_final_subtle["Delay (ms)"].max()
    y_offsets = {}

    for i, (region1, region2, p_value, sig) in enumerate(
        zip(
            t_test_results["Region 1"],
            t_test_results["Region 2"],
            t_test_results["Corrected p-value"],
            t_test_results["Significance"],
        )
    ):
        if sig != "ns":
            x1, x2 = regions.index(region1), regions.index(region2)
            y = y_max + 5 + (i * 3)
            y_offsets[(x1, x2)] = y
            ax.plot([x1, x1, x2, x2], [y, y + 1, y + 1, y], lw=1.5, color="black")
            ax.text((x1 + x2) / 2, y + 1.5, sig, ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.xlabel("Brain region")
    plt.ylabel("Time-varying delay (ms)")
    plt.title("Time-varying shifts across brain regions")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True)
    plt.show()


plot_region_shifts_with_stats(delays_final_subtle)

# Create DataFrame

plot_region_shifts_with_stats(delays_final_subtle)

# %% fig 3d
import matplotlib.pyplot as plt
import numpy as np

# Generate random synthetic data similar to the plot
np.random.seed(0)  # For reproducibility

# Simulated data
N = 862
distance = np.random.uniform(0, 30, N)  # Distances between neuron pairs (0 to 30 mm)
delay = np.random.normal(loc=distance * 0.1, scale=1.5, size=N)  # Delays correlated with distance

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(distance, delay, color="orange", marker="x", alpha=0.7, label="Neuron Pairs")

# Labels and legend
plt.xlabel("Distance (mm)", fontsize=12)
plt.ylabel("Delay (ms)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)

# Show plot
plt.show()

# %% fig 3d density Density Scatter Plot
import seaborn as sns

# Create scatter plot with density shading
def plot_density_scatter_with_regression(distance, delay):
    """
    Plot a density scatter (KDE) plot with regression line for distance vs delay.

    Parameters
    ----------
    distance : array-like
        Distances between neuron pairs.
    delay : array-like
        Delays corresponding to each pair.
    """
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=distance, y=delay, fill=True, cmap="Oranges", levels=100, alpha=0.7)
    plt.scatter(distance, delay, color="black", s=5, alpha=0.4)  # Add raw scatter points

    # Add regression line
    sns.regplot(x=distance, y=delay, scatter=False, color="blue", line_kws={"linewidth": 2})

    # Labels and title
    plt.xlabel("Distance (mm)", fontsize=12)
    plt.ylabel("Delay (ms)", fontsize=12)
    plt.title("Density scatter plot (KDE) with regression line", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)

# Example usage:
plot_density_scatter_with_regression(distance, delay)

# Show plot
plt.show()

# %% Fig 3b heat map
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import seaborn as sns

# Load the provided 32x32 matrix (without treating any column as index)
file_path = "shifted_ms_32x32.csv"
df_heatmap = pd.read_csv(file_path, header=None)  # No index column

# Take absolute value, then normalize to [0, 1]
df_heatmap = np.abs(df_heatmap) / np.max(np.abs(df_heatmap))

# Ensure row and column labels are set as 0, 1, ..., 31
df_heatmap.index = range(32)
df_heatmap.columns = range(32)

# Perform hierarchical clustering on both axes
row_linkage = sch.linkage(ssd.pdist(df_heatmap), method="ward")
col_linkage = sch.linkage(ssd.pdist(df_heatmap.T), method="ward")

# Assign random region names for 32 rows
regions = ["Hippocampus", "Superior Temporal Gyrus", "Middle Temporal Gyrus", "Insular Gyrus", "Amygdala"]
row_labels = np.random.choice(regions, size=32)

# Create a clustered heatmap
sns.set(font_scale=0.8)  # Adjust font size
g = sns.clustermap(
    df_heatmap,
    row_linkage=row_linkage,
    col_linkage=col_linkage,
    cmap="coolwarm",  # Diverging color scale (blue to red)
    linewidths=0.5,
    figsize=(10, 10),
    dendrogram_ratio=(0.1, 0.1),  # Adjust dendrogram size
    cbar_pos=(-0.05, 0.8, 0.05, 0.18),  # Move colorbar further left
)

# Get the reordered row indices from clustering
row_order = g.dendrogram_row.reordered_ind

# Reorder the row labels based on clustering
new_row_labels = [row_labels[i] for i in row_order]

# Set new y-tick labels
ax = g.ax_heatmap
ax.set_yticks(np.arange(len(new_row_labels)) + 0.5)  # Ensure proper alignment
ax.set_yticklabels(new_row_labels, rotation=0)  # Rotate labels for readability

# Move and label the colorbar
g.cax.set_ylabel("Time Delays (normalized)", fontsize=12)
g.cax.yaxis.set_label_position("left")  # Move label to left side

# Show the plot
plt.show()

# %%
