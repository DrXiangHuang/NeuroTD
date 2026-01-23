"""
simulation.py: A module for running wavelet-based signal simulations.
TODO: sliding window view, low-pass filter,
"""

import numpy as np
import scipy.signal
from dtw import dtw
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
from skimage.registration import phase_cross_correlation
from tqdm import tqdm

from neurotd.neuro_signal import my_ricker
from neurotd.utils import circular_shift, create_sliding_window_view, rmse
from neurotd.visualization import show_chirp


# %%
class Simulation:
    def __init__(self, sampling_rate, N_ricker, width, shift0, noise_levels, is_debug=False):
        # % TODO: add dshift as a parameter
        self.sampling_rate = sampling_rate
        self.N_ricker = N_ricker
        self.width = width
        self.shift0 = shift0
        self.dshift = shift0 // 8
        self.noise_levels = noise_levels
        self.shifts = np.arange(self.shift0, self.shift0 + 10 * self.dshift, self.dshift)
        self.x_base0 = my_ricker(N_ricker, width)
        self.is_debug = is_debug

    def sample_to_time(self, sample, unit="s"):
        """Converts sample to time in seconds or milliseconds."""
        time_in_sec = sample / self.sampling_rate
        return time_in_sec if unit == "s" else time_in_sec * 1000

    def add_noise(self, signal, noise_std):
        """Adds Gaussian noise to a signal."""
        np.random.seed(0)
        noise = np.random.normal(scale=noise_std, size=len(signal))
        return signal + noise

    def simulate_shifted_signal(self):
        """Simulates the shifted version of the base signal."""
        x_shifted_list = [circular_shift(self.x_base0, dt=dt) for dt in self.shifts]
        return np.hstack(x_shifted_list)

    def run_simulation(self):
        """Runs the simulation and computes alignment errors."""
        shifts_err_rmse_all = np.zeros(len(self.noise_levels))
        shifts_err_rmse_dtw_all = np.zeros(len(self.noise_levels))
        x_shifted0 = self.simulate_shifted_signal()
        t = self.sample_to_time(np.arange(len(self.x_base0) * len(self.shifts)))

        for i_noise_std, noise_std in tqdm(enumerate(self.noise_levels), total=len(self.noise_levels)):
            print(f"Running simulation with noise_std={noise_std:.3f}")

            # Add noise to the base and shifted signals
            x_base = self.add_noise(self.x_base0, noise_std)
            x = np.tile(x_base, len(self.shifts))
            x_shifted = self.add_noise(x_shifted0, noise_std)

            # Compute alignment errors
            shifts_rec_dtw, shifts_rec = self.compute_alignment_errors(x, x_shifted)

            # Ensure that shifts_rec has the same length as self.shifts for RMSE calculation
            shifts_err_rmse_all[i_noise_std] = rmse(self.shifts[: len(shifts_rec)], shifts_rec)
            shifts_err_rmse_dtw_all[i_noise_std] = rmse(self.shifts[: len(shifts_rec_dtw)], shifts_rec_dtw)

            if self.is_debug:
                self.debug_visualization(x, x_shifted, t)

        return shifts_err_rmse_all, shifts_err_rmse_dtw_all

    def compute_alignment_errors(self, x, x_shifted):
        """Computes alignment errors between base and shifted signals using DTW and cross-correlation."""
        N_per_seg = self.N_ricker
        N_gap = self.N_ricker

        x_sliding_window = create_sliding_window_view(x, N_per_seg, N_gap)
        x_shifted_sliding_window = create_sliding_window_view(x_shifted, N_per_seg, N_gap)

        shifts_rec = np.zeros(x_sliding_window.shape[0])
        shifts_rec_dtw = np.zeros(x_sliding_window.shape[0])

        for i, (x_seg, x_shifted_seg) in enumerate(zip(x_sliding_window, x_shifted_sliding_window)):
            alignment = dtw(x_shifted_seg, x_seg)
            shifts_rec_dtw[i] = (alignment.index1 - alignment.index2).mean()

            filtered_x_seg = self.apply_low_pass_filter(x_seg)
            filtered_x_shifted_seg = self.apply_low_pass_filter(x_shifted_seg)
            shift_rec, _, _ = phase_cross_correlation(filtered_x_seg, filtered_x_shifted_seg, upsample_factor=10)
            shifts_rec[i] = -shift_rec

        return shifts_rec_dtw, shifts_rec

    def apply_low_pass_filter(self, signal, cutoff_freq=200, filter_order=4):
        """Applies a low-pass filter to a signal segment."""
        b, a = butter(filter_order, cutoff_freq, fs=self.sampling_rate, btype="low")
        return filtfilt(b, a, signal)

    def debug_visualization(self, x, x_shifted, t):
        """Provides detailed visualization when debugging."""
        fig, ax = plt.subplots()
        ax.plot(t, x, label="Base Noisy")
        ax.plot(t, x_shifted, label="Shifted Noisy")
        ax.legend()
        ax.set(xlabel="Time (s)", ylabel="Amplitude", title="Signal Visualization")
        ax.grid("minor")
        plt.show()

    def visualize(self, shifts_err_rmse_all, shifts_err_rmse_dtw_all):
        """Handles visualization of the alignment RMSE results."""
        with plt.rc_context({"savefig.bbox": "tight", "savefig.transparent": True}):
            fig, ax = plt.subplots()
            ax.plot(self.noise_levels, self.sample_to_time(shifts_err_rmse_all, "ms"), "o-", label="NeuroTD")
            ax.plot(self.noise_levels, self.sample_to_time(shifts_err_rmse_dtw_all, "ms"), "x-", label="DTW")
            ax.legend(["NeuroTD", "DTW"])
            ax.set(xlabel="Noise (std)", ylabel="Alignment RMSE (ms)", title="Reconstructed Time-Varying Shift")
            ax.grid("minor")
            plt.show()


# %%
if __name__ == "__main__":
    # Show chirp signal as in the original script
    show_chirp()

    # Define simulation parameters
    N_ricker = 128 * 4
    sampling_rate = N_ricker * 10
    width = N_ricker // 16
    shift0 = N_ricker // 64
    noise_levels = np.linspace(0, 0.2, 10)

    # Create and run the simulation
    simulation = Simulation(sampling_rate, N_ricker, width, shift0, noise_levels, is_debug=False)
    shifts_err_rmse_all, shifts_err_rmse_dtw_all = simulation.run_simulation()
    simulation.visualize(shifts_err_rmse_all, shifts_err_rmse_dtw_all)

# %%
