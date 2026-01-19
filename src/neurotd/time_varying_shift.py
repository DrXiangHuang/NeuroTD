"""
TODO: EXTRACT code for Fig. 2 and supplementary material after submission
Time-varying shift signal simulation
https://laurentrdc.xyz/posts/mnxc.html

Shifts are defined as:
    shift = t_y - t_x,
where t_x and t_y are the time (or sample index) of the same event in x (reference) and y.
- A positive shift means y is delayed with respect to x (i.e., y occurs after x).
- When aligning y to x, we sample from y[i + shift] to match x[i] for a given shift.

For matched signal segments:
    y_indices = x_indices + shift
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from dtw import dtw
from matplotlib import pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq, ifft, irfft, rfft, rfftfreq
from skimage.registration import phase_cross_correlation
from tslearn.metrics import ctw_path, dtw_path

from neurotd.utils import SlidingWindowPair, rmse


# %%
@dataclass
class SignalParameters:
    sample_per_seg: int = 128 * 4
    sampling_rate: int | float = field(init=False)
    _shift0: int = field(init=False)
    _dshift: int = field(init=False)
    shifts: np.ndarray = field(init=False)
    noise_stds: np.ndarray = field(init=False)

    def __post_init__(self):
        self.sampling_rate = self.sample_per_seg * 10
        self._shift0 = self.sample_per_seg // 64
        self._dshift = self._shift0 // 8
        self.shifts = np.arange(self._shift0, self._shift0 + 10 * self._dshift, self._dshift)
        self.noise_stds = np.linspace(0, 0.2, 10)


paras = SignalParameters()

# %%


def sample_to_time_s(sample, sampling_rate=paras.sampling_rate):
    """convert sample to time (s)"""
    return sample / sampling_rate


def sample_to_time_ms(sample, sampling_rate=paras.sampling_rate):
    """convert sample to time (ms)"""
    return sample_to_time_s(sample, sampling_rate) * 1000


def show_mexican_hat_shifted(seg_signal, seg_signal_noisy, seg_signal_shifted_noisy, shifts, fig_size=(8, 5.5)):
    x_base0 = seg_signal.segments[0]
    x_base = seg_signal_noisy.segments[0]
    t = seg_signal_noisy.time

    # % plot the base signal and its noisy version
    rmse_ = rmse(x_base0, x_base)
    print(f"Mean Squared Error: {rmse_:.4f}")
    plt.plot(x_base0, label="x_base_noiseless")
    plt.plot(x_base, label="x_base_noisy")
    plt.legend()

    # % plot the noisy original and noisy shifted signal, and zoom in to show the time-varying delay
    with plt.rc_context({"savefig.bbox": "tight", "savefig.transparent": True, "figure.figsize": fig_size}):
        # plot the noisy original and noisy shifted signal
        fig, ax = plt.subplots()
        ax.plot(t, seg_signal_noisy.data, "b", label="$x_{reference}$")
        ax.plot(t, seg_signal_shifted_noisy.data, "r", label="$y_{delayed}$")
        ax.legend()
        ax.set(xlabel="time (s)", ylabel="amplitude", title="A. time-varying delay of two signals")
        ax.grid("minor")
        fig.savefig("mexican_hat_shift.png")

        # plot the zoomed-in version of the time-varying delay
        with plt.rc_context({"savefig.bbox": "tight", "savefig.transparent": True, "figure.figsize": (6, 2)}):
            for i in [0, 1, 9]:
                seg_time = seg_signal_noisy.time_of_segment[i]
                ax.set_xlim((seg_time.min(), seg_time.max()))
                ax.set_title(f"Zoomed in: delayed by {sample_to_time_ms(shifts[i]):.2f} ms")
                fig.savefig(f"mexican_hat_shift_{i}.png")

        # plot the error between the shifted and original signal
        fig, ax = plt.subplots()
        ax.plot(t, seg_signal_shifted_noisy.data - seg_signal_noisy.data, "b", label="r$x_{reference}$")
        ax.grid("minor")
        ax.set(xlabel="time (s)", ylabel="x_shifted-x", title="difference of two signal")
        fig.savefig("mexican_hat_shift_error.png")


def simulate_and_plot_shifted_signals(seg_signal, shifts, noise_std, rng, fig_size=(8, 5.5)):
    """
    Simulate and plot shifted signals with added noise.

    Parameters:
    seg_signal (SegmentedSignal): The original segmented signal.
    shifts (Iterable): An array of shifts to be applied to each segment of the base signal.
    noise_std (float): Standard deviation of the Gaussian noise to be added.
    random_seed (int): Seed for the random number generator.
    fig_size (tuple): Figure size for plotting.

    Returns:
    None
    """
    seg_signal_shifted = seg_signal.generate_shifted(shifts)

    seg_signal_noisy = seg_signal.generate_noisy(rng, noise_std)
    seg_signal_shifted_noisy = seg_signal_shifted.generate_noisy(rng, noise_std)

    seg_signal_noisy.plot()
    seg_signal_shifted_noisy.plot()

    show_mexican_hat_shifted(seg_signal, seg_signal_noisy, seg_signal_shifted_noisy, shifts, fig_size)


# %%
def compute_shifted_mse(x, y):
    """Compute the mean squared error between two signals for different shifts.
    We only compute the error for overlapped signals.
    Currently only assume len(x) = len(y), we may also restrict the shift to be less than half of the signal length.
    DONE?: Generalize this function to handle signals of different lengths
    Restrict the shift to be less than 1/4 half of the signal length
    DONE?: Normalized by the signal energy

    Parameters
    ----------
    x : np.ndarray
        The first signal, base signal not shifted
    y : np.ndarray
        The second signal, which is shifted with respect to the first signal

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        (shifts, mean_squared_errors), The shifts and the mean squared errors for each shift
    """
    M, N = len(x), len(y)
    shifts = np.arange(-N // 4 + 1, M // 4)  #  shifts = np.arange(-N + 1, M)
    mean_squared_errors = np.zeros(len(shifts))
    for i, shift in enumerate(shifts):
        if shift == 0:
            mean_squared_errors[i] = np.mean((x - y) ** 2)
        elif shift < 0:
            mean_squared_errors[i] = np.mean((x[:shift] - y[-shift:]) ** 2)
        else:
            mean_squared_errors[i] = np.mean((x[shift:] - y[:-shift]) ** 2)
    return shifts, mean_squared_errors


def compute_best_shift_via_mse(x, y):
    """Compute the best shift between two signals using the mean squared error.

    Parameters
    ----------
    x : np.ndarray
        The first signal, base signal not shifted
    y : np.ndarray
        The second signal, which is shifted with respect to the first signal

    Returns
    -------
    int
        The current shift of y vs x, which is the negative of the best shift that aligns y with x
    """
    shifts, mse = compute_shifted_mse(x, y)
    best_shift = -shifts[np.argmin(mse)]
    return best_shift


def compute_windowed_mse_with_hop_legacy(x, y, window_size, hop_size=None):
    """
    Compute the mean squared error (MSE) for each sliding window and shift with a given hop size,
    handling boundary conditions by restricting the window to valid indices.

    Parameters
    ----------
    x : np.ndarray
        The first signal (reference).
    y : np.ndarray
        The second signal (shifted).
    window_size : int
        The size of the sliding window.
    hop_size : int, optional
        The step size for the sliding window. Defaults to half of `window_size`.

    Returns
    -------
    tuple
        (window_starts, optimal_shifts) - Window start indices and the corresponding optimal shifts.

    """
    if hop_size is None:
        hop_size = window_size // 2  # Default hop size

    N = len(x)
    assert len(x) == len(y), "Signals must be of the same length"

    # Define the range of shifts
    max_shift = window_size // 2
    shifts = np.arange(-max_shift, max_shift + 1)

    # Restrict window range to avoid boundary issues
    valid_window_start = range(max_shift, N - window_size - max_shift + 1, hop_size)

    # Initialize storage for optimal shifts and window start positions
    optimal_shifts = []
    window_starts = []

    # Slide the window over the signal within the valid range
    for k in valid_window_start:
        # Track the window start index
        window_starts.append(k)

        # Extract the windowed segment of x
        window_x = x[k : k + window_size]
        min_mse = float("inf")
        best_shift = 0

        # Compute MSE for each shift
        for tau in shifts:
            # Compute indices for y
            y_start = k - tau
            y_end = k + window_size - tau

            # Ensure indices are valid and compute overlap
            if 0 <= y_start < y_end <= N:
                y_segment = y[y_start:y_end]
                mse = np.mean((window_x - y_segment) ** 2)

                # Update the best shift for this window
                if mse < min_mse:
                    min_mse = mse
                    best_shift = tau
            else:
                # should not reach here after Restrict window range to avoid boundary issues
                raise Exception("Invalid indices for shift")

        optimal_shifts.append(best_shift)

    # window_centers = np.array(window_starts) + window_size // 2
    window_centers = np.array(window_starts) + (window_size - 1) / 2  # center could be fractional, 2025.06.01_TO-CHECK
    optimal_shifts = -np.array(optimal_shifts)

    return window_centers, np.array(optimal_shifts)


def compute_windowed_mse_with_hop(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
    hop_size: int = None,
    window_fun: str = 'rectangle',
    continuous: bool = False,
):
    """
    Compute optimal shifts per sliding window by minimizing local weighted MSE between x and y,
    with support for different window functions and optional continuous reconstruction.

    Parameters
    ----------
    x
        Reference signal of length N.
    y
        Shifted signal of length N.
    window_size
        Number of samples per sliding window (W).
    hop_size
        Step between consecutive windows; if None, defaults to half of window_size.
    window_fun
        Window function: 'rectangle' or 'Hann'.
    continuous
        If True, reconstruct a continuous shift estimate over time.

    Returns
    -------
    If continuous == False:
        window_centers, optimal_shifts (as before).
    If continuous == True:
        time_indices, continuous_delay_curve.
    """

    if hop_size is None:
        hop_size = window_size // 2

    assert len(x) == len(y), "Signals x and y must have identical length"
    N = len(x)

    # Precompute window weights
    if window_fun == 'rectangle':
        weights = np.ones(window_size, dtype=float)
    elif window_fun == 'Hann':
        n = np.arange(window_size)
        weights = 0.5 - 0.5 * np.cos(2 * np.pi * n / (window_size - 1))
    else:
        raise ValueError(f"Unsupported window_fun: {window_fun}")

    # Normalize so sum(weights) == 1
    weights /= np.sum(weights)

    # Candidate shifts range
    max_shift = window_size // 2
    shifts = np.arange(-max_shift, max_shift + 1)

    # Valid window starts
    valid_starts = range(max_shift, N - window_size - max_shift + 1, hop_size)

    window_centers = []
    optimal_shifts_list = []

    for k in valid_starts:
        window_centers.append(k + (window_size - 1) / 2)
        x_seg = x[k : k + window_size]

        best_tau = 0
        min_mse = np.inf
        for tau in shifts:
            y_start = k - tau
            y_end = y_start + window_size
            if 0 <= y_start and y_end <= N:
                y_seg = y[y_start:y_end]
                diff = x_seg - y_seg
                weighted_mse = np.sum(weights * (diff * diff))
                if weighted_mse < min_mse:
                    min_mse = weighted_mse
                    best_tau = tau

        optimal_shifts_list.append(-best_tau)

    window_centers = np.array(window_centers, dtype=int)
    optimal_shifts = np.array(optimal_shifts_list)

    if not continuous:
        return window_centers, optimal_shifts

    # OVERLAP-WEIGHTED RECONSTRUCTION
    # Build a per-sample continuous delay by overlap adding weighted shifts
    delay = np.zeros(N, dtype=float)
    weight_accum = np.zeros(N, dtype=float)

    for center, tau in zip(window_centers.astype(int), optimal_shifts):
        start = center - window_size // 2
        end = start + window_size
        if 0 <= start and end <= N:
            delay[start:end] += tau * weights
            weight_accum[start:end] += weights

    valid_mask = weight_accum > 0
    delay[valid_mask] /= weight_accum[valid_mask]

    # Sample the smooth continuous curve at the window centers
    optimal_shifts = delay[window_centers.astype(int)]

    return window_centers, optimal_shifts


def compute_windowed_mse_with_hop_segments(sliding_window_pair):
    """
     NEW VERSION of compute_windowed_mse_with_hop() may used in future after fix and verification
    Uses sliding window segments directly, boundary conditon and sign convension need to be checked,

    Compute the optimal shift for each sliding window in a SlidingWindowPair,
    minimizing the MSE between each segment of x and possible shifted segments of y.

    Parameters
    ----------
    sliding_window_pair : SlidingWindowPair
        Object with .x (segments of x), .y (segments of y),
        .num_windows, .window_size, and .center_indices.

    Returns
    -------
    center_indices : np.ndarray
        Center indices (float or int) for each window.
    optimal_shifts : np.ndarray
        Optimal shift (in samples) for each window.
    """
    num_windows = sliding_window_pair.num_windows
    window_size = sliding_window_pair.window_size
    x_windows = sliding_window_pair.x
    y_windows = sliding_window_pair.y

    max_shift = window_size // 2
    shift_range = np.arange(-max_shift, max_shift + 1)

    optimal_shifts = np.full(num_windows, np.nan)

    for i in range(num_windows):
        x_seg = x_windows[i]
        min_mse = float("inf")
        best_shift = 0

        # For each possible shift, align y_seg accordingly
        for tau in shift_range:
            # Compute y indices for current shift
            idx = np.arange(window_size) + tau

            # Only use valid indices
            valid_mask = (idx >= 0) & (idx < window_size)
            if not np.any(valid_mask):
                continue  # No overlap for this shift

            y_seg = y_windows[i][idx[valid_mask]]
            x_seg_valid = x_seg[valid_mask]
            mse = np.mean((x_seg_valid - y_seg) ** 2)

            if mse < min_mse:
                min_mse = mse
                best_shift = tau

        optimal_shifts[i] = -best_shift  # Keep the same sign convention

    return sliding_window_pair.center_indices, optimal_shifts


def compute_pairwise_mse_shift(x: np.ndarray, y: np.ndarray, max_shift: int) -> int:
    """
    Compute the optimal integer shift between two equal-length signals using overlap-only MSE, consistent with
    NeuroTD local alignment principles.

    Parameters
    ----------
    x : np.ndarray
        Reference signal.
    y : np.ndarray
        Shifted signal.
    max_shift : int
        Maximum absolute shift (in samples) to consider.

    Returns
    -------
    best_shift : int
        Shift (in samples) minimizing the MSE.
    """
    N = len(x)
    assert len(y) == N

    min_mse = float("inf")
    best_shift = 0

    for tau in range(-max_shift, max_shift + 1):
        if tau < 0:
            x_seg = x[-tau:]
            y_seg = y[: N + tau]
        elif tau > 0:
            x_seg = x[: N - tau]
            y_seg = y[tau:]
        else:
            x_seg = x
            y_seg = y

        mse = np.mean((x_seg - y_seg) ** 2)

        if mse < min_mse:
            min_mse = mse
            best_shift = tau

    return best_shift


def compute_time_delays_from_spikes_neurotd(
    v: np.ndarray,
    spike_indices: np.ndarray,
    dt_ms: float,
    window_frac: float = 0.2,
    min_window: int = 8,
) -> np.ndarray:
    """
    Compute refined inter-spike time delays using NeuroTD pairwise MSE alignment
    between successive spikes.

    Parameters
    ----------
    v : np.ndarray
        Voltage trace.
    spike_indices : np.ndarray
        Spike indices (samples), sorted.
    dt_ms : float
        Sampling interval in milliseconds.
    window_frac : float, optional
        Fraction of the initial ISI used as half-window size.
    min_window : int, optional
        Minimum half-window size in samples.

    Returns
    -------
    time_delays_ms : np.ndarray
        Refined inter-spike intervals in milliseconds.
    """
    # spike_indices = np.asarray(spike_indices)
    # time_delays_ms = np.diff(spike_indices) * dt_ms if spike_indices.size >= 2 else np.array([], dtype=float)

    if len(spike_indices) < 2:
        return np.array([])

    N = len(v)
    time_delays_ms: list[float] = []

    for s0, s1 in zip(spike_indices[:-1], spike_indices[1:]):
        isi = int(s1 - s0)
        half_w = max(int(isi * window_frac), min_window)

        c0 = int(s0)
        c1 = int(s1)

        if c0 - half_w < 0 or c0 + half_w >= N or c1 - half_w < 0 or c1 + half_w >= N:
            time_delays_ms.append(isi * dt_ms)
            continue

        x = v[c0 - half_w : c0 + half_w + 1]
        y = v[c1 - half_w : c1 + half_w + 1]

        x = x / np.max(np.abs(x))
        y = y / np.max(np.abs(y))

        max_shift = max(half_w // 10, 5)
        shift = compute_pairwise_mse_shift(x, y, max_shift=max_shift)

        refined_isi = isi + shift
        time_delays_ms.append(refined_isi * dt_ms)

    return np.asarray(time_delays_ms)


class ShiftEstimationMethod(ABC):
    """Abstract base class for shift estimation methods."""

    def __init__(self, name):
        """
        name: The name of the method, such as "NeuroTD", "DTW", "CTW", "FFT", or "Ground Truth".
        indices: The indices of the base signal where the shifts are computed.
        shifts:  The estimated shifts as units of samples.
        """
        self.name = name
        self._shifts = None
        self._indices = None

    @abstractmethod
    def compute_shifts(self):
        """Each method must implement this to compute shifts."""
        pass

    @property
    def shifts(self):
        """Using properties ensures that shifts are computed only when first accessed.
        no need to call compute_shifts() explicitly.
        """
        if self._shifts is None:
            self.compute_shifts()
        return self._shifts

    @property
    def indices(self):
        if self._indices is None:
            self.compute_shifts()
        return self._indices

    def compute_mse(self, ground_truth):
        """Compute MSE with ground truth if applicable."""
        raise NotImplementedError("Not implemented yet.")
        # return np.mean((ground_truth - self.shifts) ** 2) if self.shifts is not None else None


class DTWShift(ShiftEstimationMethod):
    """Dynamic Time Warping (DTW) shift estimation method.
    package used: dtw
    may switch to tslearn.metrics.dtw_path in future
    """

    def __init__(self, sliding_window_pair, *args):
        super().__init__("DTW", *args)
        self.sliding_window_pair = sliding_window_pair

    def compute_shifts(self):
        shifts = np.zeros(self.sliding_window_pair.num_windows)

        for i in range(self.sliding_window_pair.num_windows):
            alignment = dtw(self.sliding_window_pair.y[i], self.sliding_window_pair.x[i])
            shifts[i] = (alignment.index1 - alignment.index2).mean()

        self._shifts = shifts
        self._indices = self.sliding_window_pair.center_indices


class CTWShift(ShiftEstimationMethod):
    """Canonical Time Warping (CTW) shift estimation method.
    package used: tslearn.metrics.ctw_path
    # DONE?: currently computes for each segment seperately, may compute for the whole signal
    """

    def __init__(self, sliding_window_pair, *args):
        super().__init__("CTW", *args)
        self.sliding_window_pair = sliding_window_pair

    def compute_shifts(self):
        shifts = np.zeros(self.sliding_window_pair.num_windows)

        # DONE?: currently computes for each segment seperately, may compute for the whole signal
        # path_ctw, cca, _ = ctw_path(sliding_window_pair.x, sliding_window_pair.y, max_iter=100)
        # path_ctw = np.array(path_ctw)
        # shifts_rec_ctw__per_point = path_ctw[:, 1] - path_ctw[:, 0]
        # plt.plot(sample_to_time_s(path_ctw[:, 1]), sample_to_time_ms(shifts_rec_ctw__per_point))

        for i in range(self.sliding_window_pair.num_windows):
            alignment_ctw, _, _ = ctw_path(self.sliding_window_pair.y[i], self.sliding_window_pair.x[i])
            alignment_ctw = np.array(alignment_ctw)
            shifts[i] = (alignment_ctw[1] - alignment_ctw[0]).mean()

        self._shifts = shifts
        self._indices = self.sliding_window_pair.center_indices


class MSEShift(ShiftEstimationMethod):
    """Mean Squared Error (MSE) shift estimation method."""

    def __init__(self, sliding_window_pair, *args):
        super().__init__("MSE", *args)
        self.sliding_window_pair = sliding_window_pair

    def compute_shifts(self):
        shifts = np.zeros(self.sliding_window_pair.num_windows)

        for i in range(self.sliding_window_pair.num_windows):
            shifts[i] = compute_best_shift_via_mse(self.sliding_window_pair.x[i], self.sliding_window_pair.y[i])

        self._shifts = shifts
        self._indices = self.sliding_window_pair.center_indices


class FFTShift(ShiftEstimationMethod):
    """Fast Fourier Transform (FFT) shift estimation method."""

    def __init__(self, sliding_window_pair, is_filter=False, *args):
        super().__init__("FFT", *args)
        self.sliding_window_pair = sliding_window_pair
        self.is_filter = is_filter
        # attrributes for the FFT method
        self._errors = None
        self._diffphase = None

    def compute_shifts(self):
        num_windows = self.sliding_window_pair.num_windows
        shifts = np.zeros(num_windows)
        errors = np.zeros(num_windows)
        diffphase = np.zeros(num_windows)

        for i in range(num_windows):
            x_seg, y_seg = self.sliding_window_pair.x[i], self.sliding_window_pair.y[i]
            if self.is_filter:
                # Define the cutoff frequency and filter order
                cutoff_freq = 200  # Adjust the cutoff frequency as needed
                filter_order = 4  # Adjust the filter order as needed
                # Apply a low-pass filter to x_seg
                b, a = signal.butter(filter_order, cutoff_freq, fs=seg_signal_noisy.sampling_rate, btype="low")
                x_seg = signal.filtfilt(b, a, x_seg)
                y_seg = signal.filtfilt(b, a, y_seg)

            cur_shift, errors[i], diffphase[i] = phase_cross_correlation(x_seg, y_seg, upsample_factor=10)
            cur_shift = -cur_shift[0]

            shifts[i] = cur_shift

        self._shifts = shifts
        self._indices = self.sliding_window_pair.center_indices
        self._errors = errors
        self._diffphase = diffphase


def compute_shifts_sliding_window_gt(center_indices, original_segments_center_idx, shifts):
    """compute the grouthtruth shifts for center_indices, given paras.shifts at original_segments_center_idx
    we find the closest original_segments_center_idx to each of the center_indices and use the corresponding
    shift

    Parameters
    ----------
    center_indices : _type_
        The center index of the sliding window
    original_segments_center_idx : _type_
        The center index of the original segments
    shifts : _type_
        ground truth shifts at original_segments_center_idx
    """
    shifts_sliding_window_gt = np.zeros_like(center_indices)
    for i, center in enumerate(center_indices):
        idx = np.argmin(np.abs(center - original_segments_center_idx))
        shifts_sliding_window_gt[i] = shifts[idx]
    return shifts_sliding_window_gt


def evaluate_alignment_quality_sliding(x, y, shifts, center_indices, window_size):
    """
    Compute the alignment RMSE of y aligned to x using per-window shift estimates.

    Parameters
    ----------
    x : np.ndarray
        Reference signal.
    y : np.ndarray
        Shifted signal to be aligned.
    shifts : np.ndarray
        Array of shifts (in samples), one per window.
    center_indices : np.ndarray
        Center indices of each window.
    window_size : int
        Size of the window used for alignment.

    Returns
    -------
    float
        RMSE computed over all windows where valid alignment is possible.
    np.ndarray
        Array of RMSE errors for each window.
    """
    half_w = window_size // 2
    errors = []

    center_indices = np.asarray(center_indices, dtype=int)  # Ensure indices are integers
    for center, shift in zip(center_indices, shifts):
        # Define window range
        x_start = center - half_w
        x_end = center + half_w + 1

        y_start = x_start + shift
        y_end = x_end + shift

        # Check if indices are valid: only consider windows that are fully within bounds
        if x_start >= 0 and x_end <= len(x) and y_start >= 0 and y_end <= len(y):
            x_win = x[x_start:x_end]
            y_win = y[y_start:y_end]
            mse = np.mean((x_win - y_win) ** 2)
            # errors.append(mse)  # Store mean squared error for this window
            #  symmetric normalized RMSE.
            normalizer = np.sqrt(np.mean(x_win**2) * np.mean(y_win**2)) + 1e-16  # 1e-16 Avoid division by zero
            nrmse = np.sqrt(mse) / normalizer  # Normalized RMSE
            errors.append(nrmse)  # Store normalized MSE for this window

    if not errors:
        return np.nan, np.nan
    # Return RMSE over all valid windows, and all errors for debugging
    # return np.sqrt(np.mean(errors)), np.sqrt(errors)
    return np.sqrt(np.mean(errors)), np.sqrt(errors)


def evaluate_alignment_quality_sliding_segments(sliding_window_pair, shifts):
    """
    NEW VERSION of evaluate_alignment_quality_sliding() may used in future after fix and verification
    Uses sliding window segments directly, boundary conditon and sign convension need to be checked,
    """
    errors = []
    num_windows, window_size = sliding_window_pair.num_windows, sliding_window_pair.window_size

    for i in range(num_windows):
        x_seg, y_seg = sliding_window_pair.x[i], sliding_window_pair.y[i]
        shift = shifts[i]

        # Integer shift for demonstration; may replace with interpolation for sub-sample shifts
        idx = np.arange(window_size) + int(shift)  # shift the indices of y_seg

        # Find indices that are valid (within bounds for y_seg)
        # DONE?: cross-segments indices, to use elements from y_seg that are not in the current segment
        valid_mask = (idx >= 0) & (idx < window_size)
        if np.any(valid_mask):
            x_seg_valid = x_seg[valid_mask]
            y_seg_aligned_valid = y_seg[idx[valid_mask]]
            mse = np.mean((x_seg_valid - y_seg_aligned_valid) ** 2)
            errors.append(mse)
        else:
            errors.append(np.nan)  # preserve window alignment

    errors = np.array(errors)
    return np.sqrt(np.nanmean(errors)), np.sqrt(errors)  # ignore NaNs in the mean calculation


def reconstruct_from_shifts(y, shifts, center_indices, window_size):
    """
    Reconstruct y based on local shifts per sliding window.

    Parameters
    ----------
    y : np.ndarray
        Original misaligned signal.
    shifts : np.ndarray
        Array of estimated shifts (in samples) for each window.
    center_indices : np.ndarray
        Indices in y where windows are centered.
    window_size : int
        Length of each sliding window (assumed odd).

    Returns
    -------
    y_reconstructed : np.ndarray
        Reconstructed signal by applying local shifts to y.
        Overlapping windows are averaged.
    """

    N = len(y)
    y_reconstructed = np.zeros(N)
    weight_sum = np.zeros(N)
    half_win = window_size // 2

    for shift, center in zip(shifts, center_indices):
        # Compute window bounds
        start = max(center - half_win, 0)
        end = min(center + half_win + 1, N)
        window_len = end - start

        # Extract the original windowed segment from y
        y_segment_indices = np.arange(start + int(shift), end + int(shift))
        y_segment_indices = np.clip(y_segment_indices, 0, N - 1)  # Handle boundaries

        y_segment = y[y_segment_indices]

        # Accumulate into reconstructed signal
        y_reconstructed[start:end] += y_segment[:window_len]
        weight_sum[start:end] += 1

    # Normalize by number of overlapping windows
    y_reconstructed = np.divide(y_reconstructed, weight_sum, where=weight_sum > 0)

    return y_reconstructed


def compare_signals(x, y, y_reconstructed, zoom_range=None):
    """
    Compare x (reference), y (misaligned), and y_reconstructed signals.

    Parameters
    ----------
    x : np.ndarray
        Reference signal.
    y : np.ndarray
        Misaligned signal.
    y_reconstructed : np.ndarray
        Reconstructed version of y aligned to x.
    zoom_range : tuple or None
        (start_idx, end_idx) to zoom into a segment, or None to show full length.
    """
    assert len(x) == len(y) == len(y_reconstructed), "All signals must be the same length."

    # Determine index range
    if zoom_range is not None:
        start, end = map(int, zoom_range)
        indices = np.arange(start, end)
    else:
        indices = np.arange(len(x))

    # Compute RMSE over selected region
    local_rmse = np.sqrt(np.mean((x[indices] - y_reconstructed[indices]) ** 2))

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(indices, x[indices], label="x (reference)", linewidth=2)
    plt.plot(indices, y[indices], label="y (misaligned)", alpha=0.5)
    plt.plot(indices, y_reconstructed[indices], label="y reconstructed", linestyle="--")
    plt.title(f"Comparison (RMSE={local_rmse:.2g})")
    plt.xlabel("Sample index")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.show()
