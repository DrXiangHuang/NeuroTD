import time
from collections.abc import Sequence
from functools import wraps
from typing import Union

import numpy as np
import numpy.typing as npt
from numpy import ndarray
from scipy.fft import irfft, rfft
from sklearn.metrics import mean_squared_error


def measure_time(func):
    """Decorator to measure the execution time of a function.
    # Usage example:
    @measure_time
    def myfun():
        pass
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function '{func.__name__}' ran in {end - start:.2f} seconds")
        return result

    return wrapper


def ensure_odd_window_size(window_size):
    """Ensure window size is odd for integer center alignment."""
    if not isinstance(window_size, (list, tuple, np.ndarray)):
        # single integer
        return window_size if window_size % 2 == 1 else window_size + 1
    return np.array([ensure_odd_window_size(ws) for ws in window_size])


def create_sliding_window_view(x, window_size, hop_size):
    """Create a readonly sliding window view of a 1D array x. Returns a 2D array of shape (num_windows, window_size).
    It is similar to np.lib.stride_tricks.sliding_window_view which is the same as this function with hop_size=1
    x_sliding_window[i, :] = x[i * hop_size : i * hop_size + window_size] is the ith window

    Parameters:
    ----------
    x (np.ndarray): 1D array to create sliding windows from.
    window_size (int): Number of samples per segment. TODO: force it to be odd number for now, so the center of the
        window is at the middle sample.
    hop_size (int): Number of samples between windows.

    Returns:
    np.ndarray: Readable sliding windows view of x with shape (num_windows, window_size).

    Example:
    --------
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    windows = create_sliding_window_view(x, window_size=3, hop_size=2)
    print(windows)
    # Output:
    [[1 2 3]
    [3 4 5]
    [5 6 7]]
    See also:
    ---------
    np.lib.stride_tricks.sliding_window_view funciton which only create hop_size=1 sliding windows
    https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
    """
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
    if window_size > len(x):
        raise ValueError("window_size must be less than or equal to the length of x")
    # odd window_size for now, so the center_index is integer
    if window_size % 2 == 0:
        Warning("window_size must be an odd integer for now, so the center_index is integer")

    # The ith window is x[i * hop_size : i * hop_size + window_size] for i in 0, .., num_windows - 1
    # The last window must satisfy: (num_windows - 1) * hop_size + window_size <= len(x)
    num_windows = (len(x) - window_size) // hop_size + 1

    x_sliding_window = np.lib.stride_tricks.as_strided(
        x, shape=(num_windows, window_size), strides=(x.strides[0] * hop_size, x.strides[0]), writeable=False
    )
    return x_sliding_window


class SlidingWindowPair:
    """Create sliding window views for a pair of 1D signals (arrays), x and y.

    For each signal, create a read-only 2D array of shape (num_windows, window_size).
    Additionally, store the center indices for each sliding window and the total number of windows.

    Parameters:
    ----------
    x (np.ndarray): 1D array for the first signal.
    y (np.ndarray): 1D array for the second signal.
    window_size (int): Number of samples per segment.
    hop_size (int): Number of samples between windows.

    Attributes:
    ----------
    x (np.ndarray): Read-only sliding windows view of x with shape (num_windows, window_size).
    y (np.ndarray): Read-only sliding windows view of y with shape (num_windows, window_size).
    num_windows (int): Total number of sliding windows.
    center_indices (np.ndarray): Center indices (could be non-integer) for each sliding window.

    See also:
    ---------
    create_sliding_window_view: For implementation details of the sliding window creation.
    """

    def __init__(self, x, y, window_size, hop_size):
        if len(x) != len(y):
            raise ValueError("Signals x and y must have the same length")

        self.window_size = window_size
        self.hop_size = hop_size

        self.x = create_sliding_window_view(x, window_size, hop_size)
        self.y = create_sliding_window_view(y, window_size, hop_size)

        self.num_windows = self.x.shape[0]
        #  use fractional index such as 0.5 which is more accurate than integer indices, along with InterpolatedVector
        self.center_indices = (window_size - 1) / 2 + hop_size * np.arange(self.num_windows)

    def __repr__(self):
        return (
            f"SlidingWindowPair(window_size={self.window_size}, hop_size={self.hop_size}, "
            f"num_windows={self.num_windows}, x.shape={self.x.shape}, y.shape={self.y.shape})"
        )


class InterpolatedVector:
    """Wraps a numeric 1D array, allowing linear interpolation at fractional indices.

    Supports indexing with integers or floats. Does not support slicing.

    Parameters:
    -----------
    vector : array-like
        A numeric 1D array used for interpolation.

    Example:
    --------
    >>> vec = InterpolatedVector([10, 20, 30, 40])
    >>> vec[1]
    20
    >>> vec[1.5]
    25.0
    >>> vec[2.25]
    32.5
    """

    def __init__(self, vector):
        self.vector = np.asarray(vector)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            raise TypeError("Slicing is not supported for InterpolatedVector.")

        idx_arr = np.atleast_1d(idx)
        idx_floor, idx_ceil = np.floor(idx_arr).astype(int), np.ceil(idx_arr).astype(int)
        if np.any(idx_floor < 0) or np.any(idx_ceil >= len(self.vector)):
            raise IndexError("Index out of bounds for interpolation.")

        # Perform linear interpolation
        frac = idx_arr - idx_floor
        interp_vals = self.vector[idx_floor] + frac * (self.vector[idx_ceil] - self.vector[idx_floor])

        return interp_vals if np.ndim(idx) else interp_vals.item()

    def __repr__(self):
        return f"InterpolatedVector({self.vector})"


def generate_circular_shifted_sequences(x: npt.ArrayLike, shifts: Sequence[int]) -> np.ndarray:
    """Circular shift  1D sequence ``x`` to the right (roll) by ``shifts'' amounts.

    Return result X_shifted, where its ith row is x shifted by t = shifts[i],
        x_shifted[n] = x[n-t]

    Parameters
    ----------
    x : npt.ArrayLike, shape = (N_time_samples, )
        1D sequence
    shifts : Sequence[int], shape = (N_shift, )
        1D iterable shifts

    Returns
    -------
    np.ndarray, shape = (N_shift, N_time_samples)
        2D array of shifted sequences. Its row k is ``seq`` circular shifted (roll) to the right by shifts[k]


    See also
    --------
    np.roll
        Roll array elements along a given axis.
    """
    x = np.asarray(x)
    X_shifted = np.zeros_like(x, shape=(len(shifts), len(x)))
    for idx, shift in enumerate(shifts):
        X_shifted[idx] = np.roll(x, shift)
    return X_shifted


def add_noise(Y0: np.ndarray, noise_to_max_signal_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """add random Gaussian noise to an array

    Parameters
    ----------
    rng: np.random._generator.Generator
    Y0 : np.ndarray
        original array
    noise_to_max_signal_ratio : float, optional (default = 0.1)
        0.01, 0.02, 0.05, 0.10, Corrsponding PSNR = 20 * log10(1 / ratio) = [40, 34, 26, 20] dB

    Returns
    -------
    np.ndarray
        noisy array
    """
    noise = rng.normal(0, 1, size=Y0.shape) * (Y0.max() * noise_to_max_signal_ratio)
    Y_noisy = Y0 + noise
    return Y_noisy


def compute_phase_shift_base_vector(N: int, keep_half_freq: bool = True, t: int = 1) -> np.ndarray:
    """Compute the phase shift for each frequency when a 1D signal is shifted in time by t sample.

    Fourier shift theorem states:
        DFT(x[n-t]) = e^{-j * delta * t} * DFT(x[n]),
    where DFT(x[n]) is the vector of DFT, delta = 2*pi/N [0, 1, ..., N-1], and * is element-wise multiplication.

    Parameters
    ----------
    N : int
        Number of samples
    keep_half_freq : bool, opitional with default as True
        If keep_half_freq is True, we only keep the first N//2 + 1 frequency components (similar to rfft and irfft)
        This is when the input signal is real and the DFT of real input is Hermitian-symmetric.
        If keep_half_freq is False, the input signal is complex and we need to keep the full N frequency compments
    t : int
        shift amount in number of samples
    Returns
    -------
    np.ndarray
        1D array (vector) of the ``base_phase_shift``

    Notes
    -----
    This function is simple yet important enough to written as a function with comments.
    """
    N_keep = N // 2 + 1 if keep_half_freq else N
    delta = (2 * np.pi / N) * np.arange(N_keep)
    return delta if t == 1 else delta * t


def circular_shift(arr: npt.ArrayLike, dt: Union[int, float], N: int = None) -> np.ndarray:
    """Shift an real-valued array to the right by dt elements (dt could by a fractional amount) using FFT.
    N: default length of FFT, if None, use len(arr). Not sure if need to use N = 2 * len(arr) or longer to avoid aliasing.
    Need to check for fractional t, if t is fractional, may need to use longer N to avoid aliasing.
    Need to check for zero padding
    """
    N = len(arr) if N is None else N

    # Compute the FFT of the array - assume the array is real-valued, so use rfft to save computation time, keep half the frequencies
    fft_arr = rfft(arr, n=N, norm="ortho")

    """ If arr is complexed values, may need something like this:
    # Compute the FFT of the array
    fft_arr = np.fft.fft(arr, norm='ortho')
    # # Compute the frequency values
    freq = np.fft.fftfreq(len(arr))
    # # Shift the frequency values
    freq_shifted = np.fft.fftshift(freq)
    # Compute the phase shift
    phase_shift = np.exp(2j * np.pi * freq_shifted * t)
    """

    # 2 * np.pi * np.arange(N) / N, truncated to np.ceil(N / 2)
    delta = compute_phase_shift_base_vector(N, keep_half_freq=True)
    phase_shift = np.exp(-1j * dt * delta)

    # Apply the phase shift to the FFT of the array
    fft_arr_shifted = fft_arr * phase_shift

    # Compute the inverse FFT of the shifted array
    arr_shifted = irfft(fft_arr_shifted, n=N, norm="ortho")

    return arr_shifted  #


def rmse(REF: np.ndarray, X: np.ndarray) -> np.float64:
    """
    Compute root mean square error between array X with a reference array REF of the same shape.

    Parameters:
        REF (np.ndarray): The reference array.
        X (np.ndarray): The array to compare with the reference array.

    Returns:
        np.float64: The root mean square error between the two arrays.
    """
    rmse_ = np.sqrt(mean_squared_error(REF, X))
    return rmse_
