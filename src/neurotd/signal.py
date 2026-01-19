import copy
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np

from neurotd.utils import circular_shift, create_sliding_window_view


# ============================
# Built-in waveform generators
# ============================
def cos_Gaussian(L: float = 40 * np.pi, N: int = 1024, w: float = 1) -> tuple[np.ndarray, np.ndarray]:
    """N points of a cosine times Gaussian function of length L from [-L/2, L/2)"""
    dt = L / N
    t = np.arange(-L / 2, L / 2, dt)
    x = np.cos(w * t) * np.exp(-(t**2) / L)  # Function
    return t, x


def my_ricker(N, a=None):
    """
    Return a Ricker wavelet, also known as the "Mexican hat wavelet".

    It models the function:
        A * (1 - (x/a)**2) * exp(-0.5*(x/a)**2)

    Parameters
    ----------
    N : int
        Number of points in `vector`. Will be centered around 0.
    a : float, optional
        Width parameter of the wavelet. Default is N // 16.

    Returns
    -------
    np.ndarray
        Array of length N in shape of ricker curve.

    Examples
    --------
    >>> N = 100
    >>> a = 4.0
    >>> vec2 = signal.ricker(N, a)
    >>> print(len(vec2))  # 100
    >>> plt.plot(vec2)
    >>> plt.show()

    """
    if a is None:
        a = N // 16
    vec = np.arange(N) - (N - 1.0) / 2
    factor = 1 - (vec**2) / a**2
    envelope = np.exp(-(vec**2) / (2 * a**2))  # * A
    return factor * envelope


def three_sin(x):
    return np.sin(x) + np.sin(2 * x) + np.sin(3 * x)


# Note: We intentionally do NOT use @dataclass before Signal and SegmentedSignal for the following reasons:
#
# 1. Signal requires a custom __init__ to handle both 'data' and 'sampling_rate' explicitly.
#    Using @dataclass would auto-generate an __init__ that conflicts with this logic.
#
# 2. SegmentedSignal inherits from Signal and must call super().__init__() with generated data.
#    Using @dataclass with inheritance often leads to field ordering issues or TypeErrors
#    (e.g., "non-default argument follows default argument") especially when mixing fields
#    across parent and child.
#
# 3. We also avoid @dataclass(init=False) tricks for 'data' because it silently breaks subclassing
#    expectations and test-time introspection (e.g., in pytest, IDEs, or auto-docs).
#
# Conclusion: Manually writing __init__ gives us full control, ensures compatibility with cautious mode,
# and avoids surprises caused by dataclass auto-generation in inheritance chains.
class Signal:
    """
    A base signal class holding time-series data and a sampling rate.

    Parameters
    ----------
    data : np.ndarray
        1D time-series data.
    sampling_rate : float
        Sampling rate in Hz.

    Properties
    ----------
    n_samples : int
        Total number of samples in the signal.
    time : np.ndarray
        Evenly spaced time points from 0 to (duration - 1/sampling_rate).
    duration : float
        Duration of the signal in seconds.

    Methods
    -------
    copy()
        Return a deep copy of this Signal.
    plot(title="Signal")
        Plot the time-series data.

    Examples
    --------
    >>> data = np.sin(np.linspace(0, 2*np.pi, 1000))
    >>> sig = Signal(data=data, sampling_rate=1000)
    >>> print(sig.duration)  # 1.0
    >>> sig.plot()
    """

    def __init__(self, data: np.ndarray, sampling_rate: float):
        self.data = data
        self.sampling_rate = sampling_rate

    @property
    def n_samples(self) -> int:
        return self.data.shape[-1]

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.n_samples) / self.sampling_rate

    @property
    def duration(self) -> float:
        return self.n_samples / self.sampling_rate

    def copy(self) -> "Signal":
        return copy.deepcopy(self)

    def plot(self, title="Signal"):
        plt.plot(self.time, self.data)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class SegmentedSignal(Signal):
    """
    A segmented signal class that constructs a time-series by concatenating waveform segments.

    Parameters
    ----------
    duration : float
        The intended total duration of the signal in seconds. Used to determine how many samples to generate.
    sampling_rate : float
        Sampling rate in Hz.
    n_segments : int
        Number of segments to divide the signal into.
    shape_func : Callable[[int], np.ndarray], optional
        Function that generates a segment of a given length (default: `my_ricker`).

    Inherits from
    --------------
    Signal
        Provides:
        - data : np.ndarray — full signal array
        - sampling_rate : float
        - n_samples : int
        - duration : float (computed)
        - time : np.ndarray
        - plot(), copy()

    Additional Properties
    ---------------------
    segments : list of np.ndarray
        List of waveform segments.
    iter_segments : generator of np.ndarray
        Iterator over segments (memory efficient).
    n_samples_per_seg : int
        Number of samples in each segment.
    duration_per_seg : float
        Duration of each segment in seconds.
    time_of_segment : list of np.ndarray
        Time values corresponding to each segment.

    Methods
    -------
    _get_segment_slice(i)
        Return a slice object for indexing the i-th segment.
    generate_shifted(shifts)
        Return a new SegmentedSignal with segments circularly shifted.
    generate_noisy(rng, noise_std)
        Return a new SegmentedSignal with added Gaussian noise.
    copy()
        Return a deep copy of the object.
    __eq__(other)
        Check equality with another SegmentedSignal.

    Examples
    --------
    >>> sig = SegmentedSignal(duration=1.0, sampling_rate=5120, n_segments=10)
    >>> print(sig.segments[0].shape)
    >>> print(sig.duration)  # computed from n_samples
    >>> for seg in sig.iter_segments:
    ...     print(seg.mean())
    >>> sig.plot()
    """

    def __init__(
        self,
        duration: float,
        sampling_rate: float,
        n_segments: int,
        shape_func: Callable[[int], np.ndarray] = my_ricker,
    ):
        self.sampling_rate = sampling_rate
        self.n_segments = n_segments
        self.shape_func = shape_func

        total_samples = round(duration * sampling_rate)

        if not np.isclose(total_samples, duration * sampling_rate):
            warnings.warn(
                f"duration × sampling_rate = {duration * sampling_rate}, not integer. " f"Rounded to {total_samples}."
            )

        if total_samples % n_segments != 0:
            warnings.warn(
                f"n_samples = {total_samples} is not divisible by n_segments = {n_segments}. "
                f"Truncating to {total_samples // n_segments * n_segments} samples."
            )

        samples_per_seg = total_samples // n_segments
        segments = [self.shape_func(samples_per_seg) for _ in range(n_segments)]
        data = np.concatenate(segments)

        super().__init__(data=data, sampling_rate=sampling_rate)

    def _get_segment_slice(self, i: int) -> slice:
        """Return the slice object for the i-th segment."""
        return slice(i * self.n_samples_per_seg, (i + 1) * self.n_samples_per_seg)

    @property
    def n_samples_per_seg(self) -> int:
        return self.n_samples // self.n_segments

    @property
    def duration_per_seg(self) -> float:
        return self.duration / self.n_segments

    @property
    def segments(self) -> list[np.ndarray]:
        """Return a list of individual segments."""
        return [self.data[self._get_segment_slice(i)] for i in range(self.n_segments)]

    @property
    def iter_segments(self):
        """Return an iterator over the segments."""
        return (self.segments[i] for i in range(self.n_segments))

    @property
    def time_of_segment(self) -> list[np.ndarray]:
        """Return the time array for each segment."""
        return [self.time[self._get_segment_slice(i)] for i in range(self.n_segments)]

    def generate_shifted(self, shifts: Iterable[int]) -> "SegmentedSignal":
        shifted = self.copy()
        shifted.data = np.hstack([circular_shift(seg, dt) for seg, dt in zip(self.segments, shifts)])
        return shifted

    def generate_noisy(self, rng: np.random.Generator, noise_std: float) -> "SegmentedSignal":
        noisy = self.copy()
        noise = rng.normal(scale=noise_std, size=self.n_samples)
        noisy.data = self.data + noise
        return noisy

    def copy(self) -> "SegmentedSignal":
        return copy.deepcopy(self)

    def __eq__(self, other: "SegmentedSignal") -> bool:
        return (
            np.array_equal(self.data, other.data)
            and self.sampling_rate == other.sampling_rate
            and self.n_segments == other.n_segments
            and self.shape_func == other.shape_func
        )
