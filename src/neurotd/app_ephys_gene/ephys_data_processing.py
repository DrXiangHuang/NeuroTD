"""
required to run ephys_data_phase0_load_ephys_data_build_dict.py to build the raw ephys_dict from data files
"""

import gc
import hashlib
import pickle
import re
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sp_signal
from pynwb import NWBHDF5IO

from neurotd.app_ephys_gene.core import load_cell_name_to_cell_id, load_sample_maps, parse_vector_columns
from neurotd.app_ephys_gene.paths import DATA_PATH, DATA_PROCESSED_PATH, FIGURE_PATH, RESULT_PATH
from neurotd.app_ephys_gene.plotting import plot_all_sweeps_for_cell
from neurotd.time_varying_shift import compute_time_delays_from_spikes_neurotd

time_delay_file = DATA_PROCESSED_PATH / "time_delays_all_cells.csv"
CVS_MAP_PATH = DATA_PROCESSED_PATH / "cell_name_to_cell_id_and_sample_id.csv"
cell_name_to_cell_id = load_cell_name_to_cell_id(CVS_MAP_PATH)
sample_id_to_cell_id, sample_to_nwb = load_sample_maps(CVS_MAP_PATH)


def _normalize_step_indices(v: np.ndarray, t_on: int | None, t_off: int | None) -> tuple[int, int]:
    """
    Normalize step window indices.

    If t_on is None, uses 0 (start of trace).
    If t_off is None, uses len(v) (end of trace).

    Returns
    -------
    t_on_idx : int
    t_off_idx : int
    """
    n = len(v)
    t_on_idx = 0 if t_on is None else int(t_on)
    t_off_idx = n if t_off is None else int(t_off)
    # Clamp to [0, n]
    t_on_idx = max(0, min(t_on_idx, n))
    t_off_idx = max(0, min(t_off_idx, n))
    return t_on_idx, t_off_idx


def make_short_sweep_label(
    cell_id: str,
    sweep_id: int,
    mouse_prefix: str = "sub-mouse-",
    sample_prefix: str = "sample-",
) -> str:
    """
    Construct a short label 'mouse-sample-sweep' from a full NWB cell_id.

    Extraction logic:
        - mouse ID: immediately following 'mouse_prefix' until the next '_'
        - sample number: digits immediately following 'sample_prefix'

    Example:
        cell_id:
        sub-mouse-AAYYT_ses-20180420-sample-2_slice-2_cell-...nwb
        sweep_id = 15

        -> 'AAYYT-2-15'
    """
    # Build regex dynamically based on prefixes
    pattern = rf"{re.escape(mouse_prefix)}([A-Za-z0-9]+).*?" rf"{re.escape(sample_prefix)}(\d+)"

    m = re.search(pattern, cell_id)
    if m is None:
        return f"{cell_id}-{sweep_id}"

    mouse, sample = m.groups()
    return f"{mouse}-{sample}-{sweep_id}"


def detect_spikes(
    v: np.ndarray,
    t_on: int | None,
    t_off: int | None,
    dt_ms: float,
    thresh_std: float = 4.0,
    refrac_ms: float = 2.0,
    threshold_to_max: float = 0.6,
    num_max_peaks: int = 100,
    method: str = "baseline",
) -> tuple[np.ndarray, float, float, float]:
    """
    Detect action potentials in a single sweep.

    Parameters
    ----------
    v : np.ndarray
        One-dimensional voltage trace for a single sweep (shape: [T]).
    t_on : int or None
        Index of stimulus onset in samples; defines the baseline window.
    t_off : int or None
        Index of stimulus offset in samples; included for symmetry.
    dt_ms : float
        Sampling interval in milliseconds.
    thresh_std : float, optional
        Threshold in units of baseline standard deviations (baseline mode).
    refrac_ms : float, optional
        Absolute refractory period in milliseconds.
    threshold_to_max : float, optional
        Relative threshold fraction (adaptive mode).
    num_max_peaks : int, optional
        Upper bound on the number of detected peaks (adaptive mode).
    method : {"baseline", "fixed", "adaptive"}, optional
        Spike detection strategy, adaptive generally should be the best, though it limits to 100 spikes - never seen
        more thant that in our data.
    Returns
    -------
    spikes : np.ndarray
        Spike indices (samples).
    baseline_mean : float
        Baseline mean.
    baseline_std : float
        Baseline standard deviation.
    threshold : float
        Threshold used for detection.
    """
    t_on_idx, _ = _normalize_step_indices(v, t_on, t_off)

    if t_on_idx > 0:
        baseline = v[:t_on_idx]
    else:
        baseline = v

    baseline_mean = float(baseline.mean())
    baseline_std = float(baseline.std())

    refrac_samples = max(int(round(refrac_ms / dt_ms)), 1)

    if method == "baseline":
        threshold = baseline_mean + thresh_std * baseline_std

        above = v >= threshold
        crossings = np.where(np.logical_and(above[1:], ~above[:-1]))[0] + 1

        spikes: list[int] = []
        last = -np.inf
        for c in crossings:
            if c - last >= refrac_samples:
                spikes.append(int(c))
                last = c
    elif method == "fixed":
        threshold = baseline_mean + thresh_std * baseline_std
        distance = max(int(round(refrac_ms / dt_ms)), 1)
        spikes, _ = sp_signal.find_peaks(v, height=threshold, distance=distance)
    elif method == "adaptive":
        N = len(v)

        # use only halves that contain positive-going events (candidate spikes)
        half_maxs = [np.max(v[: N // 2]), np.max(v[N // 2 :])]
        positive_half_maxs = [x for x in half_maxs if x > 0]
        if not positive_half_maxs:
            raise ValueError(
                "Adaptive spike detection failed: max(v) < 0 in both halves of the sweep. "
                "No positive-going peaks detected; this sweep should not be used for ISTD."
            )
        half_max = min(positive_half_maxs)  # use the smaller of the two or one positive maxes

        half_min = max(np.min(v[: N // 2]), np.min(v[N // 2 :]))  # use the larger of the two min
        # only the difference between the max and min matters, so don't use peaks_threshold = half_max * threshold_to_max
        threshold = (half_max - half_min) * threshold_to_max + half_min
        distance = N // num_max_peaks
        spikes, _ = sp_signal.find_peaks(v, height=threshold, distance=distance)
    else:
        raise ValueError(f"Unknown method: {method}")

    spikes = np.asarray(spikes, dtype=int)
    return spikes, baseline_mean, baseline_std, threshold


def find_peaks_adaptive(v, threshold_to_max=0.6, num_max_peaks=100):
    """_summary_
    currently only handle maxium of around 100 peaks, to be improved
    Parameters
    ----------
    v : np.ndarray
        One-dimensional voltage trace (shape: [T]).
    threshold_to_max : float, optional
        _description_, by default 0.6
    num_max_peaks : int, optional
        _description_, by default 100
    """
    N = len(v)
    # use only halves that contain positive-going events (candidate spikes)
    half_maxs = [np.max(v[: N // 2]), np.max(v[N // 2 :])]
    positive_half_maxs = [x for x in half_maxs if x > 0]
    if not positive_half_maxs:
        raise ValueError(
            "Adaptive spike detection failed: max(v) < 0 in both halves of the sweep. "
            "No positive-going peaks detected; this sweep should not be used for ISTD."
        )
    half_max = min(positive_half_maxs)  # use the smaller of the two or one positive maxes
    half_min = max(np.min(v[: N // 2]), np.min(v[N // 2 :]))  # use the larger of the two min
    # only the difference between the max and min matters, so don't use peaks_threshold = half_max * threshold_to_max
    peaks_threshold = (half_max - half_min) * threshold_to_max + half_min

    peaks_idx, _ = sp_signal.find_peaks(v, height=peaks_threshold, distance=N // num_max_peaks)
    return peaks_idx


def score_sweep_spiking(
    v: np.ndarray,
    t_on: int | None,
    t_off: int | None,
    dt_ms: float,
    n_min: int = 4,
    r_step: float = 0.8,
    isi_min_ms: float = 5.0,
    amp_min: float = 10.0,
    cv_max: float = 0.3,
    thresh_std: float = 4.0,
    refrac_ms: float = 2.0,
) -> tuple[bool, dict[str, object]]:
    """
    Evaluate whether a single sweep exhibits spiking suitable for ISTD analysis and return summary quality metrics.

    If t_on or t_off are None, the step window is taken as the full sweep (t_on = 0, t_off = len(v)). In that case, the
    fraction-of-step criterion  reduces to n_step / n_total = 1.0, so r_step only has an effect if it is less than 1.

    Parameters
    ----------
    v : np.ndarray
        One-dimensional voltage trace (shape: [T]).
    t_on : int or None
        Stimulus onset index in samples, or None to use 0.
    t_off : int or None
        Stimulus offset index in samples, or None to use len(v).
    dt_ms : float
        Sampling interval in milliseconds.
    n_min, r_step, isi_min_ms, amp_min, cv_max, thresh_std, refrac_ms :
        See previous documentation.

    Returns
    -------
    is_good : bool
        True if the sweep passes all criteria; False otherwise.
    info : dict
        QC metrics (spike counts, amplitudes, ISIs, baseline, threshold).
    """
    t_on_idx, t_off_idx = _normalize_step_indices(v, t_on, t_off)

    spikes, baseline_mean, baseline_std, threshold = detect_spikes(
        v=v,
        t_on=t_on_idx,
        t_off=t_off_idx,
        dt_ms=dt_ms,
        thresh_std=thresh_std,
        refrac_ms=refrac_ms,
    )

    n_total = int(len(spikes))
    in_step = spikes[(spikes >= t_on_idx) & (spikes < t_off_idx)]
    n_step = int(len(in_step))
    frac_step = n_step / n_total if n_total > 0 else 0.0

    info: dict[str, object] = {
        "n_total": n_total,
        "n_step": n_step,
        "frac_step": frac_step,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "threshold": threshold,
        "spikes": spikes,
        "mean_amp": 0.0,
        "cv_amp": np.inf,
        "median_isi_ms": None,
    }

    if n_total == 0:
        return False, info
    if n_step < n_min:
        return False, info
    if frac_step < r_step:
        return False, info

    amps = v[in_step] - baseline_mean
    mean_amp = float(amps.mean())
    cv_amp = float(amps.std() / mean_amp) if mean_amp > 0 else np.inf
    info["mean_amp"] = mean_amp
    info["cv_amp"] = cv_amp

    if mean_amp < amp_min or cv_amp > cv_max:
        if n_step >= 2:
            isi_ms = np.diff(in_step) * dt_ms
            info["median_isi_ms"] = float(np.median(isi_ms))
        return False, info

    if n_step >= 2:
        isi_ms = np.diff(in_step) * dt_ms
        median_isi_ms = float(np.median(isi_ms))
        info["median_isi_ms"] = median_isi_ms
        if median_isi_ms < isi_min_ms:
            return False, info

    return True, info


def filter_response_by_spiking(
    response_dict: dict[tuple[str, int], np.ndarray],
    t_on: int | None,
    t_off: int | None,
    dt_ms: float,
    n_min: int = 4,
    r_step: float = 0.8,
    isi_min_ms: float = 5.0,
    amp_min: float = 10.0,
    cv_max: float = 0.3,
    thresh_std: float = 4.0,
    refrac_ms: float = 2.0,
) -> tuple[dict[tuple[str, int], np.ndarray], dict[tuple[str, int], dict[str, object]]]:
    """
    Apply spiking-based quality control to all sweeps in a response dictionary.

    Sweeps with max(trace) < 0 are rejected explicitly, as they contain no positive-going events and cannot produce valid spike times for ISTD.

    t_on and t_off may be None, in which case the full trace is treated as
    the step window. All other parameters are passed through to
    'score_sweep_spiking'.
    """
    response_dict_qc: dict[tuple[str, int], np.ndarray] = {}
    qc_info: dict[tuple[str, int], dict[str, object]] = {}

    for key, trace in response_dict.items():
        if np.max(trace) < 0:
            qc_info[key] = {
                "reason": "max(trace) < 0; no positive-going events in sweep",
            }
            continue

        is_good, info = score_sweep_spiking(
            v=trace,
            t_on=t_on,
            t_off=t_off,
            dt_ms=dt_ms,
            n_min=n_min,
            r_step=r_step,
            isi_min_ms=isi_min_ms,
            amp_min=amp_min,
            cv_max=cv_max,
            thresh_std=thresh_std,
            refrac_ms=refrac_ms,
        )
        qc_info[key] = info
        if is_good:
            response_dict_qc[key] = trace

    return response_dict_qc, qc_info


# %%
def get_cell_responses(response_dict: dict, cell_name: str) -> dict[int, np.ndarray]:
    """
    Collect response traces for a given cell into a dict[sweep_id -> trace].
    """
    sweeps: dict[int, np.ndarray] = {}
    for (c_name, sweep_id), trace in response_dict.items():
        if c_name == cell_name:
            sweeps[sweep_id] = trace
    return sweeps


def get_cell_stimuli_from_catalog(stimulus_catalog: dict, cell_name: str) -> dict[int, np.ndarray]:
    """
    Collect stimulus waveforms for a given cell into a dict[sweep_id -> trace].

    Uses
    ----
    stimulus_catalog["waveforms"] : dict[waveform_id -> waveform]
    stimulus_catalog["sweep_to_waveform"] : dict[(cell_id, sweep_id) -> waveform_id]
    """
    waveforms = stimulus_catalog["waveforms"]
    sweep_to_waveform = stimulus_catalog["sweep_to_waveform"]

    sweeps: dict[int, np.ndarray] = {}
    for (c_name, sweep_id), wf_id in sweep_to_waveform.items():
        if c_name == cell_name:
            sweeps[sweep_id] = waveforms[wf_id]
    return sweeps


def check_stimulus_amplitude(
    stimulus_wave,
    low_bound: float = 1e-1,
    up_bound: float = np.inf,
    t_on: int | None = None,
    t_off: int | None = None,
) -> bool:
    """
    Return True if max(stimulus) in the given window is within [low_bound, up_bound].
    If t_on/t_off are None, the full waveform is used.
    """
    stim = np.asarray(stimulus_wave)
    if t_on is not None or t_off is not None:
        stim = stim[slice(t_on, t_off)]
    if stim.size == 0:
        return False
    mx = float(stim.max())
    return (mx >= low_bound) and (mx <= up_bound)


def filter_by_stimulus(
    stimulus_catalog: dict,
    response_dict: dict,
    low_bound: float = 1e-1,
    up_bound: float = np.inf,
    t_on: int | None = None,
    t_off: int | None = None,
) -> tuple[dict, dict]:
    """
    Filter sweeps based on stimulus amplitude and return both filtered
    stimulus_catalog and filtered response_dict.

    Keeps sweeps whose max stimulus in [t_on, t_off] is within [low_bound, up_bound].
    If t_on/t_off are not provided, uses the full-length stimulus.

    Assumes
    -------
    stimulus_catalog["waveforms"] : dict[waveform_id -> 1D array-like stimulus]
    stimulus_catalog["sweep_to_waveform"] : dict[(cell_id, sweep_id) -> waveform_id]
    stimulus_catalog["waveform_to_sweeps"] : dict[waveform_id -> list[(cell_id, sweep_id)]]
    response_dict[(cell_id, sweep_id)] : 1D array-like response trace

    Returns
    -------
    stimulus_catalog_filtered : dict
        Filtered catalog with keys "waveforms", "sweep_to_waveform", "waveform_to_sweeps".
    response_dict_filtered : dict
        Filtered response dictionary with only kept (cell_id, sweep_id) keys.
    """
    waveforms = stimulus_catalog["waveforms"]
    sweep_to_waveform = stimulus_catalog["sweep_to_waveform"]

    keep_keys: list[tuple[str, int]] = []
    for (cell_id, sweep_id), wf_id in sweep_to_waveform.items():
        stim_wave = waveforms[wf_id]
        if check_stimulus_amplitude(
            stim_wave,
            low_bound=low_bound,
            up_bound=up_bound,
            t_on=t_on,
            t_off=t_off,
        ):
            keep_keys.append((cell_id, sweep_id))

    keep_keys_set = set(keep_keys)

    # Filter response_dict
    response_dict_filtered = {key: response_dict[key] for key in keep_keys if key in response_dict}

    # Filter stimulus_catalog pieces
    waveforms_filtered: dict = {}
    sweep_to_waveform_filtered: dict = {}
    waveform_to_sweeps_filtered: dict = {}

    for (cell_id, sweep_id), wf_id in sweep_to_waveform.items():
        if (cell_id, sweep_id) not in keep_keys_set:
            continue

        sweep_to_waveform_filtered[(cell_id, sweep_id)] = wf_id

        if wf_id not in waveforms_filtered:
            waveforms_filtered[wf_id] = waveforms[wf_id]

        waveform_to_sweeps_filtered.setdefault(wf_id, []).append((cell_id, sweep_id))

    stimulus_catalog_filtered = {
        "waveforms": waveforms_filtered,
        "sweep_to_waveform": sweep_to_waveform_filtered,
        "waveform_to_sweeps": waveform_to_sweeps_filtered,
    }

    return stimulus_catalog_filtered, response_dict_filtered


def extract_time_delays_for_sweep(
    v: np.ndarray,
    dt_ms: float,
    t_on: int | None = None,
    t_off: int | None = None,
    thresh_std: float = 4.0,
    refrac_ms: float = 2.0,
    threshold_to_max: float = 0.6,
    num_max_peaks: int = 100,
    method: str = "baseline",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect spikes in a sweep and compute inter-spike intervals during the stimulus step window.

    Returns
    -------
    spike_indices_step : np.ndarray
        Spike indices (in samples) within the step window.
    time_delays_ms : np.ndarray
        Inter-spike intervals in milliseconds. May be empty.
    """
    spikes, _, _, _ = detect_spikes(
        v=v,
        t_on=t_on,
        t_off=t_off,
        dt_ms=dt_ms,
        thresh_std=thresh_std,
        refrac_ms=refrac_ms,
        threshold_to_max=threshold_to_max,
        num_max_peaks=num_max_peaks,
        method=method,
    )
    t_on_idx, t_off_idx = _normalize_step_indices(v, t_on, t_off)
    spike_indices_step = spikes[(spikes >= t_on_idx) & (spikes < t_off_idx)]
    time_delays_ms = compute_time_delays_from_spikes_neurotd(v, spike_indices_step, dt_ms)
    return spike_indices_step, time_delays_ms


def build_time_delay_table(
    response_dict_ISTD: dict[tuple[str, int], np.ndarray],
    dt_ms: float,
    t_on: int | None = None,
    t_off: int | None = None,
    thresh_std: float = 4.0,
    refrac_ms: float = 2.0,
    threshold_to_max: float = 0.6,
    num_max_peaks: int = 100,
    method: str = "baseline",
) -> pd.DataFrame:
    """
    Build a time-delay table for all ISTD-qualified sweeps.

    Each row corresponds to one sweep and contains:
        - 'peak_idx'        : np.ndarray of spike indices within the step
        - 'time_delays(ms)' : np.ndarray of inter-spike intervals in ms


    response_dict_ISTD : dict[(cell_id, sweep_id) -> np.ndarray] of GOOD sweeps
    dt_ms : sampling interval in ms (for 25kHz: 0.04)
    t_on, t_off : indices of stimulus step (or None to use whole sweep)

    The DataFrame index is a short label 'mouse-sample-sweep', such as
    'AAYYT-2-15'.
    """
    records: list[dict[str, object]] = []
    index_labels: list[str] = []

    # deterministic order: sort by cell_id, then sweep_id
    for cell_id, sweep_id in sorted(response_dict_ISTD.keys()):
        trace = response_dict_ISTD[(cell_id, sweep_id)]
        peak_idx, time_delays_ms = extract_time_delays_for_sweep(
            v=trace,
            dt_ms=dt_ms,
            t_on=t_on,
            t_off=t_off,
            thresh_std=thresh_std,
            refrac_ms=refrac_ms,
            threshold_to_max=threshold_to_max,
            num_max_peaks=num_max_peaks,
            method=method,
        )

        # Option: include even sweeps with <2 spikes; here we keep them with empty delays.
        short_label = make_short_sweep_label(cell_id, sweep_id)
        index_labels.append(short_label)
        records.append(
            {
                "peak_idx": peak_idx,
                "time_delays(ms)": time_delays_ms if len(time_delays_ms) > 0 else np.nan,
            }
        )

    df_time_delays = pd.DataFrame(records, index=index_labels)
    return df_time_delays


# %%
ephys_dict_path = DATA_PATH / "raw" / "raw_data_ephys_dict"
stimulus_pkl = ephys_dict_path / "stimulus_catalog.pkl"  # 27 MB
response_pkl = ephys_dict_path / "response_dict.pkl"  # 20 GB, large file

response_filtered_pkl = ephys_dict_path / "response_dict_with_positive_stimulus.pkl"


with stimulus_pkl.open("rb") as f:
    stimulus_catalog = pickle.load(f)
with response_pkl.open("rb") as f:
    response_dict = pickle.load(f)

# %% Sweep# 104409 (Orignal) -> 90271 (filtered by amplitude) -> 10639 (filtered by spiking quality without max>0)
# -> 10637 (filtered by spiking quality with max>0)
print("Original # of sweeps:", len(response_dict))
stimulus_catalog_filtered, response_dict_filtered = filter_by_stimulus(
    stimulus_catalog,
    response_dict,
    low_bound=1e-1,
    up_bound=np.inf,
    t_on=None,
    t_off=None,
)
print("Filtered # of sweeps by stimulus amplitude:", len(response_dict_filtered))


# Time parameters for ISTD QC
dt_ms = 0.04  # 0.04ms, 25kHz # 1000.0 * (time[1] - time[0])  # time is the 1D array of timestamps
t_on = None
t_off = None

# Second-stage filtering: keep only sweeps with good spiking, as most of sweeps have bad stimulus and no good spiking
response_dict_ISTD, qc_info = filter_response_by_spiking(
    response_dict=response_dict_filtered,
    t_on=t_on,
    t_off=t_off,
    dt_ms=dt_ms,
    n_min=4,
    r_step=0.8,
    isi_min_ms=5.0,
    amp_min=10.0,
    cv_max=0.3,
)

print("Filtered # of sweeps by spiking quality:", len(response_dict_ISTD))  # 10639

# %% visualize some examples
cell_name = "sub-mouse-AAYYT_ses-20180420-sample-2_slice-20180420-slice-2_cell-20180420-sample-2_icephys"  # 1st cell, normal 80 sweeps
cell_name = "sub-mouse-YAVWS_ses-20191114-sample-9_slice-20191114-slice-8_cell-20191114-sample-9_icephys"  # 99 sweeps


resp_sweeps = get_cell_responses(response_dict, cell_name)
plot_all_sweeps_for_cell(resp_sweeps, cell_name=cell_name, kind="response")

resp_sweeps_filt = get_cell_responses(response_dict_filtered, cell_name)
plot_all_sweeps_for_cell(resp_sweeps_filt, cell_name=cell_name, kind="response (filtered)")

stim_sweeps = get_cell_stimuli_from_catalog(stimulus_catalog, cell_name)
plot_all_sweeps_for_cell(stim_sweeps, cell_name=cell_name, kind="stimulus")

stim_sweeps_filt = get_cell_stimuli_from_catalog(stimulus_catalog_filtered, cell_name)
plot_all_sweeps_for_cell(stim_sweeps_filt, cell_name=cell_name, kind="stimulus (filtered)")

resp_sweeps_ISTD = get_cell_responses(response_dict_ISTD, cell_name)
fig = plot_all_sweeps_for_cell(resp_sweeps_ISTD, cell_name=cell_name, kind="response (final used for ISTD)")

# %% save the figure for all 1328 -> 1295 cell_ids (if file system crash, then run half each time, total 20 mins)
# ephys_dict_path = "C:\Dropbox\DaifengWangLab\NeuroTD\src\neurotd\app_ephys_gene\data\raw_data_ephys_ISTD_sweeps_figs"
save_all_figures = False
if save_all_figures:
    ephys_dict_path = Path(
        "/mnt/c/Dropbox/DaifengWangLab/NeuroTD/src/neurotd/app_ephys_gene/data/raw_data_ephys_ISTD_sweeps_figs"
    )
    cell_name_list = sorted(list(set([key[0] for key in response_dict_ISTD.keys()])))
    for cell_name in cell_name_list:  # for cell_name in cell_name_list[798:]:
        resp_sweeps_ISTD = get_cell_responses(response_dict_ISTD, cell_name)
        fig = plot_all_sweeps_for_cell(resp_sweeps_ISTD, cell_name=cell_name, kind="response (final used for ISTD)")
        fig.savefig(ephys_dict_path / f"{cell_name}__ISTD_sweeps.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
# %% refined time_dealys extraction for resp_sweeps_ISTD
# Example configuration
time_delay_file = DATA_PROCESSED_PATH / "time_delays_all_cells.csv"
dt_ms = 0.04  # 25 kHz
t_on = None  # t_on = stim_on_index   # or None if not defined yet
t_off = None  # t_off = stim_off_index # or None

df_time_delays_all = build_time_delay_table(
    response_dict_ISTD=response_dict_ISTD,
    dt_ms=dt_ms,
    t_on=t_on,
    t_off=t_off,
    method="adaptive",
)

save_time_delay_table = True
if save_time_delay_table:
    df_time_delays_all.to_csv(time_delay_file)

# total spikes: 206323 -> 191073 for adaptive, 185865 for baseline
df_time_delays_all["peak_idx"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0).sum()

# with open("df_time_delays_all_20260115_adaptive_2_afterNeuroTDRefine.pkl", "rb") as f:
#     df = pickle.load(f)
# %%

# %%  Verify saved time delays for a specific cell
td = pd.read_csv(time_delay_file, index_col=0)  # 10639 rows: sweeps, columns: 'peak_idx', 'time_delays(ms)'
td_pd = parse_vector_columns(td, vec_cols=["peak_idx", "time_delays(ms)"])  # same as df_time_delays_all
tdvs = td_pd["time_delays(ms)"].values

cell_name = "sub-mouse-AAYYT_ses-20180420-sample-2_slice-20180420-slice-2_cell-20180420-sample-2_icephys"  # 1st cell, normal 80 sweeps
cell_name = "sub-mouse-ZZVQT_ses-20181003-sample-9_slice-20181003-slice-9_cell-20181003-sample-9_icephys"  # no problem for all
cell_name = "sub-mouse-ZZVQT_ses-20181003-sample-10_slice-20181003-slice-10_cell-20181003-sample-10_icephys"
cell_id = cell_name_to_cell_id[cell_name]

resp_sweeps_ISTD = get_cell_responses(response_dict_ISTD, cell_name)
fig = plot_all_sweeps_for_cell(resp_sweeps_ISTD, cell_name=cell_name, kind="response (final used for ISTD)")
df_time_delays_all.loc[df_time_delays_all.index.str.startswith("AAYYT-2-")]  # sweep_id from 0 to 79

sweep_ids = list(sorted(resp_sweeps_ISTD.keys()))

# plot the sweep along with the 'peak idx'
fig, axs = plt.subplots(figsize=(12, 6), nrows=(len(sweep_ids) + 1) // 2, ncols=2, sharex=True)
for sweep_id, ax in zip(sweep_ids, axs.flatten()):
    debug = False
    if debug:
        sweep_id = 15
        fig, ax = plt.subplots()
    short_label = make_short_sweep_label(cell_id, sweep_id)
    peak_idx = td_pd.loc[short_label, "peak_idx"]
    v = trace = resp_sweeps_ISTD[sweep_id]
    time_axis = np.arange(len(trace)) * dt_ms
    ax.plot(time_axis, trace, label="Voltage Trace")
    ax.scatter(np.array(peak_idx) * dt_ms, trace[peak_idx], color="red", label="Detected Spikes")
    ax.set_title(f"Sweep {sweep_id} for Cell {cell_id}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    ax.legend()
    if debug:
        break
plt.show()

# %%
