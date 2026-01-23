"""
Fig. 3. Time delays of neural electrical activities associated with 3D positions across the human
medial temporal lobe

The (large) raw dataset should be downloaded from DANDI:
https://dandiarchive.org/dandiset/000574

Parallel computation of NeuroTD is used across all trials and regions for iEEG microelectrode data.
This parallel execution may be sensitive to the computing environment and package paths; disable it if errors occur.
"""
# %%
import os
import pickle
import sys
from itertools import combinations, product
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynwb
import scipy.stats
import seaborn as sns
import statsmodels.stats.multitest as smm
from joblib import Parallel, delayed
from scipy.stats import ttest_ind

from neurotd.time_varying_shift import (
    compute_windowed_mse_with_hop,
    evaluate_alignment_quality_sliding,
    sample_to_time_ms,
)
from neurotd.utils import InterpolatedVector, SlidingWindowPair, ensure_odd_window_size, measure_time, rmse

# below only Required for running parallel using joblib in a remote machine: makes the neurotd package visible for all child processes spawned by joblib.
sys.path.insert(0, "/mnt/c/Dropbox/DaifengWangLab/NeuroTD/src")
from neurotd.hiEEG_functions import detect_segments as detect_trials
from neurotd.hiEEG_functions import load_iEEG_micro

# StyleO
sns.set_theme(context='talk', style='white', palette='Set2')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.ion()  # enable interactive mode for command-line script execution (non-blocking figure display)

# %% functions
def compute_distance(electrode_positions, idx1, idx2):
    """Compute distance in mm between two electrode indices."""
    pos1, pos2 = electrode_positions[idx1], electrode_positions[idx2]
    return np.linalg.norm(pos1 - pos2)


def build_trial_dataframes(trials, electrode_labels):
    """Convert waveform trials into a list of DataFrames with time index.
    Example usage:
        dfs_per_trial = build_trial_dataframes(trials, micro_data['electrodes'])
    """
    dfs_per_trial = []
    for i in range(len(trials["time"])):
        df = pd.DataFrame(
            trials["waveform"][i],
            index=pd.Series(trials["time"][i], name='Time (s)'),
            columns=electrode_labels,
        )
        dfs_per_trial.append(df)
    return dfs_per_trial


def run_neurotd_for_pair(x, y, window_size, hop_size, fs=32000):
    """Compute shifts (ms) and quality metrics for a single signal pair of x, y."""
    center_idx, shifts = compute_windowed_mse_with_hop(x, y, window_size, hop_size)
    quality_rmse, quality_per_window = evaluate_alignment_quality_sliding(x, y, shifts, center_idx, window_size)
    shifts_ms = sample_to_time_ms(shifts, fs)  # Convert samples to ms
    return shifts_ms, quality_rmse, quality_per_window


def process_region_trial(df_trial, indices, window_size, hop_size=None, fs=32000, debug=False, verbose=False):
    """Processes all unique pairs of electrodes specified by `indices` within a single trial.
    It computes pairwise metrics (e.g., shifts, quality) for every unique electrode pair among the given indices.
    Typically, `indices` corresponds to all electrodes in a specific brain region, but may also be any custom list
    (e.g., across regions or a selected subset).

    Parameters
    ----------
    df_trial : pd.DataFrame
        DataFrame containing the signal data for all electrodes in this trial (columns = electrodes).
    indices : list of int
        List of electrode indices to process. May represent a region or any subset.
    window_size : int
        Size of the sliding window for NeuroTD.
    hop_size : int
        Step size between windows.  default is window_size // 2
    fs : int, optional
        Sampling frequency (Hz), default is 32000.
    debug : bool, optional
        If True, only process the first 2 electrodes for speed/testing.
    verbose : bool, optional
        If True, print progress.

    Returns
    -------
    results : list of dict
        Each dict corresponds to one electrode pair and contains:
            - 'pair': tuple (index_i, index_j)
            - 'dist': float, distance between electrodes i and j
            - 'shifts_ms': np.ndarray, inferred time-varying shifts (ms) per window
            - 'quality_rmse': float, global alignment RMSE for this pair
            - 'quality_rmse_per_window': np.ndarray, per-window RMSE values

        The list length is # of pairs, such as C(n, 2), where n is the number of indices processed (number of unique pairs).

    Notes
    -----
    - If debug=True, only first 2 electrodes are used.
    - If window_size > signal length, that pair is skipped.
    """
    if hop_size is None:
        hop_size = window_size // 2

    results = []

    if debug:
        N_electrodes = min(2, len(indices))  # For debugging, only use first two electrodes
    else:
        N_electrodes = len(indices)

    # Loop over all unique, unordered electrode pairs (i < j)
    for i in range(N_electrodes - 1):  # full version of all pairs
        # for i in range(1):  # For debugging, set i in [0] to process only the first electrode as i
        # for j in [i + 1]:  # For further debugging, set j in [i+1] to process only one pair
        for j in range(i + 1, N_electrodes):
            if verbose:
                print(f"Processing pair: ({indices[i]}, {indices[j]})")
            x = df_trial.iloc[:, indices[i]].values
            y = df_trial.iloc[:, indices[j]].values
            if len(x) < window_size:
                continue
            shifts_ms, quality_rmse, quality_per_window = run_neurotd_for_pair(x, y, window_size, hop_size, fs)
            dist = compute_distance(micro_data['electrode_positions'], indices[i], indices[j])
            results.append(
                {
                    'pair': (indices[i], indices[j]),
                    'dist': dist,
                    'shifts_ms': shifts_ms,
                    'quality_rmse': quality_rmse,
                    'quality_rmse_per_window': quality_per_window,
                }
            )
    return results


def process_one_trial_for_region(region, indices, df_trial, window_size, hop_size=None, fs=32000):
    """
       Processes one trial of neural data for a specific brain region.

       This function is designed for parallel execution. It allows efficient, modular analysis of
       all region–trial combinations by wrapping the region-level processing for a single trial.
       Typically used together with joblib.Parallel or similar tools to speed up large-scale analysis.

       Parameters
       ----------
       region : str
           Name of the brain region.
       indices : list of int
           Electrode indices corresponding to the region.
       df_trial : pandas.DataFrame
           DataFrame for the current trial (rows: time, columns: electrode signals).
       window_size : int
           Length of sliding window (in samples).
       hop_size : int
           Step size between windows (in samples). default is window_size // 2
       fs : int
           Sampling frequency (Hz).

       Returns
       -------
       list of dict
           List of result dicts (e.g., one per electrode pair) with computed shift, quality, etc.

    Notes
       -----
       This wrapper is used for clean parallelization; its input signature matches the requirements
       of tools such as joblib.delayed, enabling parallel execution across trials and regions.
    """
    if hop_size is None:
        hop_size = window_size // 2
    return process_region_trial(df_trial, indices, window_size, hop_size, fs, debug=False)


def process_one_trial(trial_idx, df_trial, region_to_index, window_size, hop_size=None, fs=32000, n_trials=None):
    """
    Processes all brain regions for a single trial of neural data.

    Parameters
    ----------
    trial_idx : int
        Index of the current trial.
    df_trial : pandas.DataFrame
        DataFrame for the current trial (rows: time, columns: electrode signals).
    region_to_index : dict
        Mapping from region name to list of electrode indices.
    window_size : int
        Length of the sliding window (in samples).
    hop_size : int, optional
        Step size between windows (in samples). If None, defaults to window_size // 2.
    fs : int
        Sampling frequency (Hz).
    n_trials : int, optional
        Total number of trials (for printing progress).

    Returns
    -------
    list of dict
        List of result dictionaries (each containing region, trial, pair, shift, quality, etc.).
    """
    if hop_size is None:
        hop_size = window_size // 2
    trial_results = []
    for region, indices in region_to_index.items():
        if len(indices) < 2:
            continue
        if n_trials is not None:
            print(f"Processing trial {trial_idx + 1}/{n_trials} for region {region}")
        else:
            print(f"Processing trial {trial_idx} for region {region}")
        pair_results = process_region_trial(df_trial, indices, window_size, hop_size, fs, debug=False)
        for pair_result in pair_results:
            trial_results.append({'region': region, 'trial': trial_idx, **pair_result})
    return trial_results


def run_parallel_trials(dfs_per_trial, region_to_index, window_size=1001, hop_size=None, fs=32000, n_jobs=8):
    """
    Runs parallel analysis of all trials, processing all regions in each trial,
    and returns a flattened DataFrame of results.
    Divide the jobs among each trial.

    Parameters
    ----------
    dfs_per_trial : list of pd.DataFrame
        List of trial DataFrames.
    region_to_index : dict
        Mapping from region name to list of electrode indices.
    window_size : int
        Sliding window size (default: 1001).
    hop_size : int, optional
        Hop size between windows (default: window_size // 2).
    fs : int
        Sampling frequency (Hz).
    n_jobs : int
        Number of parallel jobs. For maximized performance, set it to the number of physical cores (not the threads).

    Returns
    -------
    pd.DataFrame
        DataFrame with all pairwise results across all trials and regions.

    Example Usage
    --------------
    @measure_time
    def run_and_flatten_all_trials():
        return run_parallel_trials(
            dfs_per_trial, region_to_index, window_size=1001, hop_size=None, fs=32000, n_jobs=8
        )

    df_results = run_and_flatten_all_trials()
    """
    n_trials = len(dfs_per_trial)
    # max_nbytes=None to disable memmapping to avoid joblib filling /dev/shm (joblib’s default memmap spot).
    # but this may increase memory use and cause error
    results = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        # results = Parallel(n_jobs=n_jobs)(
        delayed(process_one_trial)(trial_idx, df_trial, region_to_index, window_size, hop_size, fs, n_trials)
        for trial_idx, df_trial in enumerate(dfs_per_trial)
    )
    # Flatten results
    rows = [row for trial_res in results for row in trial_res]
    df_results = pd.DataFrame(rows)
    return df_results


@measure_time
def analyze_dataset(region_to_index, dfs_per_trial, micro_data, window_size, fs=32000, debug=False):
    """
    Returns: {region: [list of trials, each with list of pair dicts]}
    all_results = {
    "regionA": [
        [  # trial 0
            {'pair': (i, j), ...},  # first pair in trial 0
            {'pair': (i, j), ...},  # second pair in trial 0
            ...
        ],
        [  # trial 1
            {'pair': (i, j), ...},
            ...
        ],
        ...
    ],
    "regionB": [
        ...
    ]
    }
    """
    window_size = ensure_odd_window_size(window_size)
    hop_size = window_size // 2
    all_results = {}
    n_trials = len(dfs_per_trial)

    # Limit to first two regions for debugging
    region_items = list(region_to_index.items())
    if debug:
        region_items = region_items[:2]  # Only first two regions

    for region, indices in region_items:
        print(f"Region: {region}, Indices: {indices}")
        if len(indices) < 2:
            continue

        region_results = []
        for i_trial, df_trial in enumerate(dfs_per_trial):
            print(f"Processing trial {i_trial + 1}/{n_trials} for region {region}")
            trial_results = process_region_trial(df_trial, indices, window_size, hop_size, fs, debug=debug)
            region_results.append(trial_results)
        all_results[region] = region_results

    return all_results


def flatten_all_results(all_results):
    """Flatten the nested all_results dictionary into a DataFrame.
    region	trial	pair	dist	shifts_ms	quality_rmse	quality_rmse_per_window
    Amygdala	0	(8, 9)	5.835659336	"[ 0,...]    0.446988666  [0.40323034, ...]"
    """
    rows = []
    for region, region_trials in all_results.items():
        for trial_idx, trial in enumerate(region_trials):
            for pair_result in trial:
                row = {'region': region, 'trial': trial_idx, **pair_result}  # expands keys: pair, dist, shifts_ms, ...
                rows.append(row)
    return pd.DataFrame(rows)


#  Plot the results
def plot_region_shifts_with_stats(shifts_ms_per_region, ax=None):
    """
    Generate violin plots and perform pairwise t-tests with Bonferroni correction
    for time-varying shifts across brain regions.

    Parameters
    ----------
    shifts_ms_per_region : dict
        Dictionary mapping region names to arrays of shift values.
    """
    # Create DataFrame
    data_final_subtle = []
    regions = list(shifts_ms_per_region.keys())
    for region in regions:
        data_final_subtle.extend([(region, val) for val in shifts_ms_per_region[region]])

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
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    # , the default is density_norm='area', meaning all violins have the same area, regardless of how many points each group has.
    sns.violinplot(data=df_final_subtle, x="Region", y="Delay (ms)", palette="Blues", inner="box", cut=0, ax=ax)
    # Each violin has the same maximum width, regardless of sample size.
    # sns.violinplot(data=df_final_subtle, density_norm='width', x="Region", y="Delay (ms)", palette="Blues",
    # inner="box", cut=0, ax=ax)
    # The width of the violin is proportional to the number of data points in that group.
    # sns.violinplot(data=df_final_subtle, density_norm='count', x="Region", y="Delay (ms)", palette="Blues", inner="box", cut=0, ax=ax)

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
    return fig


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
    fig, ax = plt.subplots(figsize=(8, 6))
    # sns.kdeplot(x=distance, y=delay, fill=True, cmap="Oranges", levels=100, alpha=0.7)
    ax.scatter(distance, delay, color="black", s=5, alpha=0.4)  # Add raw scatter points

    # Add regression line
    # sns.regplot(x=distance, y=delay, scatter=False, color="blue", line_kws={"linewidth": 2})

    # Labels and title
    plt.xlabel("Distance (mm)", fontsize=12)
    plt.ylabel("Delay (ms)", fontsize=12)
    ax.set_title("Density scatter plot (KDE) with regression line", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)
    return fig


def compute_normalized_shifts_per_region(df_results, min_shift_threshold=1e-8):
    """
    Compute normalized absolute nonzero shifts per region.

    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame containing columns 'region', 'shifts_ms', and 'dist'.
    min_shift_threshold : float, optional
        Minimum threshold for considering a shift as nonzero.

    Returns
    -------
    dict
        Dictionary mapping region names to arrays of normalized shift values.

    Example
    -------
    >>> df_results = pd.DataFrame({
            'region': ['A', 'A', 'B', 'B'],
            'shifts_ms': [np.array([0.1, 0.2]), np.array([0.3]), np.array([0.05, 0.15]), np.array([0.2])],
            'dist': [10, 10, 5, 5]
        })
    >>> min_shift_threshold = 1e-8
    >>> compute_normalized_shifts_per_region(df_results, min_shift_threshold)
    {
        'A': np.array([0.01, 0.02, 0.03]),
        'B': np.array([0.01, 0.03])
    }
    """
    shifts_ms_per_region_abs_nonzero_norm = {}
    for region in df_results["region"].unique():
        sub_df = df_results[df_results["region"] == region]
        delays = []
        for _, row in sub_df.iterrows():
            shift_arr = np.abs(row["shifts_ms"])
            dist = row["dist"]
            mask = shift_arr > min_shift_threshold
            if np.any(mask):
                normalized_shifts = shift_arr[mask] / dist if dist != 0 else np.zeros(np.sum(mask))
                delays.extend(normalized_shifts)
        if len(delays) > 0:
            shifts_ms_per_region_abs_nonzero_norm[region] = np.array(delays)
    return shifts_ms_per_region_abs_nonzero_norm


# %%
computer = 'R13'  # 'R12' or 'R13'
format = 'new'  # 'new' or 'old', should never use 'old' after June 2025
is_parallel = False  # Set to True to run in parallel (be careful - may easily crashes), False for single cell version
# iEEG Microelectrode Data (32kHz)
# first subject, first trial has 12_800_000 time samples (400s * 32kHz) at time <0, 572> with empty times
# Total about 100 of these files, total 1_280_000_000 time samples (40000s * 32kHz)
# Above just for one electrode, total 32/64/40/62 electrodes, so about 40 billion samples in total
if computer == 'R12':
    folder = "/mnt/d/DaifengWangLab/data/dandiset_000574_20250610Ver/000574"  # WSL R12
elif computer == 'R13':
    folder = "/mnt/c/DaifengWangLab/data/dandiset_000574_20250610Ver/000574"  # WSL R13
n_jobs = n_cpu_cores = 8 if computer == 'R12' else 12  # 'R12' i9-11900KF 'R13' i7-12700KF
subjects = [1]
sessions = [1, 2, 3]
subject_sessions = [(s, ses) for s in subjects for ses in sessions]  # R12

# %% 2025.10.21_extract the character size (4/6/8) for the size
future_work = False
if future_work:
    subject = 1
    session = 1
    format_types = ('old', 'new')
    assert format in format_types, f'Unknown format type "{format}", expected one of {format_types}'

    # Formatting
    file_string = f'sub-{subject:02d}/'
    file_string = os.path.join(file_string, os.listdir(os.path.join(folder, file_string))[session - 1])

    # Load file
    nwbfile = pynwb.NWBHDF5IO(os.path.join(folder, file_string), mode='r').read()
    micro_data = nwbfile.acquisition['ecephys.lfp'].data[:]
    micro_time = nwbfile.acquisition['ecephys.lfp'].timestamps[:]
    micro_electrodes = nwbfile.acquisition['ecephys.lfp'].electrodes[:]
    micro_trials = nwbfile.trials[:]
    nwbfile.units[:]['spike_times']  # has spike_times for each of 37 id?
    nwbfile.units[:]['electrode_group']  # has electrode_group for each of 37 id?

    units = nwbfile.units.to_dataframe()
    print(units.iloc[0]['spike_times'][:10])
# %%
for subject, session in subject_sessions:
    print(f'Processing subject {subject}, session {session}')
    file_prefix = f"subject_{subject}_session_{session}_"
    micro_data, micro_meta = load_iEEG_micro(subject=subject, session=session, folder=folder, format=format)

    # % subject=1, session=1 has 50 trials  of 8 seconds each, 32kHz sampling rate
    trials = detect_trials(micro_data['time'], micro_data['waveform'])  #
    # There are 50 segments, 8.0s each (256000 timepoints).
    print(
        f'There are {len(trials["time"])} trials, {trials["time"][0].shape[0] / 32e3:.1f}s each ({trials["time"][0].shape[0]} timepoints).'
    )
    # trials["time"].shape  # (50, 256_000)  # 50 segments, each segment has 256_000 timepoints (8s at 32kHz)
    # trials["waveform"].shape  # (50, 256_000, 32) # 50 segments, each segment has 256_000 timepoints, 32 electrodes

    # %
    # micro_data['electrode_major_regions']
    # micro_data['time']  # (12_800_000, 1)  plt.plot(micro_data['time']) shows jumps instead a line
    # micro_data['waveform']  # (12_800_000, 32)
    # micro_data['electrode_positions']  # (32, 3)
    # idx = range(int(17 * 32_000), int(22 * 32_000))  # first 10 seconds
    # idx = slice(None)
    # df = pd.DataFrame(
    #     micro_data['waveform'][idx],
    #     index=pd.Series(micro_data['time'][idx], name='Time (s)'),
    #     columns=micro_data['electrodes'],
    # )

    dfs_per_trial = build_trial_dataframes(trials, micro_data['electrodes'])

    # % find unique regions, sort them, and create a mapping for each region where is the index
    # This is for the 32 electrodes, each electrode has a major region
    # The regions are not unique, some electrodes have the same region
    # We want to find the unique regions and sort them, then create a mapping for each region
    # The regions are in the form of ['Amygdala', 'Hippocampus', 'Inferior Temporal Gyrus',
    # 'Insular Gyrus', 'Middle Temporal Gyrus', 'Superior Temporal Gyrus']， remove the 'unspecific' region
    regions = micro_data['electrode_major_regions']
    unique_regions = np.unique(regions)
    unique_regions = unique_regions[unique_regions != 'unspecific']
    #  create a mapping for each region to its index
    region_to_index = {}
    for region in unique_regions:
        # 1D array of indices need to use [0], otherwise it will return a tuple
        region_to_index[region] = np.where(micro_data['electrode_major_regions'] == region)[0]

    # % Find window size with small subset of data for testing: [101, 501, 1001, 2001] where 1001 is good
    # debug
    # all_results = analyze_dataset(
    #     region_to_index, dfs_per_trial, micro_data, window_size=1001, debug=True
    # )
    if is_parallel:
        @measure_time
        def run_and_flatten_all_trials():
            return run_parallel_trials(
                dfs_per_trial, region_to_index, window_size=1001, hop_size=None, fs=32000, n_jobs=n_jobs
            )

        # Function 'run_and_flatten_all_trials' ran in 1107.84 seconds for 50 trials * 17 pairs
        # Function 'run_and_flatten_all_trials' ran in 3376.32 seconds for 50 trials * 60 pairs

        df_results = run_and_flatten_all_trials()
    else:
        # # %% single cell version
        all_results = analyze_dataset(region_to_index, dfs_per_trial, micro_data, window_size=1001, debug=False)
        df_results = flatten_all_results(all_results)

    # % Save the results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    use_time_stamp = False
    df_ret_filename = f"{file_prefix}_neurotd_df_results"
    if use_time_stamp:
        df_ret_filename = f"{df_ret_filename}_{timestamp}"
    df_results.to_csv(f"{df_ret_filename}.csv", index=False)

    is_save_pickle = False
    if is_save_pickle:
        pickle_filename = f"{df_ret_filename}.pkl"
        with open(pickle_filename, 'wb') as f:
            pickle.dump(df_results, f)
    # with open('neurotd_all_results__2025.08.06_1stElectroVsRest.pkl', 'rb') as f:
    #    df_results = pickle.load(f)

    # % Analyze all regions
    # Suppose df_results has columns: region, trial, pair, dist, shifts_ms, quality_rmse, ...

    # Get region list (preserving order if desired)
    regions = df_results["region"].unique().tolist()

    # Build the required dict for plotting
    shifts_ms_per_region = {
        region: np.concatenate(df_results[df_results["region"] == region]["shifts_ms"].values) for region in regions
    }

    min_shift_threshold = 1e-8  # adjust threshold if needed to exclude near-zero shifts

    shifts_ms_per_region_abs_nonzero = {
        region: np.array([abs(val) for val in delays if abs(val) > min_shift_threshold])
        for region, delays in shifts_ms_per_region.items()
        if np.any(np.abs(delays) > min_shift_threshold)  # only keep region if any nonzero
    }
    # plot_region_shifts_with_stats(shifts_ms_per_region)
    fig = plot_region_shifts_with_stats(shifts_ms_per_region_abs_nonzero)
    fig.savefig(f"{file_prefix}_shifts_ms_per_region_abs_nonzero_{timestamp}.png", dpi=300, bbox_inches="tight")

    future_work = False
    if future_work:
        # (I) normalized delay
        shifts_ms_per_region_abs_nonzero_normalized = compute_normalized_shifts_per_region(
            df_results, min_shift_threshold=1e-8
        )
        fig = plot_region_shifts_with_stats(shifts_ms_per_region_abs_nonzero_normalized)
        fig.savefig(
            f"{file_prefix}_shifts_ms_per_region_abs_nonzero_normalized_{timestamp}.png", dpi=300, bbox_inches="tight"
        )

        # (II) distance vs delays
        all_delays = np.concatenate(df_results["shifts_ms"].values)
        all_distances = np.repeat(df_results["dist"].values, [len(x) for x in df_results["shifts_ms"].values])
        # Take absolute value and filter nonzero
        abs_delays = np.abs(all_delays)
        nonzero_mask = abs_delays > min_shift_threshold

        # Take absolute value and filter nonzero
        abs_delays = np.abs(all_delays)
        nonzero_mask = abs_delays > min_shift_threshold

        filtered_delays = abs_delays[nonzero_mask]
        filtered_distances = all_distances[nonzero_mask]

        fig = plot_density_scatter_with_regression(filtered_distances, filtered_delays)
        fig.savefig(f"{file_prefix}_density_scatter_distance_delay_{timestamp}.png", dpi=300, bbox_inches="tight")


# %% load computed results for multiple sessions, and show each session separately for Supplementary Figure S1
def str_to_array(s):
    """Remove stray whitespace and convert string to list, then to np.array for shifts_ms"""
    # Remove brackets and newlines, then parse numbers, otherwise, df_results["shifts_ms"][0] is str like
    #  '[ 0.       0.       0.       0.       0.       0.       0.       0.\n  0.
    s_clean = s.replace('\n', ' ').replace('[', '').replace(']', '')
    arr = np.fromstring(s_clean, sep=' ')
    return arr

# data_folder = Path('/mnt/c/Dropbox/DaifengWangLab/NeuroTD/results').resolve()
data_folder = None  # set to None to use current folder
file_suffix = 'neurotd_df_results.csv'
file_names = [
    'subject_1_session_1__neurotd_df_results.csv',
    'subject_1_session_2__neurotd_df_results.csv',
    'subject_1_session_3__neurotd_df_results.csv',
]
subjects = [1]
sessions = [1, 2, 3]
subject_sessions = [(s, ses) for s in subjects for ses in sessions]  # R12
dfs = []
for subject, session in subject_sessions:
    print(f'Processing subject {subject}, session {session}')
    file_prefix = f"subject_{subject}_session_{session}_"
    file_name = f"{file_prefix}_{file_suffix}"
    file_path = data_folder / file_name if data_folder else file_name
    print(file_path)
    #
    df_results = pd.read_csv(file_path, converters={"shifts_ms": str_to_array})
    dfs.append(df_results)

    #  Analyze all regions
    # Suppose df_results has columns: region, trial, pair, dist, shifts_ms, quality_rmse, ...

    # Get region list (preserving order if desired)
    regions = df_results["region"].unique().tolist()

    # Build the required dict for plotting
    shifts_ms_per_region = {
        region: np.concatenate(df_results[df_results["region"] == region]["shifts_ms"].values) for region in regions
    }

    min_shift_threshold = 1e-8  # adjust threshold if needed to exclude near-zero shifts

    shifts_ms_per_region_abs_nonzero = {
        region: np.array([abs(val) for val in delays if abs(val) > min_shift_threshold])
        for region, delays in shifts_ms_per_region.items()
        if np.any(np.abs(delays) > min_shift_threshold)  # only keep region if any nonzero
    }
    # plot_region_shifts_with_stats(shifts_ms_per_region)
    fig = plot_region_shifts_with_stats(shifts_ms_per_region_abs_nonzero)
    fig.savefig(f"{file_prefix}_shifts_ms_per_region_abs_nonzero.png", dpi=300, bbox_inches="tight")

# %% Consolidate all sessions for Figure 4c
df_all = pd.concat(dfs, ignore_index=True)
regions = df_all["region"].unique().tolist()

shifts_ms_per_region = {
    region: np.concatenate(df_all[df_all["region"] == region]["shifts_ms"].values) for region in regions
}

min_shift_threshold = 1e-8

shifts_ms_per_region_abs_nonzero = {
    region: np.array([abs(val) for val in delays if abs(val) > min_shift_threshold])
    for region, delays in shifts_ms_per_region.items()
    if np.any(np.abs(delays) > min_shift_threshold)
}

fig = plot_region_shifts_with_stats(shifts_ms_per_region_abs_nonzero)
fig.savefig("ALL_subjects_sessions_shifts_ms_per_region_abs_nonzero.png", dpi=300, bbox_inches="tight")


plt.show(block=True)  # keep the figures open for scripts run outside of an interactive environment
# %%
