import gc
import hashlib
import pickle
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pynwb import NWBHDF5IO

from neurotd.app_ephys_gene.paths import DATA_PATH

###############################################################################
# Optional fast hash backend
###############################################################################
try:
    import xxhash

    _HAVE_XXHASH = True
except ImportError:
    xxhash = None
    _HAVE_XXHASH = False


# %%
###############################################################################
# Helpers
###############################################################################
def extract_sweeps_from_nwb_group(group, prefix: str):
    """
    Extract sweeps from an NWB group (stimulus or acquisition) whose keys
    start with a given prefix, using string operations only.

    Returns
    -------
    sweeps_by_idx : dict[int, np.ndarray]
        Mapping from sweep index (int) to data array.
    """
    sweeps_by_idx = {}

    for name, ts in group.items():
        if not name.startswith(prefix):
            continue

        suffix = name[len(prefix) :]  # e.g. '009'
        if not suffix.isdigit():
            print(f"Skip unexpected key in NWB group: '{name}'")
            continue

        idx = int(suffix)
        sweeps_by_idx[idx] = np.array(ts.data)

    return sweeps_by_idx


def check_indices(indices, label: str, expected_count: int | None = None):
    """
    Check integer sweep indices for missing values.

    Parameters
    ----------
    indices : Iterable[int]
        Observed indices.
    label : str
        Label for printed messages.
    expected_count : int or None
        If not None, expected indices are range(0, expected_count).
        If None, expected indices are inferred as contiguous [min, max].
    """
    indices = sorted(set(indices))
    if not indices:
        print(f"{label}: no sweeps found")
        return

    if expected_count is not None:
        expected = set(range(expected_count))
    else:
        expected = set(range(indices[0], indices[-1] + 1))

    missing = sorted(expected - set(indices))
    if missing:
        print(f"{label}: missing sweeps {missing}")


def check_all_cells_in_dicts(sti_dic, res_dic, expected_count: int | None = None):
    """
    Run sweep-missing checks for all cells stored in sti_dic / res_dic.

    Dictionaries use keys of the form (cell_name, sweep_idx).

    Parameters
    ----------
    sti_dic : dict[(str, int) -> np.ndarray]
    res_dic : dict[(str, int) -> np.ndarray]
    expected_count : int or None
        Passed through to check_indices.
    """
    stim_map = defaultdict(set)
    resp_map = defaultdict(set)

    for cell_name, sweep_idx in sti_dic.keys():
        stim_map[cell_name].add(sweep_idx)

    for cell_name, sweep_idx in res_dic.keys():
        resp_map[cell_name].add(sweep_idx)

    all_cells = sorted(set(stim_map.keys()) | set(resp_map.keys()))

    for cell in all_cells:
        stim_ids = stim_map.get(cell, set())
        resp_ids = resp_map.get(cell, set())

        check_indices(stim_ids, f"{cell} (stored stimuli)", expected_count)
        check_indices(resp_ids, f"{cell} (stored responses)", expected_count)

        only_resp = sorted(resp_ids - stim_ids)
        only_stim = sorted(stim_ids - resp_ids)

        if only_resp:
            print(f"{cell}: response without stimulus sweeps {only_resp}")
        if only_stim:
            print(f"{cell}: stimulus without response sweeps {only_stim}")


###############################################################################
# Pass 1: responses
###############################################################################
def build_response_dict(
    data_ephys_path: Path,
    expected_sweeps: int | None = 80,
    subdir_pattern: str = "sub-mouse-*",
    sub_dirs: list[Path] | None = None,
) -> dict[tuple[str, int], np.ndarray]:
    """
    Build a dictionary of response traces for all cells.
    If sub_dirs is None, directories matching subdir_pattern are scanned.
    Returns a mapping {(cell_name, sweep_idx): np.ndarray}.
    example:
    key = (cell_name, sweep_idx)
    cell_name = 'sub-mouse-AAYYT_ses-20180420-sample-2_slice-20180420-slice-2_cell-20180420-sample-2_icephys'
    sweep_idx = 0
    value = np.ndarray of response trace for that sweep, typically 25k samples.

    """
    if sub_dirs is None:
        sub_dirs = sorted(data_ephys_path.glob(subdir_pattern))

    res_dic: dict[tuple[str, int], np.ndarray] = {}

    for sub_dir in sub_dirs:
        sub_id_simple = sub_dir.name.split("-")[-1]
        print(f"---------------------- {sub_id_simple} (responses) ----------------------")

        nwb_files = sorted(sub_dir.glob("*.nwb"))
        for file_path in nwb_files:
            print(file_path.name)
            cell_name = file_path.stem

            with NWBHDF5IO(str(file_path), "r") as io:
                nwb = io.read()
                resp_by_idx = extract_sweeps_from_nwb_group(nwb.acquisition, prefix="CurrentClampSeries")
                resp_ids = sorted(resp_by_idx.keys())

                if expected_sweeps is not None and resp_ids:
                    expected = set(range(expected_sweeps))
                    missing = sorted(expected - set(resp_ids))
                    if missing:
                        print(f"{file_path.name} response: missing sweeps {missing}")

                for idx, arr in resp_by_idx.items():
                    res_dic[(cell_name, idx)] = arr

    return res_dic


###############################################################################
# Pass 2: stimuli
###############################################################################
def hash_waveform(arr: np.ndarray) -> str:
    """
    Compute a stable hash representing a waveform's dtype, shape, and byte content.

    If xxhash is available, use xxh3_64 for speed. Otherwise, fall back to SHA-256.
    Two identical waveforms produce the same hash; different waveforms (in value,
    shape, or dtype) produce different hashes with very high probability.
    """
    arr_c = np.ascontiguousarray(arr)
    if _HAVE_XXHASH:
        h = xxhash.xxh3_64()
    else:
        h = hashlib.sha256()
    h.update(str(arr_c.dtype).encode("utf-8"))
    h.update(str(arr_c.shape).encode("utf-8"))
    h.update(arr_c.tobytes())
    return h.hexdigest()


def summarize_catalog(
    waveforms: dict[str, np.ndarray],
    waveform_to_sweeps: dict[str, list[tuple[str, int]]],
    sweep_to_waveform: dict[tuple[str, int], str],
) -> None:
    """
    Print a short summary of the catalog, including counts and a histogram of
    duplicate usage per waveform.
    """
    n_waveforms = len(waveforms)
    n_sweeps = len(sweep_to_waveform)
    # number of sweeps per waveform_id
    sizes = [len(v) for v in waveform_to_sweeps.values()]
    if sizes:
        max_use = max(sizes)
        min_use = min(sizes)
        mean_use = sum(sizes) / len(sizes)
    else:
        max_use = 0
        min_use = 0
        mean_use = 0.0
    hist = Counter(sizes)

    print("===== Stimulus catalog summary =====")
    print(f"Unique waveforms      : {n_waveforms}")
    print(f"Total (cell, sweep)   : {n_sweeps}")
    print(f"Min sweeps per wf_id  : {min_use}")
    print(f"Max sweeps per wf_id  : {max_use}")
    print(f"Mean sweeps per wf_id : {mean_use:.2f}")
    print("Histogram (sweeps_per_waveform → count_of_waveforms):")
    for k in sorted(hist):
        print(f"  {k:4d} → {hist[k]}")


###############################################################################
# Stimulus catalog
###############################################################################
def build_stimulus_catalog(
    data_ephys_path: Path,
    subdir_pattern: str = "sub-mouse-*",
    sub_dirs: list[Path] | None = None,
    summarize: bool = True,
) -> dict[str, dict]:
    """
    Build a waveform-based stimulus catalog.

    Description
    -----------
    Every unique stimulus waveform is hashed; all sweeps sharing the same waveform
    map to the same waveform_id. No assumption is made about sweep index consistency
    across cells.

    Returned structure
    ------------------
    catalog = {
        "waveforms":          {waveform_id: np.ndarray},
        "waveform_to_sweeps": {waveform_id: [(cell_id, sweep_id), ...]},
        "sweep_to_waveform":  {(cell_id, sweep_id): waveform_id},
    }

    Example
    -------
    Suppose the file:
        sub-mouse-AVLEY_ses-20180917-cell-2_icephys.nwb
    contains a stimulus for sweep 7. Then a concrete key is:

        (cell_id, sweep_id) = ("sub-mouse-AVLEY_ses-20180917-cell-2_icephys", 7)

    If its waveform hash is:
        wf_id = "d934c1f8f62a2bc0a4b0..."

    Then the catalog contains:
        sweep_to_waveform[("sub-mouse-AVLEY_ses-20180917-cell-2_icephys", 7)] = wf_id
        waveform_to_sweeps[wf_id].append(("sub-mouse-AVLEY_ses-20180917-cell-2_icephys", 7))
        waveforms[wf_id] = representative np.ndarray of the waveform.

    Usage
    -----
    catalog = build_stimulus_catalog(DATA_PATH / "raw" / "raw_data_ephys")
    wf_id = catalog["sweep_to_waveform"][(cell_id, sweep_id)]
    waveform = catalog["waveforms"][wf_id]
    all_sweeps = catalog["waveform_to_sweeps"][wf_id]
    """
    if sub_dirs is None:
        sub_dirs = sorted(data_ephys_path.glob(subdir_pattern))

    waveforms: dict[str, np.ndarray] = {}
    waveform_to_sweeps: dict[str, list[tuple[str, int]]] = {}
    sweep_to_waveform: dict[tuple[str, int], str] = {}

    for sub_dir in sub_dirs:
        sub_id_simple = sub_dir.name.split("-")[-1]
        print(f"---------------------- {sub_id_simple} (stim catalog) ----------------------")

        for file_path in sorted(sub_dir.glob("*.nwb")):
            print(file_path.name)
            cell_id = file_path.stem

            with NWBHDF5IO(str(file_path), "r") as io:
                nwb = io.read()
                stim_by_idx = extract_sweeps_from_nwb_group(nwb.stimulus, prefix="CurrentClampStimulusSeries")

            for sweep_id, arr in stim_by_idx.items():
                wf_id = hash_waveform(arr)

                if wf_id not in waveforms:
                    waveforms[wf_id] = arr
                    waveform_to_sweeps[wf_id] = []

                waveform_to_sweeps[wf_id].append((cell_id, sweep_id))
                sweep_to_waveform[(cell_id, sweep_id)] = wf_id

    if summarize:
        summarize_catalog(waveforms, waveform_to_sweeps, sweep_to_waveform)

    return {
        "waveforms": waveforms,
        "waveform_to_sweeps": waveform_to_sweeps,
        "sweep_to_waveform": sweep_to_waveform,
    }


def plot_all_unique_stimuli(
    waveforms: dict[str, np.ndarray],
    waveform_to_sweeps: dict[str, list[tuple[str, int]]],
    nColor: int = 8,
    nCol: int = 3,
) -> None:
    """
    Plot all unique stimulus waveforms in subplots grouped by their global sweep id.
    global_sweep_id:  Each waveform may appear at different sweep indices across cells. To assign a
    single representative index, we take the mode (most frequent sweep_id) among all (cell_id, sweep_id) pairs for that
    waveform. If multiple sweep_ids tie,the smallest is chosen for determinism. This global_sweep_id is used to sort
    waveforms and label them consistently in plots.

    Parameters
    ----------
    waveforms : dict[str, np.ndarray]
        Mapping from waveform_id to 1D stimulus waveform array.
    waveform_to_sweeps : dict[str, list[tuple[str, int]]]
        Mapping from waveform_id to a list of (cell_id, sweep_id) pairs.
    nColor : int, default 8
        Number of waveforms to overlay per subplot (each with a distinct color).
    nCol : int, default 3
        Number of subplot columns. The number of rows is computed automatically.
    """
    n_waveforms = len(waveforms)
    if n_waveforms == 0:
        print("No waveforms to plot.")
        return

    # Compute representative sweep_id (global_sweep_id) for each waveform_id
    global_sweep_id: dict[str, int] = {}
    for wf_id, sweeps in waveform_to_sweeps.items():
        if not sweeps:
            continue
        sweep_ids = [sweep_id for _, sweep_id in sweeps]
        counts = Counter(sweep_ids)
        max_count = max(counts.values())
        mode_candidates = [sid for sid, c in counts.items() if c == max_count]
        global_sweep_id[wf_id] = min(mode_candidates)

    # Filter waveform_ids to those present in waveforms, sort by global_sweep_id then wf_id
    wf_ids = [wf_id for wf_id in waveforms.keys() if wf_id in global_sweep_id]
    wf_ids.sort(key=lambda wf: (global_sweep_id[wf], wf))

    n_subplots = ceil(len(wf_ids) / nColor)
    n_rows = ceil(n_subplots / nCol)

    fig, axes = plt.subplots(n_rows, nCol, sharex=False, figsize=(4 * nCol, 2.5 * n_rows))
    if n_rows * nCol == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i in range(n_subplots):
        ax = axes[i]
        start = i * nColor
        end = min(start + nColor, len(wf_ids))
        chunk_ids = wf_ids[start:end]

        for j, wf_id in enumerate(chunk_ids):
            y = waveforms[wf_id]
            x = np.arange(len(y))
            color = colors[j % len(colors)]
            gs_id = global_sweep_id[wf_id]
            ax.plot(x, y, color=color, linewidth=0.8, alpha=0.9, label=f"sweep {gs_id}")

        ax.set_title(f"Waveforms {start}–{end - 1}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, ncol=2, loc="upper right", frameon=False)

    for k in range(n_subplots, len(axes)):
        axes[k].axis("off")

    fig.suptitle(f"All N={n_waveforms} unique stimulus waveforms", fontsize=14)
    plt.tight_layout()
    plt.show()


# %%
if __name__ == "__main__":
    data_ephys_path = DATA_PATH / "raw" / "raw_data_ephys"
    ephys_dict_path = DATA_PATH / "raw" / "raw_data_ephys_dict"
    ephys_dict_path.mkdir(parents=True, exist_ok=True)

    # Build and save responses
    response_dict = build_response_dict(data_ephys_path, expected_sweeps=80)
    response_pkl = ephys_dict_path / "response_dict.pkl"  # 20 GB, large file
    with response_pkl.open("wb") as f:
        pickle.dump(response_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved responses to: {response_pkl}")

    del response_dict
    gc.collect()

    # Build and save stimulus bundle
    catalog = build_stimulus_catalog(data_ephys_path, summarize=True)
    stimulus_pkl = ephys_dict_path / "stimulus_catalog.pkl"
    with stimulus_pkl.open("wb") as f:
        pickle.dump(catalog, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved catalog → {stimulus_pkl}")

    # # load back and test
    # with stimulus_pkl.open("rb") as f:
    #     catalog = pickle.load(f)

    sweep_ids = sorted(set(sweep_id for cell_id, sweep_id in catalog['sweep_to_waveform'].keys()))
    print(f'Total {len(sweep_ids)} unique sweep_ids:', sweep_ids)

    # print sweeps for each cell_id
    for cell_id in set(cell_id for cell_id, _ in catalog['sweep_to_waveform'].keys()):
        sweeps_for_cell = [sweep_id for c_id, sweep_id in catalog['sweep_to_waveform'].keys() if c_id == cell_id]
        print(f"Cell {cell_id} has sweeps: {sweeps_for_cell}")

    # let's show the 105 Unique waveforms: , showing it in n * 2 subplots
    plot_all_unique_stimuli(catalog["waveforms"], catalog["waveform_to_sweeps"], nColor=8, nCol=3)

# NOTE: Extract traces from NWB into dictionaries, need to do response and stimulus separately due to memory constraints
###############################################################################
# Optional: global consistency check for all cells after saving
###############################################################################
# check_all_cells_in_dicts(sti_dic, res_dic, expected_count=EXPECTED_SWEEPS)
