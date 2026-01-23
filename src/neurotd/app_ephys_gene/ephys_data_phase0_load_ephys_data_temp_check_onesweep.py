from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pynwb import NWBHDF5IO

from neurotd.app_ephys_gene.paths import DATA_PATH


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


def _chunk_indices(indices, chunk_size=8):
    indices = list(indices)
    for i in range(0, len(indices), chunk_size):
        yield indices[i : i + chunk_size]


def plot_all_sweeps_for_cell(
    sweeps_by_idx: dict[int, np.ndarray],
    cell_name: str,
    kind: str = "response",
    sweeps_per_subplot: int = 8,
):
    if not sweeps_by_idx:
        print(f"{cell_name} [{kind}]: no sweeps")
        return

    all_ids = sorted(sweeps_by_idx.keys())
    chunks = list(_chunk_indices(all_ids, chunk_size=sweeps_per_subplot))
    n_subplots = len(chunks)

    # 2 columns, computed rows
    cols = 2
    rows = ceil(n_subplots / 2)

    fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(12, 2.5 * rows))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Color cycle
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
    ]

    # Plot each chunk in a subplot
    for ax, chunk_ids in zip(axes, chunks):
        for j, sweep_id in enumerate(chunk_ids):
            y = sweeps_by_idx[sweep_id]
            x = np.arange(len(y))
            ax.plot(
                x,
                y,
                color=colors[j % 8],
                alpha=0.8,
                linewidth=0.8,
                label=f"{sweep_id}",
            )
        ax.set_title(f"Sweeps {chunk_ids[0]} – {chunk_ids[-1]}", fontsize=8)
        ax.tick_params(labelsize=7)

    # Turn off unused axes
    for k in range(n_subplots, rows * cols):
        axes[k].axis("off")

    fig.suptitle(f"{cell_name} — all {kind} sweeps", fontsize=14)
    plt.tight_layout()
    plt.show()



###############################################################################
# 2. Load NWB and extract sweeps
###############################################################################
with NWBHDF5IO(str(first_file), "r") as io:
    nwb = io.read()

    stim_by_idx = extract_sweeps_from_nwb_group(
        nwb.stimulus,
        prefix="CurrentClampStimulusSeries",
    )
    resp_by_idx = extract_sweeps_from_nwb_group(
        nwb.acquisition,
        prefix="CurrentClampSeries",
    )

    stim_ids = set(stim_by_idx.keys())
    resp_ids = set(resp_by_idx.keys())
    common_ids = sorted(stim_ids & resp_ids)

    print("Stim sweeps:", sorted(stim_ids))
    print("Resp sweeps:", sorted(resp_ids))
    print("Common sweeps:", common_ids)

    if not common_ids:
        raise RuntimeError("No overlapping stimulus/response sweeps for this cell.")

###############################################################################
# 3. Plot one sweep (for example, the first common sweep)
###############################################################################
sweep_id = common_ids[0]  # change to another index if needed
sti = stim_by_idx[sweep_id]
res = resp_by_idx[sweep_id]

fig, (ax_sti, ax_res) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

x_sti = np.arange(len(sti))
x_res = np.arange(len(res))

ax_sti.plot(x_sti, sti)
ax_sti.set_ylabel("Stimulus (arb. units)")
ax_sti.set_title(f"{first_file.stem} — sweep {sweep_id} (stimulus)")

ax_res.plot(x_res, res)
ax_res.set_ylabel("Response (arb. units)")
ax_res.set_xlabel("Sample index")
ax_res.set_title(f"{first_file.stem} — sweep {sweep_id} (response)")

plt.tight_layout()
plt.show()

# %%
data_ephys_path = DATA_PATH / "raw" / "raw_data_ephys"
sub_dir = data_ephys_path / "sub-mouse-YAVWS"  # HXCVU, ZZVQT, YAVWS # sample 9

nwb_files = sorted(sub_dir.glob("*.nwb"))
first_file = nwb_files[-1]
print(f"Examining: {first_file.name}")

with NWBHDF5IO(str(first_file), "r") as io:
    nwb = io.read()

    stim_by_idx = extract_sweeps_from_nwb_group(
        nwb.stimulus,
        prefix="CurrentClampStimulusSeries",
    )
    resp_by_idx = extract_sweeps_from_nwb_group(
        nwb.acquisition,
        prefix="CurrentClampSeries",
    )

    stim_ids = set(stim_by_idx.keys())
    resp_ids = set(resp_by_idx.keys())
    common_ids = sorted(stim_ids & resp_ids)

    print("Stim sweeps:", sorted(stim_ids))
    print("Resp sweeps:", sorted(resp_ids))
    print("Common sweeps:", common_ids)

    if not common_ids:
        raise RuntimeError("No overlapping stimulus/response sweeps for this cell.")


cell_name = first_file.stem

# Plot all responses (main interest)
plot_all_sweeps_for_cell(resp_by_idx, cell_name=cell_name, kind="response")

# Optionally, also plot all stimuli
plot_all_sweeps_for_cell(stim_by_idx, cell_name=cell_name, kind="stimulus")
