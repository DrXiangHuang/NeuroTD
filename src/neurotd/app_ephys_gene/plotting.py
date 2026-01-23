from math import ceil

import matplotlib.pyplot as plt
import numpy as np


def _chunk_indices(indices, chunk_size=8):
    indices = list(indices)
    for i in range(0, len(indices), chunk_size):
        yield indices[i : i + chunk_size]


def plot_all_sweeps_for_cell(
    sweeps_by_idx: dict[int, np.ndarray],
    cell_name: str,
    kind: str = "response",
    sweeps_per_subplot: int = 8,
) -> plt.Figure:
    """
    Plot all sweeps for a single cell. Returns the matplotlib Figure object
    so the caller can save it to disk.

    Parameters
    ----------
    sweeps_by_idx : dict[int, np.ndarray]
        Mapping from sweep_id to 1D trace (stimulus or response).
    cell_name : str
        Cell identifier for the title. such as:
        'sub-mouse-AAYYT_ses-20180420-sample-2_slice-20180420-slice-2_cell-20180420-sample-2_icephys'
    kind : str, default "response"
        Label used in the figure title ("response", "stimulus", etc.).
    sweeps_per_subplot : int, default 8
        Number of sweeps plotted in each subplot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure (not shown or closed), allowing the caller
        to save it manually.
    """
    if not sweeps_by_idx:
        print(f"{cell_name} [{kind}]: no sweeps")
        return None

    # Sort sweep IDs
    all_ids = sorted(sweeps_by_idx.keys())

    # Chunk sweep IDs
    def _chunk_indices(lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i : i + chunk_size]

    chunks = list(_chunk_indices(all_ids, sweeps_per_subplot))
    n_subplots = len(chunks)

    cols = 2
    rows = ceil(n_subplots / cols)

    fig, axes = plt.subplots(rows, cols, sharex=False, figsize=(12, 2.5 * rows))

    # Flatten axes for iteration
    axes = np.atleast_1d(axes).flatten()

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

    # Plot sweeps
    for ax, chunk_ids in zip(axes, chunks):
        for j, sweep_id in enumerate(chunk_ids):
            y = sweeps_by_idx[sweep_id]
            x = np.arange(len(y))
            ax.plot(
                x,
                y,
                color=colors[j % len(colors)],
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
    fig.tight_layout()

    return fig
