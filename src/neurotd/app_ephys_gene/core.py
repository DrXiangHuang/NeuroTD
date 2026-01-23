"""
Core utilities for application of ephys-gene correlation analysis.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from neurotd.app_ephys_gene.paths import CVS_MAP_PATH, META_FILE_PATH


def extract_sample_id(nwb_name: str) -> str:
    """
    Extract a sample_id using the first occurrence of 'YYYYMMDD-sample-N' in the filename. The extracted substring
    'YYYYMMDD-sample-N' is converted to 'YYYYMMDD_sample_N'.

    Example
    -------
    nwb_name = 'sub-mouse-AAYYT_ses-20180420-sample-2_slice-20180420-slice-2_cell-20180420-sample-2_icephys.nwb'
    -> '20180420_sample_2'
    """
    parts = nwb_name.split("-")
    for i in range(len(parts) - 2):
        if parts[i + 1] == "sample":
            ymd = parts[i]
            sample_num = parts[i + 2].split("_")[0]
            return f"{ymd}_sample_{sample_num}"
    raise ValueError(f"Could not find 'YYYYMMDD-sample-N' pattern in filename: {nwb_name}")


def extract_cell_id(nwb_name: str) -> str:
    """
    Extract a cell_id from the same NWB filename. The mouse ID is parsed from the 'sub-mouse-XXX' prefix, and the
    sample number N is taken from the same 'YYYYMMDD-sample-N' substring used to construct sample_id. The output
    format is 'XXX-N'.

    Example
    -------
    nwb_name = 'sub-mouse-AAYYT_ses-20180420-sample-2_slice-20180420-slice-2_cell-20180420-sample-2_icephys.nwb'
    -> 'AAYYT-2'
    """
    # mouse_id extracted from 'sub-mouse-XXX'
    mouse_id = nwb_name.split("sub-mouse-")[-1].split("_")[0]

    parts = nwb_name.split("-")
    for i in range(len(parts) - 2):
        if parts[i + 1] == "sample":
            sample_num = parts[i + 2].split("_")[0]
            return f"{mouse_id}-{sample_num}"
    raise ValueError(f"Could not find 'YYYYMMDD-sample-N' pattern in filename: {nwb_name}")


def build_sample_maps(raw_ephys_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """
    Build two mappings based on NWB files under raw_ephys_root.
    Build mapping sample_id -> cell_id by extracting both identifiers from each NWB filename. This unifies the mapping
    used in gene expression processing, where sample_ids appear in raw transcriptomic data, and cell_ids label ephys
    sweeps.

    Returns
    -------
    sample_to_cell_id : dict
        Mapping sample_id -> cell_id, used to link gene expression sample_ids to ephys-based cell_ids.
    sample_to_nwb : dict
        Mapping sample_id -> raw NWB filename, recording provenance for later inspection or auditing.

    NOTE: we may have 1329 cells have gene expression, but 1328 have ephys, with "20190211_sample_10" missing in ephys.
    """
    sample_to_cell_id: dict[str, str] = {}
    sample_to_nwb: dict[str, str] = {}

    for sub_dir in raw_ephys_root.iterdir():
        if not sub_dir.is_dir():
            continue
        for nwb_path in sub_dir.glob("*.nwb"):
            name = nwb_path.name
            sample_id = extract_sample_id(name)
            cell_id = extract_cell_id(name)
            sample_to_cell_id[sample_id] = cell_id
            sample_to_nwb[sample_id] = name

    return sample_to_cell_id, sample_to_nwb


def save_sample_maps(
    sample_to_cell_id: dict[str, str | None],
    sample_to_nwb: dict[str, str],
    csv_path: Path,
) -> None:
    """
    Save sample-to-cell and sample-to-NWB mappings into a single CSV with columns:
    sample_id, cell_id, nwb_file_name.

    Assumes both dictionaries use sample_id as keys. If one dictionary contains keys not present in the other, the
    union of keys is used and missing values become NaN in the CSV.
    """
    all_ids = sorted(set(sample_to_cell_id.keys()) | set(sample_to_nwb.keys()))
    cell_ids = [sample_to_cell_id.get(sid) for sid in all_ids]
    nwb_names = [sample_to_nwb.get(sid) for sid in all_ids]
    # remove .nwb extension from nwb_file_name column to generate cell_name
    cell_names = [fname[:-4] if fname is not None and fname.endswith(".nwb") else fname for fname in nwb_names]

    df = pd.DataFrame({"sample_id": all_ids, "cell_id": cell_ids, "cell_name": cell_names, "nwb_file_name": nwb_names})
    df.to_csv(csv_path, index=False)


def load_sample_maps(csv_path: Path = CVS_MAP_PATH) -> tuple[dict[str, str | None], dict[str, str]]:
    """
    Load sample-to-cell and sample-to-NWB mappings from a CSV produced by save_sample_maps().
    NaN cell_id values are converted to None.
    sample_id: str, e.g. '20180420_sample_2'
    cell_id: str | None, e.g. 'AAYYT-2'
    """
    df = pd.read_csv(csv_path)
    df = df.where(df.notna(), None)

    sample_id_to_cell_id: dict[str, str | None] = {}
    sample_id_to_nwb: dict[str, str] = {}

    for sid, cid, fname in zip(df["sample_id"], df["cell_id"], df["nwb_file_name"]):
        sample_id_to_cell_id[sid] = cid
        if fname is not None:
            sample_id_to_nwb[sid] = fname

    return sample_id_to_cell_id, sample_id_to_nwb


def load_cell_name_to_cell_id(csv_path: Path = CVS_MAP_PATH) -> dict[str, str]:
    """
    Load mapping from full NWB-derived cell_name to cell_id.

    Example
    -------
    cell_name:
        sub-mouse-AAYYT_ses-20180420-sample-2_slice-20180420-slice-2_cell-20180420-sample-2_icephys
    cell_id:
        AAYYT-2

    RRequired .csv columns: cell_name, cell_id.

    Example usage
    -------
    cell_name_to_cell_id = load_cell_name_to_cell_id(CVS_MAP_PATH)
    cell_name = "sub-mouse-AAYYT_ses-20180420-sample-2_slice-20180420-slice-2_cell-20180420-sample-2_icephys"
    cell_id = cell_name_to_cell_id[cell_name]
    """
    df = pd.read_csv(csv_path)
    required = {"cell_name", "cell_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    return dict(zip(df["cell_name"], df["cell_id"], strict=True))


def load_cell_name_to_sample_id(csv_path: Path = CVS_MAP_PATH) -> dict[str, str]:
    """
    Load mapping from full NWB-derived cell_name to sample_id.

    Example
    -------
    cell_name:
        sub-mouse-AAYYT_ses-20180420-sample-2_slice-20180420-slice-2_cell-20180420-sample-2_icephys
    sample_id:
        20180420_sample_2

    Required .csv columns: cell_name, sample_id.

    Example usage
    -------
    cell_name_to_sample_id_map = load_cell_name_to_sample_id(CVS_MAP_PATH)
    cell_name = "sub-mouse-AAYYT_ses-20180420-sample-2_slice-20180420-slice-2_cell-20180420-sample-2_icephys"
    sample_id = cell_name_to_sample_id_map[cell_name]
    """
    df = pd.read_csv(csv_path)
    required = {"cell_name", "sample_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    return dict(zip(df["cell_name"], df["sample_id"], strict=True))


def parse_array(text: str) -> np.ndarray:
    """
    Convert a bracketed numeric string into a NumPy array. Works with whitespace, including line breaks, because
    NumPy's fromstring() treats any whitespace (space, tab, newline) as a separator. Preserves integer dtype when
    possible; supports both comma-separated and space-separated formats.
    --------
    >>> parse_array("[ 1060, 2916, 5254 ]")
    array([1060, 2916, 5254])

    >>> parse_array("[ 74.24 93.52 104.64 ]")
    array([ 74.24,  93.52, 104.64])
    """
    text = text.strip("[]").replace(",", " ")  # In order to handle both commas and spaces, replace commas with spaces
    arr = np.fromstring(text, sep=" ")  # any whitespace as a separator: spaces: " ", tabs: "\t", newlines: "\n"

    # Preserve integer dtype if no decimals
    if np.all(arr == arr.astype(int)):
        return arr.astype(int)
    return arr


def parse_vector_columns(df: pd.DataFrame, vec_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Convert string-encoded arrays in selected DataFrame columns into NumPy arrays.

    Example
    -------
    df = pd.DataFrame({
        "peak_idx": ["[ 1060 2916 5254 ]"],
        "time_delays(ms)": ["[ 74.24 93.52 104.64 ]"]
    }, index=["AAYYT-2-15"])

    df2 = parse_vector_columns(df)
    """
    if vec_cols is None:
        vec_cols = ["peak_idx", "time_delays(ms)"]

    df2 = df.copy()
    for col in vec_cols:
        df2[col] = df2[col].apply(parse_array)  # type: ignore[arg-type]
    return df2


def get_cell_ids_by_rna_family(
    rna_family: str,
    sample_id_to_cell_id: dict[str, str],
    *,
    meta_file: Path = META_FILE_PATH,
    sample_id_col: str = "Cell",
    rna_family_col: str = "RNA family",
) -> list[str]:
    """
    Retrieve NeuroTD cell_id values for a given Allen RNA family using Patch-seq metadata
    (Scala et al., Nature 2021).

    The canonical mapping is:
        RNA family → sample_id → cell_id

    Parameters
    ----------
    rna_family : str
        Allen RNA family name, ordered by dataset abundance (mouse M1 Patch-seq, n = 1329 cells total):

        - 'Pvalb'        (289 cells): parvalbumin interneurons, InN
        - 'Sst'          (272 cells): somatostatin interneurons, InN
        - 'IT'           (254 cells): intratelencephalic excitatory, ExN
        - 'Vip'          (153 cells): VIP interneurons, InN
        - 'CT'           (106 cells): corticothalamic excitatory, ExN
        - 'low quality'   (97 cells): low-quality Patch-seq profiles, mixed
        - 'Lamp5'         (91 cells): L2 IT / neurogliaform-like interneurons, InN
        - 'ET'            (49 cells): extratelencephalic excitatory, ExN
        - 'Sncg'          (13 cells): Sncg interneurons, InN
        - 'NP'             (5 cells): near-projecting excitatory, ExN

    meta_file : Path
        Path to `m1_patchseq_meta_data.csv`.

    sample_id_to_cell_id : dict[str, str]
        Mapping from Patch-seq sample_id to compact NeuroTD cell_id, typically returned by
        `load_sample_maps()`.

    sample_id_col : str, default='Sample ID'
        Column in metadata corresponding to Patch-seq sample identifiers (e.g. '20171204_sample_4').

    rna_family_col : str, default='RNA family'
        Column in metadata corresponding to Allen RNA family labels.

    Returns
    -------
    list[str]
        Sorted list of NeuroTD cell_id values corresponding to the requested RNA family. Sample IDs
        without a valid mapping are silently skipped.

    Notes
    -----
    Inhibitory neuron families (InN): Pvalb, Sst, Vip, Lamp5, Sncg.

    Excitatory neuron families (ExN): IT, CT, ET, NP.

    The 'low quality' category is heterogeneous and typically excluded from downstream analyses.

    Examples
    --------
    Retrieve IT neuron cell IDs::

        meta_file = DATA_PATH / "raw/patchseq-Scala/data/m1_patchseq_meta_data.csv"
        it_cell_ids = get_cell_ids_by_rna_family(
            "IT",
            sample_id_to_cell_id=sample_id_to_cell_id,
            meta_file=meta_file,
        )

    Retrieve Pvalb interneuron cell IDs::

        pvalb_cell_ids = get_cell_ids_by_rna_family(
            "Pvalb",
            sample_id_to_cell_id=sample_id_to_cell_id,
            meta_file=meta_file,
        )

    Filter a NeuroTD result table indexed by cell_id::

        df_it = df.loc[df.index.isin(it_cell_ids)]

    Filter all inhibitory neurons (InN)::

        inn_families = ["Pvalb", "Sst", "Vip", "Lamp5", "Sncg"]
        inn_cell_ids = []

        for fam in inn_families:
            inn_cell_ids.extend(
                get_cell_ids_by_rna_family(
                    fam,
                    sample_id_to_cell_id=sample_id_to_cell_id,
                    meta_file=meta_file,
                )
            )

        inn_cell_ids = sorted(set(inn_cell_ids))
        df_inn = df.loc[df.index.isin(inn_cell_ids)]
    """
    meta = pd.read_csv(meta_file, sep="\t")

    subset = meta.loc[meta[rna_family_col] == rna_family]

    sample_ids = subset[sample_id_col].astype(str).unique()

    cell_ids = sorted(sample_id_to_cell_id[sid] for sid in sample_ids if sid in sample_id_to_cell_id)

    return cell_ids
