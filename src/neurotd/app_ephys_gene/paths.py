import os
from pathlib import Path

# Base directory for data; can be overridden by environment variable
GOOGLE_DRIVE_ROOT = Path(
    os.environ.get("NEUROTD_APP_EPHYS_GENE_DIR", "/mnt/g/My Drive/DaifengWangLab/AthanLi-NeuroTD/")
)
DATA_PATH = GOOGLE_DRIVE_ROOT / "data"

DROPBOX_APP_PATH = Path("/mnt/c/Dropbox/DaifengWangLab/NeuroTD/src/neurotd/app_ephys_gene")
RESULT_PATH = DROPBOX_APP_PATH / "results"
FIGURE_PATH = DROPBOX_APP_PATH / "figures"
DATA_PROCESSED_PATH = DROPBOX_APP_PATH / "data/processed"

CVS_MAP_PATH = DATA_PROCESSED_PATH / "cell_name_to_cell_id_and_sample_id.csv"
META_FILE_PATH = DATA_PATH / "raw/patchseq-Scala/data/m1_patchseq_meta_data.csv"
