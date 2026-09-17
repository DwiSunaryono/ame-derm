"""Central path configuration for AME-Derm.

All values can be overridden via environment variables (see .env.example)
instead of editing source files directly.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).parent

# Root used by the PanDerm-comparison scripts to locate sibling data/checkpoints.
AME_DERM_ROOT = os.getenv("AME_DERM_ROOT", str(BASE_DIR))

# Phase-1 segmentation crops consumed by every downstream classifier script.
CROP_DIR = os.getenv("CROP_DIR", "BIRNet_Research/outputs/cropped_phase1_final")

# BCN20000 external-validation cohort (see docs/DATA_PREPARATION.md).
BCN_CSV = os.getenv("BCN_CSV", "panderm_datasets/BCN20000/BCN20000/bcn20000.csv")
BCN_IMG = os.getenv("BCN_IMG", "panderm_datasets/BCN20000/BCN20000/images")
