"""Shared filesystem-path settings for AME-Derm.

Rather than editing the source, each value may be replaced through an
environment variable (refer to .env.example).
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).parent

# Directory root the PanDerm-comparison scripts use to find sibling data and checkpoints.
AME_DERM_ROOT = os.getenv("AME_DERM_ROOT", str(BASE_DIR))

# Segmentation crops from Phase 1, read by all downstream classifier scripts.
CROP_DIR = os.getenv("CROP_DIR", "BIRNet_Research/outputs/cropped_phase1_final")

# External-validation cohort from BCN20000 (refer to docs/DATA_PREPARATION.md).
BCN_CSV = os.getenv("BCN_CSV", "panderm_datasets/BCN20000/BCN20000/bcn20000.csv")
BCN_IMG = os.getenv("BCN_IMG", "panderm_datasets/BCN20000/BCN20000/images")
