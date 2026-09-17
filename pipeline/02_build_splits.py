#!/usr/bin/env python3
"""Step 2: build the lesion-grouped, metadata-stratified 5-fold split and
export its fold CSVs.

Thin wrapper around `python -m ame_derm.data_splits.export_representative_folds`;
all CLI arguments are forwarded. Run `--help` for the full argument list.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET = "ame_derm.data_splits.export_representative_folds"

if __name__ == "__main__":
    runpy.run_module(TARGET, run_name="__main__")
