#!/usr/bin/env python3
"""Step 2: construct the lesion-grouped, metadata-stratified 5-fold split and
export the CSVs for its folds.

A thin wrapper over `python -m ame_derm.data_splits.export_representative_folds`;
every CLI argument is forwarded. See `--help` for the complete argument list.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET = "ame_derm.data_splits.export_representative_folds"

if __name__ == "__main__":
    runpy.run_module(TARGET, run_name="__main__")
