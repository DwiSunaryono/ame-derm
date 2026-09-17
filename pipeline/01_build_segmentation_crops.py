#!/usr/bin/env python3
"""Step 1: execute the locked Phase-1 segmentation ensemble across the dataset
and write lesion crops into config.CROP_DIR.

A thin wrapper over `python -m ame_derm.segmentation.generate_phase1_crops`;
every CLI argument is forwarded. See `--help` for the complete argument list.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET = "ame_derm.segmentation.generate_phase1_crops"

if __name__ == "__main__":
    runpy.run_module(TARGET, run_name="__main__")
