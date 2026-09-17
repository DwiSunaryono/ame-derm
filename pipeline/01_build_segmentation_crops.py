#!/usr/bin/env python3
"""Step 1: run the locked Phase-1 segmentation ensemble over the dataset and
write lesion crops to config.CROP_DIR.

Thin wrapper around `python -m ame_derm.segmentation.generate_phase1_crops`;
all CLI arguments are forwarded. Run `--help` for the full argument list.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET = "ame_derm.segmentation.generate_phase1_crops"

if __name__ == "__main__":
    runpy.run_module(TARGET, run_name="__main__")
