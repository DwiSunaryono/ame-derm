#!/usr/bin/env python3
"""Step 4: train the DINOv2 generalist and/or an anatomical-zone expert
(torso/limb/head). Run once per zone (and per fold, for the full 5-fold
protocol) with the appropriate CLI arguments.

Thin wrapper around `python -m ame_derm.experts.train_specialist_expert`;
all CLI arguments are forwarded. Run `--help` for the full argument list.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET = "ame_derm.experts.train_specialist_expert"

if __name__ == "__main__":
    runpy.run_module(TARGET, run_name="__main__")
