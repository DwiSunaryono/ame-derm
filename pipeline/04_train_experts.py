#!/usr/bin/env python3
"""Step 4: train either the DINOv2 generalist, an anatomical-zone expert
(torso/limb/head), or both. Invoke once per zone (and per fold, for the full
5-fold protocol) with the matching CLI arguments.

A thin wrapper over `python -m ame_derm.experts.train_specialist_expert`;
every CLI argument is forwarded. See `--help` for the complete argument list.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET = "ame_derm.experts.train_specialist_expert"

if __name__ == "__main__":
    runpy.run_module(TARGET, run_name="__main__")
