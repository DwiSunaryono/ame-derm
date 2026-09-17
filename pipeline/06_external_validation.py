#!/usr/bin/env python3
"""Step 6: retrain and evaluate on the independent BCN20000 cohort, following
the same lesion-grouped, matched-seed protocol applied internally.

A thin wrapper over `python -m ame_derm.external_validation.retrain_bcn`;
every CLI argument is forwarded. See `--help` for the complete argument list.
For zero/few-shot baselines rather than retraining, call
`ame_derm.external_validation.external_zero_shot` /
`external_fewshot_router` directly.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET = "ame_derm.external_validation.retrain_bcn"

if __name__ == "__main__":
    runpy.run_module(TARGET, run_name="__main__")
