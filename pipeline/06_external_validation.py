#!/usr/bin/env python3
"""Step 6: retrain/evaluate on the independent BCN20000 cohort under the
same lesion-grouped, matched-seed protocol used internally.

Thin wrapper around `python -m ame_derm.external_validation.retrain_bcn`;
all CLI arguments are forwarded. Run `--help` for the full argument list.
For zero/few-shot baselines instead of retraining, use
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
