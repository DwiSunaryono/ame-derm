#!/usr/bin/env python3
"""Step 5: evaluate the static-fusion MoE (generalist + anatomical experts,
6-view TTA) over all 5 folds — this supplies the generalist-only and
symmetric-MoE rows of the main ablation ladder (see results/README.md).

A thin wrapper over `python -m ame_derm.moe_fusion.eval_moe_champion_5fold`;
every CLI argument is forwarded. See `--help` for the complete argument list.
For the static-vs-learnable-gater ablation, call
`python -m ame_derm.moe_fusion.eval_adaptive_gater` directly instead.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET = "ame_derm.moe_fusion.eval_moe_champion_5fold"

if __name__ == "__main__":
    runpy.run_module(TARGET, run_name="__main__")
