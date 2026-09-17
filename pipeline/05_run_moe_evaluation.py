#!/usr/bin/env python3
"""Step 5: evaluate the static-fusion MoE (generalist + anatomical experts,
6-view TTA) across all 5 folds — the source of the main ablation ladder's
generalist-only and symmetric-MoE rows (see results/README.md).

Thin wrapper around `python -m ame_derm.moe_fusion.eval_moe_champion_5fold`;
all CLI arguments are forwarded. Run `--help` for the full argument list.
For the static-vs-learnable-gater ablation, use
`python -m ame_derm.moe_fusion.eval_adaptive_gater` directly instead.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET = "ame_derm.moe_fusion.eval_moe_champion_5fold"

if __name__ == "__main__":
    runpy.run_module(TARGET, run_name="__main__")
