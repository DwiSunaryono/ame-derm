#!/usr/bin/env python3
"""Step 7: matched-protocol comparison with the PanDerm foundation model (the
separate PanDerm repository and its pretrained weights are required; see
docs/DATA_PREPARATION.md).

A thin wrapper over
`python -m baselines.panderm_comparison.dinov2_official_protocol`; every CLI
argument is forwarded. See `--help` for the complete argument list. For the
PanDerm-backboned MoE cell of the 2x2 comparison, call
`baselines.panderm_comparison.panderm_moe_prep` /
`panderm_moe_eval` directly instead.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET = "baselines.panderm_comparison.dinov2_official_protocol"

if __name__ == "__main__":
    runpy.run_module(TARGET, run_name="__main__")
