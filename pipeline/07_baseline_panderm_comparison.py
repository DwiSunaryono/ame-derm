#!/usr/bin/env python3
"""Step 7: matched-protocol comparison against the PanDerm foundation model
(requires the separate PanDerm repository and its pretrained weights; see
docs/DATA_PREPARATION.md).

Thin wrapper around
`python -m baselines.panderm_comparison.dinov2_official_protocol`; all CLI
arguments are forwarded. Run `--help` for the full argument list. For the
PanDerm-backboned MoE cell of the 2x2 comparison, use
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
