#!/usr/bin/env python3
"""Step 3: audit the split built in step 2 for lesion leakage and metadata
representativeness (see docs/SYSTEM_DOCUMENTATION.md, "Why grouping by
lesion matters").

Thin wrapper around `python -m ame_derm.data_splits.audit_split_representativeness`;
all CLI arguments are forwarded. Run `--help` for the full argument list.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET = "ame_derm.data_splits.audit_split_representativeness"

if __name__ == "__main__":
    runpy.run_module(TARGET, run_name="__main__")
