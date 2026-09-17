#!/usr/bin/env python3
"""Step 3: check the split produced in step 2 for lesion leakage and for
metadata representativeness (see docs/SYSTEM_DOCUMENTATION.md, "Why grouping by
lesion matters").

A thin wrapper over `python -m ame_derm.data_splits.audit_split_representativeness`;
every CLI argument is forwarded. See `--help` for the complete argument list.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET = "ame_derm.data_splits.audit_split_representativeness"

if __name__ == "__main__":
    runpy.run_module(TARGET, run_name="__main__")
