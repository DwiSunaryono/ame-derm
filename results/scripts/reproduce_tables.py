#!/usr/bin/env python3
"""Regenerates results/tables/*.csv from the raw JSON in results/data/ —
no training, no GPU, no API calls, just arithmetic over already-computed
per-sample predictions and per-fold metrics.

Run from the results/ directory:

    cd results
    python scripts/reproduce_tables.py

Produces:
    tables/main_ablation_ladder.csv     - generalist vs. symmetric MoE vs.
                                           AME-Derm (see caveat below)
    tables/external_validation_bcn.csv  - per-seed macro-F1 on the BCN20000
                                           external test set
    tables/split_audit_summary.csv      - per-field split-representativeness
                                           audit, averaged over 5 folds

See docs/SYSTEM_DOCUMENTATION.md in the repository root for what each
number means and which paper section it backs.
"""
import csv
import json
import os

import numpy as np

try:
    from sklearn.metrics import f1_score
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required (pip install -r ../requirements.txt)"
    ) from exc

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.dirname(HERE)
DATA_DIR = os.path.join(RESULTS_DIR, "data")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# Per-fold difference (AME-Derm - symmetric MoE), macro-F1 points, as reported
# in the manuscript (Results, Section 3, "Statistical Significance") — the
# asymmetric AME-Derm model's own raw per-fold output was not recovered from
# archived logs (see docs/SYSTEM_DOCUMENTATION.md), so its ladder entry below
# is champ_static_f1 + this documented delta, not an independently-computed
# value.
PAPER_DELTAS_PTS = np.array([1.33, 0.71, 0.73, 1.81, -0.37])


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {os.path.relpath(path, RESULTS_DIR)}")


def main_ablation_ladder():
    with open(os.path.join(DATA_DIR, "moe_champion_5fold_results.json")) as f:
        folds = json.load(f)

    gen_only = np.array([r["gen_only_f1"] for r in folds]) * 100
    symmetric_moe = np.array([r["champ_static_f1"] for r in folds]) * 100
    ame_derm = symmetric_moe + PAPER_DELTAS_PTS

    rows = [
        ("Single DINOv2 generalist", f"{gen_only.mean():.2f}", f"{gen_only.std(ddof=1):.2f}"),
        ("Symmetric MoE (base experts + base generalist)",
         f"{symmetric_moe.mean():.2f}", f"{symmetric_moe.std(ddof=1):.2f}"),
        ("AME-Derm (asymmetric; reconstructed, see script docstring)",
         f"{ame_derm.mean():.2f}", f"{ame_derm.std(ddof=1):.2f}"),
    ]
    write_csv(
        os.path.join(TABLES_DIR, "main_ablation_ladder.csv"),
        ["method", "macro_f1_mean_pct", "macro_f1_std_pct"],
        rows,
    )


def external_validation_bcn():
    rows = []
    for fname in sorted(os.listdir(os.path.join(DATA_DIR, "bcn_external"))):
        tag = fname.replace(".json", "")
        with open(os.path.join(DATA_DIR, "bcn_external", fname)) as f:
            d = json.load(f)
        macro_f1 = f1_score(d["labs"], d["preds"], average="macro") * 100
        rows.append((tag, d["n"], f"{macro_f1:.2f}"))
    write_csv(
        os.path.join(TABLES_DIR, "external_validation_bcn.csv"),
        ["run", "n_test", "macro_f1_pct"],
        rows,
    )


def split_audit_summary():
    with open(os.path.join(DATA_DIR, "split_audit.json")) as f:
        audit = json.load(f)

    fields = audit["fields"]
    rows = []
    for split_name in ("test", "val"):
        for field in fields:
            mean_diffs, max_diffs, coverages = [], [], []
            for fold_key, fold_data in audit["folds"].items():
                stat = fold_data[split_name][field]
                mean_diffs.append(stat["mean_abs_diff"])
                max_diffs.append(stat["max_abs_diff"])
                coverages.append(stat["coverage"])
            rows.append((
                split_name, field,
                f"{np.mean(mean_diffs):.4f}",
                f"{np.max(max_diffs):.4f}",
                f"{np.mean(coverages):.4f}",
            ))
    write_csv(
        os.path.join(TABLES_DIR, "split_audit_summary.csv"),
        ["split", "field", "mean_abs_diff_avg_over_folds", "max_abs_diff_over_folds", "coverage_avg_over_folds"],
        rows,
    )


if __name__ == "__main__":
    main_ablation_ladder()
    external_validation_bcn()
    split_audit_summary()
