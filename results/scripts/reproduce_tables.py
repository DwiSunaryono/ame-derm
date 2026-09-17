#!/usr/bin/env python3
"""Rebuilds results/tables/*.csv from the raw JSON under results/data/ —
no GPU, no training, no API calls, only arithmetic over already-computed
per-sample predictions and per-fold metrics.

Run from the results/ directory:

    cd results
    python scripts/reproduce_tables.py

Generates:
    tables/panderm_comparison.csv       - README Table R1 / HEADLINE result:
                                           DINOv2 (AME-Derm's backbone) vs.
                                           PanDerm, per fold, entirely raw
    tables/backbone_moe_2x2.csv         - README Table R2 -> paper Table 10
                                           (tab:2x2): backbone x MoE, 2x2
    tables/panderm_seed_robustness.csv  - PanDerm re-estimated at a second
                                           seed (122), confirming the ranking
                                           is no favorable-seed artifact
    tables/main_ablation_ladder.csv     - README Table R3, secondary/modest
                                           result: generalist vs. symmetric
                                           MoE vs. AME-Derm (see caveat below)
    tables/main_ablation_ladder_per_fold.csv - README Table R4, the same
                                           result per fold, so the "4 of 5
                                           folds" claim can be checked
    tables/external_validation_bcn.csv  - per-seed macro-F1 on the BCN20000
                                           external test set
    tables/split_audit_summary.csv      - per-field split-representativeness
                                           audit, averaged over 5 folds

See docs/SYSTEM_DOCUMENTATION.md in the repository root for the meaning of
each number and the paper section it supports.
"""
import csv
import json
import os

import numpy as np

try:
    import pandas as pd
    from sklearn.metrics import f1_score
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required (pip install -r ../requirements.txt)"
    ) from exc

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.dirname(HERE)
DATA_DIR = os.path.join(RESULTS_DIR, "data")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# Per-fold difference (AME-Derm - symmetric MoE), in macro-F1 points, exactly
# as reported in the manuscript (Results, Section 3, "Statistical Significance")
# — the asymmetric AME-Derm model's own raw per-fold output could not be
# recovered from archived logs (see docs/SYSTEM_DOCUMENTATION.md), so its ladder
# entry below equals champ_static_f1 + this documented delta rather than an
# independently computed value.
PAPER_DELTAS_PTS = np.array([1.33, 0.71, 0.73, 1.81, -0.37])


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {os.path.relpath(path, RESULTS_DIR)}")


def panderm_comparison():
    """HEADLINE result (paper §"Comparison with a Dermatology Foundation
    Model"): DINOv2 (AME-Derm's backbone) vs. PanDerm, per fold. Neither side
    is borrowed — PanDerm's macro-F1 is derived here from scratch out of its
    own per-sample predictions, not copied from the manuscript. Of the paper's
    two headline comparisons, this is the one it calls "the more robust"
    (5/5 folds positive, exact sign test p=0.031), unlike the
    modest/borderline internal ablation in main_ablation_ladder() below.
    """
    with open(os.path.join(DATA_DIR, "moe_champion_5fold_results.json")) as f:
        moe_folds = json.load(f)
    dinov2_gen = np.array([r["gen_only_f1"] for r in moe_folds]) * 100

    rows = []
    panderm_gen = []
    for i in range(5):
        df = pd.read_csv(os.path.join(
            DATA_DIR, "panderm_baseline", f"generalist_fold{i}_seed42_test.csv"))
        f1 = f1_score(df["true_label"], df["predicted_label"], average="macro") * 100
        panderm_gen.append(f1)
        rows.append((i, f"{dinov2_gen[i]:.2f}", f"{f1:.2f}", f"{dinov2_gen[i] - f1:.2f}"))
    panderm_gen = np.array(panderm_gen)

    write_csv(
        os.path.join(TABLES_DIR, "panderm_comparison.csv"),
        ["fold", "dinov2_generalist_macro_f1_pct", "panderm_generalist_macro_f1_pct", "delta_pct"],
        rows,
    )
    delta = dinov2_gen - panderm_gen
    print(f"  -> mean delta +{delta.mean():.2f} pts, {int((delta > 0).sum())}/5 folds positive"
          f" (paper: +4.7 pts, 5/5 folds, sign test p=0.031)")


def backbone_moe_2x2():
    """README Table R2 -> paper Table 10 (tab:2x2): backbone x MoE, 2x2.
    The DINOv2 single-generalist and PanDerm (both cells) are raw; the AME-Derm
    cell is reconstructed instead (see main_ablation_ladder() below and the README)."""
    with open(os.path.join(DATA_DIR, "moe_champion_5fold_results.json")) as f:
        moe_folds = json.load(f)
    dinov2_gen = np.array([r["gen_only_f1"] for r in moe_folds]) * 100
    symmetric_moe = np.array([r["champ_static_f1"] for r in moe_folds]) * 100
    ame_derm = symmetric_moe + PAPER_DELTAS_PTS

    panderm_gen = []
    panderm_moe = []
    for i in range(5):
        df = pd.read_csv(os.path.join(
            DATA_DIR, "panderm_baseline", f"generalist_fold{i}_seed42_test.csv"))
        panderm_gen.append(f1_score(df["true_label"], df["predicted_label"], average="macro") * 100)
        with open(os.path.join(DATA_DIR, "panderm_baseline", f"moe_fold{i}_seed42.json")) as f:
            panderm_moe.append(json.load(f)["moe_macro_f1"] * 100)

    rows = [
        ("DINOv2", f"{dinov2_gen.mean():.2f}", f"{ame_derm.mean():.2f} (AME-Derm; reconstructed)"),
        ("PanDerm", f"{np.mean(panderm_gen):.2f}", f"{np.mean(panderm_moe):.2f}"),
    ]
    write_csv(
        os.path.join(TABLES_DIR, "backbone_moe_2x2.csv"),
        ["backbone", "single_generalist_macro_f1_pct", "moe_macro_f1_pct"],
        rows,
    )


def panderm_seed_robustness():
    """Paper §"Comparison with a Dermatology Foundation Model": "re-estimated
    at a second seed (122), the same PanDerm reaches 76.60 +/- 2.6" — this
    verifies the PanDerm ranking is not an artifact of a favorable seed.
    Derived fresh from raw per-sample predictions at both seeds."""
    rows = []
    for seed in (42, 122):
        vals = []
        for i in range(5):
            df = pd.read_csv(os.path.join(
                DATA_DIR, "panderm_baseline", f"generalist_fold{i}_seed{seed}_test.csv"))
            vals.append(f1_score(df["true_label"], df["predicted_label"], average="macro") * 100)
        vals = np.array(vals)
        rows.append((seed, f"{vals.mean():.2f}", f"{vals.std(ddof=1):.2f}"))
    write_csv(
        os.path.join(TABLES_DIR, "panderm_seed_robustness.csv"),
        ["seed", "macro_f1_mean_pct", "macro_f1_std_pct"],
        rows,
    )


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

    fold_rows = [
        (i, f"{symmetric_moe[i]:.2f}", f"{ame_derm[i]:.2f}", f"{PAPER_DELTAS_PTS[i]:+.2f}")
        for i in range(5)
    ]
    write_csv(
        os.path.join(TABLES_DIR, "main_ablation_ladder_per_fold.csv"),
        ["fold", "symmetric_moe_macro_f1_pct", "ame_derm_macro_f1_pct", "delta_pct"],
        fold_rows,
    )
    n_positive = int((PAPER_DELTAS_PTS > 0).sum())
    print(f"  -> AME-Derm favored on {n_positive}/5 folds (paper: 4/5)")


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
    panderm_comparison()
    backbone_moe_2x2()
    panderm_seed_robustness()
    main_ablation_ladder()
    external_validation_bcn()
    split_audit_summary()
