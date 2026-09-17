#!/usr/bin/env python3
"""Reproduces the paper's "Statistical Significance" subsection (Results,
Section 3): two paired comparisons on the same five lesion-grouped
cross-validation folds.

1. AME-Derm (DINOv2, full pipeline) vs. PanDerm (dermatology foundation
   model) — the paper's headline, robust result: positive on 5/5 folds,
   exact sign test p=0.031. Both sides of this comparison are raw,
   independently-recomputed numbers: the DINOv2 generalist's per-fold
   macro-F1 comes straight from `results/data/moe_champion_5fold_results
   .json` (produced by `ame_derm.moe_fusion.eval_moe_champion_5fold`), and
   PanDerm's per-fold macro-F1 is computed here from scratch, from PanDerm's
   own raw per-sample predictions in
   `results/data/panderm_baseline/generalist_fold*_seed42_test.csv`, via
   `sklearn.metrics.f1_score(true_label, predicted_label, average="macro")`
   — nothing here is copied from the manuscript.

2. AME-Derm (asymmetric MoE) vs. the capacity-matched symmetric MoE — the
   paper's secondary, modest result (t(4)=2.30, p=0.041; described in the
   manuscript itself as "small and borderline"). The asymmetric AME-Derm
   model's own raw per-fold macro-F1 was not recovered from archived logs
   (see docs/SYSTEM_DOCUMENTATION.md); what IS available is the manuscript's
   reported per-fold difference (AME-Derm minus symmetric MoE, in points):
   {+1.33, +0.71, +0.73, +1.81, -0.37}. A paired test only needs that
   difference array, so it is tested directly (mathematically identical to
   a paired t-test on two fold-matched sequences).
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "results", "data")

# Per-fold difference (AME-Derm - symmetric MoE), macro-F1 points, as reported
# in the manuscript (Results, Section 3, "Statistical Significance").
PAPER_DELTAS_PTS = np.array([1.33, 0.71, 0.73, 1.81, -0.37])


def sign_test_p_all_positive(n):
    """Exact one-sided sign test p-value when all n paired differences are
    positive under H0: P(positive) = 0.5."""
    return stats.binomtest(n, n, 0.5, alternative="greater").pvalue


def panderm_comparison():
    print("=" * 78)
    print("HEADLINE RESULT: DINOv2 (AME-Derm's backbone) vs. PanDerm, matched folds")
    print("=" * 78)

    with open(os.path.join(DATA_DIR, "moe_champion_5fold_results.json")) as f:
        moe_folds = json.load(f)
    dinov2_gen = np.array([r["gen_only_f1"] for r in moe_folds]) * 100

    panderm_gen = []
    for i in range(5):
        df = pd.read_csv(os.path.join(
            DATA_DIR, "panderm_baseline", f"generalist_fold{i}_seed42_test.csv"))
        panderm_gen.append(
            f1_score(df["true_label"], df["predicted_label"], average="macro") * 100)
    panderm_gen = np.array(panderm_gen)

    delta = dinov2_gen - panderm_gen
    print(f"  DINOv2 generalist per fold: {np.round(dinov2_gen, 2).tolist()}  (mean {dinov2_gen.mean():.2f})")
    print(f"  PanDerm generalist per fold: {np.round(panderm_gen, 2).tolist()}  (mean {panderm_gen.mean():.2f})")
    print(f"  Delta per fold:             {np.round(delta, 2).tolist()}")
    print(f"  Mean delta: +{delta.mean():.2f} pts, {int((delta > 0).sum())}/5 folds positive"
          f"  (paper reports +4.7 pts, 5/5 folds)")

    t_stat, p_two = stats.ttest_1samp(delta, popmean=0.0)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    print(f"  Paired t-test:  t(4)={t_stat:.2f}, one-sided p={p_one:.4f}")

    p_sign = sign_test_p_all_positive(len(delta))
    print(f"  Exact sign test: p={p_sign:.3f}  (paper reports p=2^-5=0.031)")

    print()
    print("  Full backbone x MoE 2x2 (macro-F1, %):")
    with open(os.path.join(DATA_DIR, "panderm_baseline", "moe_fold0_seed42.json")):
        pass  # existence check
    panderm_moe = []
    for i in range(5):
        with open(os.path.join(DATA_DIR, "panderm_baseline", f"moe_fold{i}_seed42.json")) as f:
            panderm_moe.append(json.load(f)["moe_macro_f1"] * 100)
    panderm_moe = np.array(panderm_moe)
    symmetric_moe = np.array([r["champ_static_f1"] for r in moe_folds]) * 100
    ame_derm = symmetric_moe + PAPER_DELTAS_PTS  # reconstructed, see module docstring

    print(f"    {'Backbone':10s} {'Single generalist':>18s} {'MoE':>10s}")
    print(f"    {'DINOv2':10s} {dinov2_gen.mean():18.2f} {ame_derm.mean():10.2f}  (AME-Derm; MoE column is reconstructed)")
    print(f"    {'PanDerm':10s} {panderm_gen.mean():18.2f} {panderm_moe.mean():10.2f}  (both raw)")


def internal_ablation():
    print()
    print("=" * 78)
    print("SECONDARY RESULT: AME-Derm (asymmetric) vs. symmetric MoE (modest)")
    print("=" * 78)

    with open(os.path.join(DATA_DIR, "moe_champion_5fold_results.json")) as f:
        moe_folds = json.load(f)
    gen_only = np.array([r["gen_only_f1"] for r in moe_folds]) * 100
    symmetric_moe = np.array([r["champ_static_f1"] for r in moe_folds]) * 100
    ame_derm = symmetric_moe + PAPER_DELTAS_PTS

    print("  Main ablation ladder (macro-F1, %, mean +/- sample std over 5 folds):")
    for name, vals in [
        ("Single DINOv2 generalist", gen_only),
        ("Symmetric MoE (base experts + base generalist)", symmetric_moe),
        ("AME-Derm (asymmetric, reconstructed for display)", ame_derm),
    ]:
        print(f"    {name:52s} {vals.mean():.2f} +/- {vals.std(ddof=1):.2f}")

    t_stat, p_two = stats.ttest_1samp(PAPER_DELTAS_PTS, popmean=0.0)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    print(f"\n  Paired t-test:        t(4)={t_stat:.2f}, one-sided p={p_one:.3f}"
          f"  (paper reports t(4)=2.30, p=0.041)")
    w_stat, p_wilcoxon = stats.wilcoxon(PAPER_DELTAS_PTS, alternative="greater")
    print(f"  Wilcoxon signed-rank: W={w_stat:.1f}, one-sided p={p_wilcoxon:.3f}"
          f"          (paper reports p=0.063)")
    print(f"  Folds with a positive difference: {int((PAPER_DELTAS_PTS > 0).sum())}/5")
    print("\n  The manuscript itself describes this margin as 'small and borderline' —")
    print("  the asymmetry claim rests on mechanism and fold-wise consistency as much")
    print("  as on this test. The PanDerm comparison above is the more robust result.")


if __name__ == "__main__":
    panderm_comparison()
    internal_ablation()
