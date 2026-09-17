#!/usr/bin/env python3
"""Reproduces the paper's "Statistical Significance" subsection (Results,
Section 3): a one-sided paired test of whether AME-Derm's asymmetric mixture
improves on the capacity-matched symmetric MoE, evaluated on the same five
lesion-grouped cross-validation folds.

Two of the three methods compared in the main ablation ladder (the single
DINOv2 generalist, `gen_only_f1`, and the symmetric MoE, `champ_static_f1`)
have their raw per-fold macro-F1 in `results/data/moe_champion_5fold_results
.json` (the direct output of `ame_derm.moe_fusion.eval_moe_champion_5fold`).
The asymmetric AME-Derm model's raw per-fold output was not recovered from
archived logs (see docs/SYSTEM_DOCUMENTATION.md); what IS available is the
manuscript's own reported per-fold difference (AME-Derm minus symmetric MoE,
in points): {+1.33, +0.71, +0.73, +1.81, -0.37}. A paired test only needs
that difference array, so this script tests it directly with
`scipy.stats.ttest_1samp` (mathematically identical to a paired t-test on
two fold-matched sequences) and reproduces the manuscript's reported
t(4)=2.30, p=0.041 (one-sided paired t) and p=0.063 (Wilcoxon signed-rank).

The reconstructed AME-Derm column in the ladder table this script also
prints (champ_static_f1 + delta, assuming the same fold ordering as the raw
JSON) is for display only; the significance test above does not depend on
that alignment assumption.
"""
import json
import os

import numpy as np
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "data")

# Per-fold difference (AME-Derm - symmetric MoE), macro-F1 points, as reported
# in the manuscript (Results, Section 3, "Statistical Significance").
PAPER_DELTAS_PTS = np.array([1.33, 0.71, 0.73, 1.81, -0.37])


def load_moe_5fold(path=None):
    path = path or os.path.join(DATA_DIR, "moe_champion_5fold_results.json")
    with open(path) as f:
        return json.load(f)


def main():
    folds = load_moe_5fold()
    gen_only = np.array([f["gen_only_f1"] for f in folds]) * 100
    symmetric_moe = np.array([f["champ_static_f1"] for f in folds]) * 100
    ame_derm_reconstructed = symmetric_moe + PAPER_DELTAS_PTS

    print("Main ablation ladder (macro-F1, %, mean +/- sample std across 5 folds)")
    print("-" * 70)
    for name, vals in [
        ("Single DINOv2 generalist", gen_only),
        ("Symmetric MoE (base experts + base generalist)", symmetric_moe),
        ("AME-Derm (asymmetric, reconstructed for display)", ame_derm_reconstructed),
    ]:
        print(f"  {name:52s} {vals.mean():.2f} +/- {vals.std(ddof=1):.2f}")

    print()
    print("Significance test: AME-Derm vs. symmetric MoE, H0: mean(delta) <= 0")
    print("-" * 70)
    t_stat, p_two_sided = stats.ttest_1samp(PAPER_DELTAS_PTS, popmean=0.0)
    p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
    print(f"  Paired t-test:        t(4)={t_stat:.2f}, one-sided p={p_one_sided:.3f}"
          f"  (paper reports t(4)=2.30, p=0.041)")

    w_stat, p_wilcoxon = stats.wilcoxon(PAPER_DELTAS_PTS, alternative="greater")
    print(f"  Wilcoxon signed-rank: W={w_stat:.1f}, one-sided p={p_wilcoxon:.3f}"
          f"          (paper reports p=0.063)")

    n_positive = int((PAPER_DELTAS_PTS > 0).sum())
    print(f"  Folds with a positive difference: {n_positive}/5")


if __name__ == "__main__":
    main()
