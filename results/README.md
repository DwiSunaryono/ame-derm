# Results

Raw per-fold / per-sample outputs backing three of the paper's numeric
claims, plus a script that regenerates CSV tables from them deterministically
— no training, no GPU, no re-running the pipeline.

## Contents

| Path | Description |
|---|---|
| [`data/moe_champion_5fold_results.json`](data/moe_champion_5fold_results.json) | Raw output of `ame_derm.moe_fusion.eval_moe_champion_5fold`: per-fold macro-F1 for the single generalist and the symmetric (base-expert) static MoE. |
| [`data/split_audit.json`](data/split_audit.json) | Raw output of `ame_derm.data_splits.audit_split_representativeness`: per-fold, per-field (label/dataset/sex/localization/age) distribution-gap and coverage statistics for the lesion-grouped 5-fold split. |
| [`data/bcn_external/*.json`](data/bcn_external) | Per-sample predictions (`preds`/`labs`) from nine independent BCN20000 external-validation runs (checkpoint `best` + generalist-only `gen` + seeds 7/123/314/555/777/888/999), produced by `ame_derm.external_validation.retrain_bcn` / `diagnose_bcn_f1`. |
| [`scripts/reproduce_tables.py`](scripts/reproduce_tables.py) | Regenerates everything in `tables/` from the JSON above. |
| [`tables/main_ablation_ladder.csv`](tables/main_ablation_ladder.csv) | Generalist vs. symmetric MoE vs. AME-Derm, macro-F1 mean ± std over 5 folds. |
| [`tables/external_validation_bcn.csv`](tables/external_validation_bcn.csv) | Macro-F1 on the BCN20000 test set, one row per run/seed. |
| [`tables/split_audit_summary.csv`](tables/split_audit_summary.csv) | Split-representativeness audit, averaged over folds. |

## Reproduce

```bash
cd results
python scripts/reproduce_tables.py
```

## What this does and does not prove

- `gen_only_f1` / `champ_static_f1` in `moe_champion_5fold_results.json` are
  **raw, independently re-derivable** numbers (this repo's `reproduce_tables.py`
  computes their mean/std directly; nothing is hand-transcribed).
- The **AME-Derm** row of `main_ablation_ladder.csv` is `champ_static_f1` plus
  the per-fold delta reported in the manuscript's Results section
  ({+1.33, +0.71, +0.73, +1.81, -0.37} points) — the asymmetric model's own
  raw per-fold macro-F1 was not recovered from archived logs (see
  `../docs/SYSTEM_DOCUMENTATION.md`). The statistical-significance test in
  [`../evaluation/statistical_significance.py`](../evaluation/statistical_significance.py)
  only needs that delta array, so it is not affected by this — running it
  reproduces the manuscript's reported t(4)=2.30, p=0.041 (paired t-test) and
  p≈0.062 (Wilcoxon signed-rank) exactly.
- `external_validation_bcn.csv` is a genuine, from-scratch macro-F1
  computation (`sklearn.metrics.f1_score(labs, preds, average="macro")`) over
  each run's real per-sample predictions — not copied from the paper.
