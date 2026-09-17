# Results

Raw per-fold and per-sample outputs behind several of the paper's numbers,
plus a script that turns them into CSV tables. No training, no GPU, no
re-running the pipeline — just arithmetic over data that's already here.

## Contents

| Path | Description |
|---|---|
| [`data/panderm_baseline/generalist_fold*_seed42_test.csv`](data/panderm_baseline) | **Headline result's raw data.** PanDerm's own per-sample test predictions (`true_label`/`predicted_label`), 5 folds, seed 42 — copied unmodified from the vendored PanDerm repo's `results/rigorous_5fold/`. |
| [`data/panderm_baseline/generalist_fold*_seed122_test.csv`](data/panderm_baseline) | Same, at a second seed (122) — backs the paper's seed-robustness claim (§"Comparison with a Dermatology Foundation Model"). |
| [`data/panderm_baseline/moe_fold*_seed42.json`](data/panderm_baseline) | Per-fold summary (`gen_macro_f1`, `moe_macro_f1`) from `baselines.panderm_comparison.panderm_moe_eval` — the PanDerm-backboned MoE cell of the 2×2 table. |
| [`data/moe_champion_5fold_results.json`](data/moe_champion_5fold_results.json) | Raw output of `ame_derm.moe_fusion.eval_moe_champion_5fold`: per-fold macro-F1 for the single generalist and the symmetric (base-expert) static MoE. |
| [`data/split_audit.json`](data/split_audit.json) | Raw output of `ame_derm.data_splits.audit_split_representativeness`: per-fold, per-field (label/dataset/sex/localization/age) distribution-gap and coverage statistics for the lesion-grouped 5-fold split. |
| [`data/bcn_external/*.json`](data/bcn_external) | Per-sample predictions (`preds`/`labs`) from nine independent BCN20000 external-validation runs (checkpoint `best` + generalist-only `gen` + seeds 7/123/314/555/777/888/999), produced by `ame_derm.external_validation.retrain_bcn` / `diagnose_bcn_f1`. |
| [`scripts/reproduce_tables.py`](scripts/reproduce_tables.py) | Regenerates everything in `tables/` from the raw data above. |
| [`tables/panderm_comparison.csv`](tables/panderm_comparison.csv) | **Headline table.** DINOv2 generalist vs. PanDerm generalist, per fold — both sides raw. |
| [`tables/panderm_seed_robustness.csv`](tables/panderm_seed_robustness.csv) | PanDerm mean±std at seeds 42 and 122 — confirms the ranking isn't a favorable-seed artifact. |
| [`tables/main_ablation_ladder.csv`](tables/main_ablation_ladder.csv) | Secondary/modest result: generalist vs. symmetric MoE vs. AME-Derm, macro-F1 mean ± std over 5 folds. |
| [`tables/external_validation_bcn.csv`](tables/external_validation_bcn.csv) | Macro-F1 on the BCN20000 test set, one row per run/seed. |
| [`tables/split_audit_summary.csv`](tables/split_audit_summary.csv) | Split-representativeness audit, averaged over folds. |

## Reproduce

```bash
cd results
python scripts/reproduce_tables.py
```

## What's raw and what isn't

**`panderm_comparison.csv` is raw on both sides.** PanDerm's macro-F1 is
computed here from its own per-sample predictions, and it lands on 75.34 —
matching the manuscript to four decimal places. This is also the paper's
strongest result: 5 of 5 folds positive, exact sign test p=0.031. Treat it
as the one to trust most.

**`moe_champion_5fold_results.json` is raw too.** Its `gen_only_f1` and
`champ_static_f1` columns are exactly what
`ame_derm.moe_fusion.eval_moe_champion_5fold` produced; `reproduce_tables.py`
just computes the mean and std, nothing is hand-typed.

**The AME-Derm row in `main_ablation_ladder.csv` is the one exception.** We
never recovered the asymmetric model's own per-fold macro-F1 from archived
logs, so that row is `champ_static_f1` plus the per-fold delta the
manuscript reports ({+1.33, +0.71, +0.73, +1.81, −0.37} points) — a
reconstruction, not a fresh measurement (see `../docs/SYSTEM_DOCUMENTATION.md`
for how this happened). This doesn't affect the significance test, though:
[`../evaluation/statistical_significance.py`](../evaluation/statistical_significance.py)
only needs that delta array, and running it reproduces the manuscript's
t(4)=2.30, p=0.041 (paired t-test) and p≈0.062 (Wilcoxon) exactly.

**`external_validation_bcn.csv` is raw as well** — a fresh
`sklearn.metrics.f1_score(labs, preds, average="macro")` over each run's
real per-sample predictions, not anything copied from the paper.
