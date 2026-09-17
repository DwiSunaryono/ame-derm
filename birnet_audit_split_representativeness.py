"""Audit representativeness of the test partitions (Jalur A, CPU-only).

For each of the 5 folds of the lesion-grouped, fully-representative protocol
(stratified_ham_kfold_fully_representative, stratify_fields=["dataset"]),
compare the *test* partition's metadata distribution against the full
dataset's distribution for every metadata attribute of HAM10000:
  - label (class)       -- explicit stratification key (exact by construction)
  - dataset (source)    -- explicit stratification key (exact by construction)
  - sex, localization, age_bucket (quantile-binned age)
       -- NOT explicit stratification keys -> measure the induced shift

Metric per attribute (same logic as compare_val_to_reference in
birnet_representative_val_split.py):
  mean_abs_diff / max_abs_diff of category proportions (test vs full), and
  coverage = fraction of the full attribute's value-set that appears in the
  test fold (guarantee that every metadata value is present in the test).

The same statistics are reported for the validation partition, since the
inner val split uses the identical stratification keys.

The reported numbers feed a small table / Fig. 3 panel for the manuscript:
keeping the headline claim at class x source, and showing that the
remaining metadata attributes are also mirrored (induced representativeness).

Usage (CPU only, no GPU needed):
    .venv/bin/python3 birnet_audit_split_representativeness.py
"""
import json
import os
import numpy as np
import pandas as pd

import birnet_final as bf
from birnet_diagnose_metadata_imbalance import detect_metadata_fields
from birnet_representative_val_split import stratified_ham_kfold_fully_representative

CROP_DIR = "BIRNet_Research/outputs/cropped_phase1_final"
SEED = 42

STRATIFY_FIELDS = ["dataset"]
FIELDS = ("label", "dataset", "sex", "localization", "age_bucket")


def prop_shift(test_dist, full_dist, all_vals):
    """Mean and max absolute difference of category proportions."""
    diffs = [abs(test_dist.get(v, 0.0) - full_dist.get(v, 0.0)) for v in all_vals]
    return float(np.mean(diffs)), float(np.max(diffs)) if diffs else 0.0


def audit_fold(fold_df, full_df):
    """Return per-attribute distribution shift + coverage of one partition
    (test or val) against the full dataset."""
    out = {}
    for f in FIELDS:
        full_dist = full_df[f].value_counts(normalize=True).to_dict()
        fold_dist = fold_df[f].value_counts(normalize=True).to_dict()
        all_vals = sorted(set(full_dist.keys()) | set(fold_dist.keys()))
        mean_abs, max_abs = prop_shift(fold_dist, full_dist, all_vals)
        coverage = len([v for v in full_dist if v in fold_dist]) / max(len(full_dist), 1)
        out[f] = {
            "mean_abs_diff": round(mean_abs, 4),
            "max_abs_diff": round(max_abs, 4),
            "coverage": round(float(coverage), 4),
            "n_categories": len(all_vals),
            "n_in_fold": int(len(fold_df)),
        }
    return out


def main():
    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    ham_df = bf.build_ham10000_dataframe(paths)
    has_crop = ham_df["image_id"].apply(lambda iid: os.path.exists(os.path.join(CROP_DIR, f"{iid}.jpg")))
    ham_df = ham_df[has_crop].reset_index(drop=True)

    bucketed, field_names = detect_metadata_fields(ham_df, n_bins_numeric=9)
    overlap = [c for c in bucketed.columns if c in ham_df.columns]
    ham_meta = pd.concat([ham_df.drop(columns=overlap), bucketed], axis=1)
    n_total = len(ham_meta)
    print(f"Sampel dgn crop tersedia: {n_total}")

    results = {"seed": SEED, "stratify_fields": list(STRATIFY_FIELDS), "n_total": n_total,
               "fields": list(FIELDS), "folds": {}}

    for repeat_i, fold_i, train_idx, val_idx, test_idx in stratified_ham_kfold_fully_representative(
            ham_meta, cfg, STRATIFY_FIELDS, n_splits=5, n_repeats=1):
        full_df = ham_meta
        test_df = ham_meta.iloc[test_idx]
        val_df = ham_meta.iloc[val_idx]
        test_audit = audit_fold(test_df, full_df)
        val_audit = audit_fold(val_df, full_df)
        results["folds"][str(fold_i)] = {
            "test": test_audit,
            "val": val_audit,
        }

        print(f"\n=== FOLD {fold_i} | n_test={len(test_df)} n_val={len(val_df)} ===")
        print(f"  {'attribute':<14}{'mean|Δ|':>10}{'max|Δ|':>10}   coverage(test)   coverage(val)")
        for f in FIELDS:
            t = test_audit[f]
            v = val_audit[f]
            print(f"  {f:<14}{t['mean_abs_diff']:>10.4f}{t['max_abs_diff']:>10.4f}"
                  f"   {t['coverage']:>10.4f}   {v['coverage']:>10.4f}")

    print("\n=== RATA-RATA LIMA FOLD (test partition) ===")
    print(f"  {'attribute':<14}{'mean|Δ|':>10}{'max|Δ|':>9}   {'coverage':>10}")
    agg = {}
    fold_keys = sorted(results["folds"].keys())
    for f in FIELDS:
        means = [results["folds"][k]["test"][f]["mean_abs_diff"] for k in fold_keys]
        maxes = [results["folds"][k]["test"][f]["max_abs_diff"] for k in fold_keys]
        covs = [results["folds"][k]["test"][f]["coverage"] for k in fold_keys]
        agg[f] = {"mean_abs_diff": round(float(np.mean(means)), 4),
                  "max_abs_diff": round(float(np.max(maxes)), 4),
                  "coverage": round(float(np.mean(covs)), 4)}
        print(f"  {f:<14}{agg[f]['mean_abs_diff']:>10.4f}{agg[f]['max_abs_diff']:>9.4f}"
              f"   {agg[f]['coverage']:>10.4f}")
    results["summary"] = agg

    out_json = "birnet_audit_split_representativeness.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDisimpan ke {out_json}")


if __name__ == "__main__":
    main()