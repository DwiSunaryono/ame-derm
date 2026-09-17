"""
A TRAIN-VAL split whose VAL mirrors the metadata (not only the class) --
user idea: if VAL is made statistically close to TEST (over class AND
metadata: age/sex/location/source-dataset) and then LOCKED (never altered
during adaptive training), raising VAL will more genuinely raise TEST --
because VAL is not just a random sample that happens to represent, but is
deliberately constructed to represent.

DIFFERENT from bf.stratified_ham_kfold: the OUTER split (which samples
become TEST) stays IDENTICAL (stratification still uses only the label,
EXACTLY as before) -- so no TEST fold changes at all (the 20+ historical
figures in percobaan.md remain comparable apple-to-apple). What CHANGES is
only the INNER split (train vs val out of the 80% pool): stratification now
uses a COMBINATION (label + chosen metadata fields, default 'dataset' -- the
field with the strongest and most numerous imbalance correlations from the
Phase 8 diagnosis), instead of the label alone.

Per-lesion consistency: age/sex/location/dataset were verified to be ALMOST
ALWAYS consistent within one lesion_id (0-9 out of 7470 lesions show small
variation, ignored) -- so they are safe to use as per-lesion stratification
keys.

Usage (called from other scripts, not a CLI):
    from ame_derm.data_splits.representative_val_split import stratified_ham_kfold_representative_val
    for fold_i, train_idx, val_idx, test_idx in stratified_ham_kfold_representative_val(
            ham_df, cfg, val_stratify_fields=["dataset"]):
        ...
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def stratified_ham_kfold_representative_val(df, cfg, val_stratify_fields=("dataset",), n_splits=5):
    idx = np.arange(len(df))
    labels = df["label"].values
    groups = df["lesion_id"].values

    combined_key = df["label"].astype(str)
    for f in val_stratify_fields:
        combined_key = combined_key + "_" + df[f].astype(str)
    combined_codes = pd.factorize(combined_key)[0]

    # Outer split (the one that fixes TEST) still uses ONLY the label --
    # identical to bf.stratified_ham_kfold, so no TEST fold ever changes.
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    for fold_i, (train_val_idx, test_idx) in enumerate(sgkf.split(idx, labels, groups)):
        sub_groups = groups[train_val_idx]
        sub_combined = combined_codes[train_val_idx]
        sgkf_val = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=cfg.seed)
        train_sub_idx, val_sub_idx = next(
            sgkf_val.split(np.arange(len(sub_combined)), sub_combined, sub_groups)
        )
        train_idx = train_val_idx[train_sub_idx]
        val_idx = train_val_idx[val_sub_idx]
        assert set(groups[train_idx]).isdisjoint(set(groups[val_idx])), "leakage lesion train/val!"
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), "leakage lesion train/test!"
        assert set(groups[val_idx]).isdisjoint(set(groups[test_idx])), "leakage lesion val/test!"
        yield fold_i, train_idx, val_idx, test_idx


def stratified_ham_kfold_fully_representative(df, cfg, stratify_fields=("dataset",), n_splits=5, n_repeats=1, seed_offset=0):
    """NEW EVALUATION FOUNDATION (different from stratified_ham_kfold_representative_val
    above): both the outer split (which fixes TEST) AND the inner split
    (which fixes VAL) are stratified on label+stratify_fields -- so TEST &
    VAL both become genuinely representative and resemble each other, rather
    than only VAL being "forced" representative while TEST is arbitrary.

    THIS ALTERS TEST IDENTITY (different images from bf.stratified_ham_kfold
    or stratified_ham_kfold_representative_val) -- DELIBERATELY, following an
    explicit user decision after finding VAL could swing 0.78->0.95 purely
    from individual-lesion sampling variance (not metadata imbalance) --
    percobaan.md (the old TEST) becomes the Phase 1-8 archive, while the new
    evaluation onward uses this split foundation, tracked separately.

    n_repeats>1: RepeatedStratifiedGroupKFold style -- repeat the WHOLE split
    process with a different random_state per repeat (seed=cfg.seed+
    seed_offset+repeat_i), to damp variance from the CHANCE grouping of
    lesions (newly shown to be the dominant variance source, larger than
    metadata imbalance alone). Default n_repeats=1 (cheap, as before) --
    raise it ONLY when a more stable estimate is needed (training cost
    multiplies by n_repeats x n_splits total runs).

    Yields: (repeat_i, fold_i, train_idx, val_idx, test_idx)
    """
    idx = np.arange(len(df))
    groups = df["lesion_id"].values

    combined_key = df["label"].astype(str)
    for f in stratify_fields:
        combined_key = combined_key + "_" + df[f].astype(str)
    combined_codes = pd.factorize(combined_key)[0]

    for repeat_i in range(n_repeats):
        seed = cfg.seed + seed_offset + repeat_i
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_i, (train_val_idx, test_idx) in enumerate(sgkf.split(idx, combined_codes, groups)):
            sub_groups = groups[train_val_idx]
            sub_combined = combined_codes[train_val_idx]
            sgkf_val = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
            train_sub_idx, val_sub_idx = next(
                sgkf_val.split(np.arange(len(sub_combined)), sub_combined, sub_groups)
            )
            train_idx = train_val_idx[train_sub_idx]
            val_idx = train_val_idx[val_sub_idx]
            assert set(groups[train_idx]).isdisjoint(set(groups[val_idx])), "leakage lesion train/val!"
            assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), "leakage lesion train/test!"
            assert set(groups[val_idx]).isdisjoint(set(groups[test_idx])), "leakage lesion val/test!"
            yield repeat_i, fold_i, train_idx, val_idx, test_idx


def compare_val_to_reference(val_df, reference_df, fields, label_col="label"):
    """Check statistics: how close the VAL distribution comes to the
    reference distribution (usually TEST or the pool) over the class and
    each metadata field. Returns a dict holding the distance (mean abs diff
    of proportions) per field -- smaller means more representative."""
    results = {}
    for f in [label_col] + list(fields):
        val_dist = val_df[f].value_counts(normalize=True)
        ref_dist = reference_df[f].value_counts(normalize=True)
        all_vals = set(val_dist.index) | set(ref_dist.index)
        diffs = [abs(val_dist.get(v, 0) - ref_dist.get(v, 0)) for v in all_vals]
        results[f] = {"mean_abs_diff": float(np.mean(diffs)), "max_abs_diff": float(np.max(diffs))}
    return results


if __name__ == "__main__":
    from ame_derm.common import birnet_final as bf
    from ame_derm.data_splits.diagnose_metadata_imbalance import detect_metadata_fields

    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    ham_df = bf.build_ham10000_dataframe(paths)

    bucketed, field_names = detect_metadata_fields(ham_df, n_bins_numeric=9)
    overlap = [c for c in bucketed.columns if c in ham_df.columns]
    ham_meta = pd.concat([ham_df.drop(columns=overlap), bucketed], axis=1)

    old_test_idx, old_val_idx = None, None
    print("=== SPLIT LAMA (bf.stratified_ham_kfold, VAL stratify=label saja) ===")
    for fold_i, train_idx, val_idx, test_idx in bf.stratified_ham_kfold(ham_meta, cfg, n_splits=5):
        if fold_i != 0:
            continue
        old_test_idx, old_val_idx = set(test_idx), set(val_idx)
        val_df_old = ham_meta.iloc[val_idx]
        test_df = ham_meta.iloc[test_idx]
        print(f"  n_val={len(val_df_old)} n_test={len(test_df)}")
        cmp_old = compare_val_to_reference(val_df_old, test_df, field_names)
        for f, v in cmp_old.items():
            print(f"  {f}: mean_abs_diff={v['mean_abs_diff']:.4f} max_abs_diff={v['max_abs_diff']:.4f}")
        break

    print("\n=== SPLIT BARU (representative, VAL stratify=label+dataset) ===")
    for fold_i, train_idx, val_idx, test_idx in stratified_ham_kfold_representative_val(
            ham_meta, cfg, val_stratify_fields=["dataset"], n_splits=5):
        if fold_i != 0:
            continue
        print(f"  TEST identik dgn split lama? {set(test_idx) == old_test_idx}")
        print(f"  VAL identik dgn split lama? {set(val_idx) == old_val_idx} (HARUS beda -- itu tujuannya)")
        val_df_new = ham_meta.iloc[val_idx]
        test_df = ham_meta.iloc[test_idx]
        print(f"  n_val={len(val_df_new)} n_test={len(test_df)}")
        cmp_new = compare_val_to_reference(val_df_new, test_df, field_names)
        for f, v in cmp_new.items():
            print(f"  {f}: mean_abs_diff={v['mean_abs_diff']:.4f} max_abs_diff={v['max_abs_diff']:.4f}")
        break
