"""Corrected HAM10000 split: stratified by diagnostic class AND grouped by
lesion_id, so no lesion appears in more than one of train/val/test.

Context: the split used throughout this project (birnet_final.py's
split_stratified, and the paper's Section 2.4.1) is stratified by class but
NOT grouped by lesion_id. HAM10000 has 7,470 unique lesions across 10,015
images (1,956 lesions contribute more than one image, up to 6). Verification
against the public metadata showed 34.8% of test-split images and 36.1% of
val-split images share a lesion_id with a training-split image -- i.e. the
model can see another photo of the same physical lesion during training and
then be evaluated on a near-duplicate of it. This is a well-documented
HAM10000 pitfall and is not accounted for in any classification, ensemble,
or NCL result currently reported in the paper.

This script produces the corrected split and writes it to a CSV so every
downstream script (train_classifier_generic.py, birnet_joint_ncl_finetune.py,
birnet_fusion_v2.py/v3_clinical.py/positive_control.py,
birnet_ensemble_*.py) can be re-pointed at it for a clean re-run. It changes
nothing by itself -- no model is retrained here.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
METADATA_PATH = "BIRNet_Research/dataset/ham10000/HAM10000_classification/HAM10000_metadata.csv"
OUTPUT_PATH = "BIRNet_Research/dataset/ham10000/HAM10000_classification/HAM10000_split_lesion_grouped.csv"


def build_split(df: pd.DataFrame) -> pd.DataFrame:
    labels = df["dx"].values
    groups = df["lesion_id"].values
    idx = np.arange(len(labels))

    # Stage 1: 1/7 fold ~= 14.3% held out as test, grouped by lesion, stratified by class.
    sgkf1 = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    trainval_idx, test_idx = next(sgkf1.split(idx, labels, groups))

    # Stage 2: from the remaining ~85.7%, hold out 1/6 (~16.7% of the remainder,
    # ~14.3% of the original total) as validation, same constraints.
    sub_labels = labels[trainval_idx]
    sub_groups = groups[trainval_idx]
    sgkf2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
    train_sub_idx, val_sub_idx = next(
        sgkf2.split(np.arange(len(sub_labels)), sub_labels, sub_groups)
    )
    train_idx = trainval_idx[train_sub_idx]
    val_idx = trainval_idx[val_sub_idx]

    split = np.empty(len(df), dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"
    out = df.copy()
    out["split"] = split
    return out


def verify_no_leakage(out: pd.DataFrame) -> None:
    groups_by_split = {
        s: set(out.loc[out["split"] == s, "lesion_id"]) for s in ("train", "val", "test")
    }
    overlaps = {
        "train&val": groups_by_split["train"] & groups_by_split["val"],
        "train&test": groups_by_split["train"] & groups_by_split["test"],
        "val&test": groups_by_split["val"] & groups_by_split["test"],
    }
    for name, ov in overlaps.items():
        assert len(ov) == 0, f"Lesion leakage detected in {name}: {len(ov)} lesions"
    print("Verified: zero lesion_id overlap across train/val/test.")


def main():
    df = pd.read_csv(METADATA_PATH)
    out = build_split(df)
    verify_no_leakage(out)

    n = len(out)
    for s in ("train", "val", "test"):
        sub = out[out["split"] == s]
        print(f"{s}: {len(sub)} images ({100*len(sub)/n:.1f}%), "
              f"{sub['lesion_id'].nunique()} unique lesions")
        print("  class distribution:", sub["dx"].value_counts().to_dict())

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWritten: {OUTPUT_PATH}")
    print("Columns: all original metadata columns + 'split' (train/val/test).")
    print("Downstream scripts should load this file and filter by the 'split' "
          "column instead of recomputing train_test_split themselves.")


if __name__ == "__main__":
    main()
