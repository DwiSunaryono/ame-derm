"""Round 2 of the boundary-injection strategy: retrain with the
GENERALIZED, QPSO-optimized synthetic set (953 accepted samples, ALL 7
classes, BOTH "confident wrong" and "unconfident correct" categories) --
unlike round 1 (train_classifier_phase2_boundary_aug.py), which only
pushed MEL samples away from NV and, per the ledger (round 1 entry),
caused NV->MEL confusion to WORSEN (+24) as an uncontrolled side effect
because the other direction was never reinforced. This round adds
synthetic samples for BOTH directions of every confused pair simultaneously
(e.g. NV-true-confused-as-MEL samples are also pushed toward genuine NV),
which should control the two-sided trade-off the ledger flagged.

Same leakage safeguard as round 1: a synthetic image is added to a given
main fold's TRAINING data only if its source image is in THAT fold's own
train_df (never its val_df or test_df).

Usage:
    python train_classifier_phase2_boundary_aug_v2.py --backbone swin_base_patch4_window7_224
"""
import argparse
import os
import numpy as np
import pandas as pd

from ame_derm.common import birnet_final as bf
from ame_derm.experts.train_classifier_phase2_final import CROP_DIR
from ame_derm.experts import train_classifier_phase2_mixup as m
from ame_derm.experts.train_classifier_phase2_mixup import run_fold

MANIFEST_CSV = "BIRNet_Research/outputs/synthetic_boundary_generalized_qpso_manifest.csv"


def build_augmented_train_df(train_df, manifest_df, ham_df):
    accepted = manifest_df[manifest_df["accepted"]]
    train_source_ids = set(train_df["image_id"])
    usable = accepted[accepted["source_image_id"].isin(train_source_ids)]

    lesion_map = dict(zip(ham_df["image_id"], ham_df["lesion_id"]))
    extra_rows = []
    for _, r in usable.iterrows():
        base = ham_df[ham_df["image_id"] == r["source_image_id"]].iloc[0].to_dict()
        base["image_id"] = r["synth_image_id"]
        base["label"] = int(r["synth_label"])
        base["lesion_id"] = f"{lesion_map[r['source_image_id']]}_synthC"
        extra_rows.append(base)

    if not extra_rows:
        return train_df, 0
    extra_df = pd.DataFrame(extra_rows)[train_df.columns]
    augmented = pd.concat([train_df, extra_df], ignore_index=True)
    return augmented, len(extra_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="swin_base_patch4_window7_224")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()

    m.CKPT_DIR = "BIRNet_Research/outputs/checkpoints/phase2_boundary_aug_v2"

    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    ham_df = bf.build_ham10000_dataframe(paths)
    has_crop = ham_df["image_id"].apply(lambda iid: os.path.exists(os.path.join(CROP_DIR, f"{iid}.jpg")))
    ham_df = ham_df[has_crop].reset_index(drop=True)
    print(f"{len(ham_df)} gambar dgn crop tersedia (real, split baseline tidak berubah)")

    manifest_df = pd.read_csv(MANIFEST_CSV)
    print(f"Manifest generalized QPSO: {manifest_df['accepted'].sum()}/{len(manifest_df)} diterima QC")

    results = []
    for fold_i, train_idx, val_idx, test_idx in bf.stratified_ham_kfold(ham_df, cfg, n_splits=args.n_splits):
        if args.fold is not None and fold_i != args.fold:
            continue
        train_df = ham_df.iloc[train_idx].reset_index(drop=True)
        val_df = ham_df.iloc[val_idx].reset_index(drop=True)
        test_df = ham_df.iloc[test_idx].reset_index(drop=True)

        aug_train_df, n_added = build_augmented_train_df(train_df, manifest_df, ham_df)
        print(f"[FOLD {fold_i}] train asli={len(train_df)} + sintetis (source ada di TRAIN fold ini)={n_added} "
              f"= {len(aug_train_df)}")

        result = run_fold(fold_i, aug_train_df, val_df, test_df, args.backbone, args.seed, cfg)
        results.append(result)
        print(f"RESULT_SUMMARY backbone={args.backbone} variant=boundary_aug_v2 fold={fold_i} "
              f"n_synthetic_added={n_added} test_acc={result['test_acc']:.4f} "
              f"test_bal={result['test_bal']:.4f} test_f1={result['test_f1']:.4f}")

    if len(results) > 1:
        f1s = np.array([r["test_f1"] for r in results])
        print(f"\n=== RINGKASAN {len(results)}-FOLD BOUNDARY-AUG V2 (backbone={args.backbone}) ===")
        print(f"f1_macro: mean={f1s.mean():.4f} std={f1s.std():.4f}")
        print("Baseline Swin+MixUp (tanpa sintetis): mean=0.7902 std=0.0146")
        print("Round 1 (MEL->NV satu arah saja): mean=0.7815 std=0.0237")
        print(f"RESULT_SUMMARY_FINAL backbone={args.backbone} variant=boundary_aug_v2 n_folds={len(results)} "
              f"mean_f1={f1s.mean():.4f} std_f1={f1s.std():.4f}")


if __name__ == "__main__":
    main()
