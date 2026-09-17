#!/usr/bin/env python3
"""
Export the champion's STRICT 5-fold (fully_representative + lesion-grouped,
stratify label+dataset) into one CSV per fold, so PanDerm & DINOv2 can be
pitched against folds that are IDENTICAL to the MoE champion's (strict
protocol, not their own label-only single split).

Rebuilds the df + split EXACTLY as birnet_eval_hybrid_council.py does.
Output: <out>/rep_fold{0..4}.csv with columns [image,label,split,localization,lesion_id,image_id]
  - 'image' = <image_id>.jpg  (works for full-image ISIC2018/ as well as crop)
  - 'split' = train/val/test for that fold
Compatible with PanDerm run_class_finetuning (--csv_path) & birnet_dinov2_official_protocol.py (--csv).
"""
import os, argparse
import numpy as np, pandas as pd

from ame_derm.common import birnet_final as bf
from ame_derm.experts.train_specialist_expert import set_active_meta_fields, detect_metadata_fields
from ame_derm.experts.train_classifier_phase2_final import CROP_DIR
from ame_derm.data_splits.representative_val_split import stratified_ham_kfold_fully_representative


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="BIRNet_Research/outputs/rep_folds")
    ap.add_argument("--n_splits", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    set_active_meta_fields("localization")
    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    ham_df = bf.build_ham10000_dataframe(paths)
    # keep only images with a crop (same as champion) -> identical sample set
    has = ham_df["image_id"].apply(lambda i: os.path.exists(os.path.join(CROP_DIR, f"{i}.jpg")))
    ham_df = ham_df[has].reset_index(drop=True)
    print(f"ham_df (crop-exist filter): {len(ham_df)} sampel, {ham_df['lesion_id'].nunique()} lesi, "
          f"{ham_df['label'].nunique()} kelas")

    bucketed, _ = detect_metadata_fields(ham_df, 9)
    ov = [c for c in bucketed.columns if c in ham_df.columns]
    hm = pd.concat([ham_df.drop(columns=ov), bucketed], axis=1)

    for _, fi, tr, va, te in stratified_ham_kfold_fully_representative(
            hm, cfg, ["dataset"], n_splits=args.n_splits, n_repeats=1):
        df = ham_df.copy()
        df["split"] = "train"
        df.loc[va, "split"] = "val"
        df.loc[te, "split"] = "test"
        # only rows belonging to this fold (train=tr, val=va, test=te) -> all ham_df rows included (rest as train)
        out = df.loc[np.concatenate([tr, va, te])].copy()
        out["image"] = out["image_id"].astype(str) + ".jpg"
        cols = ["image", "label", "split", "localization", "lesion_id", "image_id"]
        out = out[cols].reset_index(drop=True)
        # confirm no lesion leakage between splits (double-check)
        s = {k: set(out[out.split == k]["lesion_id"]) for k in ["train", "val", "test"]}
        assert s["train"].isdisjoint(s["val"]) and s["train"].isdisjoint(s["test"]) and s["val"].isdisjoint(s["test"])
        path = os.path.join(args.out, f"rep_fold{fi}.csv")
        out.to_csv(path, index=False)
        vc = out.groupby("split").size().to_dict()
        print(f"  fold{fi}: {vc} -> {path}")

    print(f"\nSelesai. 5 fold CSV ketat di: {args.out}")


if __name__ == "__main__":
    main()
