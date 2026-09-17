#!/usr/bin/env python3
"""
Ekspor 5-fold KETAT champion (fully_representative + lesion-grouped, stratify
label+dataset) ke CSV per-fold, supaya PanDerm & DINOv2 bisa diadu di fold yang
IDENTIK dengan MoE champion (protokol ketat, bukan single-split label-only mereka).

Mereproduksi konstruksi df + split PERSIS seperti birnet_eval_hybrid_council.py.
Output: <out>/rep_fold{0..4}.csv dgn kolom [image,label,split,localization,lesion_id,image_id]
  - 'image' = <image_id>.jpg  (cocok utk full-image ISIC2018/ maupun crop)
  - 'split' = train/val/test utk fold itu
Kompatibel dgn PanDerm run_class_finetuning (--csv_path) & birnet_dinov2_official_protocol.py (--csv).
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
    # filter ke gambar yg punya crop (identik dgn champion) -> set sampel sama
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
        # hanya baris yg ikut fold ini (train=tr, val=va, test=te) -> semua ham_df ikut (train sisanya)
        out = df.loc[np.concatenate([tr, va, te])].copy()
        out["image"] = out["image_id"].astype(str) + ".jpg"
        cols = ["image", "label", "split", "localization", "lesion_id", "image_id"]
        out = out[cols].reset_index(drop=True)
        # verifikasi tak ada leakage lesi antar-split (double-check)
        s = {k: set(out[out.split == k]["lesion_id"]) for k in ["train", "val", "test"]}
        assert s["train"].isdisjoint(s["val"]) and s["train"].isdisjoint(s["test"]) and s["val"].isdisjoint(s["test"])
        path = os.path.join(args.out, f"rep_fold{fi}.csv")
        out.to_csv(path, index=False)
        vc = out.groupby("split").size().to_dict()
        print(f"  fold{fi}: {vc} -> {path}")

    print(f"\nSelesai. 5 fold CSV ketat di: {args.out}")


if __name__ == "__main__":
    main()
