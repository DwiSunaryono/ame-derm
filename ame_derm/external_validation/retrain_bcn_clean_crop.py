#!/usr/bin/env python3
"""STRATEGI 4C — RETRAIN BCN dengan CROP BERSIH (tanpa segmen dilatih-HAM).

Motivasi: crop yg dipakai champion (cropped_phase1_final) dihasilkan pipeline
segmentasi Phase1 yg DILATIH DI HAM10000, lalu diterapkan ke gambar BCN
(domain shift). Karena itu tidak jelas apakah degradasi DF-BCN berasal dari
arkeitiktur model atau dari segmen yg kurang cocok di domain BCN.

Eksperimen ini mengganti crop BCN dengan 2 varian "bersih" yg TIDAK bergantung
pada model yang dilatih di HAM:
  1. center  : center-crop simetris (asumsi lesi terpusat di dermoskopi).
  2. otsu    : bbox lesi via Otsu thresholding + komponen terbesar + pad 15%
               (unsupervised, tanpa training model domain manapun).
  3. raw     : kontrol — gambar BCN asli (referensi sebelumnya).
  4. crop    : kontrol — crop champion (segmen HAM-domain).

Semua varian di-retrain identik (gen-only, full retrain tanpa freeze, resep
champion: DINOv2 base, meta_dim=15 localization, MixUp+CutMix, SupCon,
inverse-freq + oversample, AdamW 3-groups, warmup+cosine, patience 12),
evaluasi di clean lesion-grouped test (makro-7, TTA 6-view).

Ckpt tiap varian disimpan terpisah (suffix mode), raw/crop TIDAK tertimpa.

Usage (user menjalankan sendiri, satu varian per run):
  .venv/bin/python3 birnet_retrain_bcn_clean_crop.py --mode center --out birnet_retrain_bcn_center.json
  .venv/bin/python3 birnet_retrain_bcn_clean_crop.py --mode otsu   --out birnet_retrain_bcn_otsu.json
  .venv/bin/python3 birnet_retrain_bcn_clean_crop.py --mode raw    --out birnet_retrain_bcn_raw2.json
  .venv/bin/python3 birnet_retrain_bcn_clean_crop.py --mode crop   --out birnet_retrain_bcn_crop2.json
"""
import argparse
import os
import json
import numpy as np
import cv2

from ame_derm.common import birnet_final as bf
from ame_derm.external_validation.retrain_bcn_lpft import (
    build_bcn_df, lesion_grouped_split, make_loader, train_retrained,
    EXPERT_MAPPING, BACKBONE,
)
from ame_derm.experts.train_specialist_expert import set_active_meta_fields

CACHES = {
    "center": "BIRNet_Research/outputs/crops_bcn_center",
    "otsu": "BIRNet_Research/outputs/crops_bcn_otsu",
}
CKPT_DIR = "BIRNet_Research/outputs/checkpoints/retrain_bcn_clean"


def center_crop(img, frac=0.72):
    h, w = img.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def otsu_bbox_crop(img, pad_frac=0.15):
    """bbox lesi via Otsu + largest connected component (tanpa model)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # background (kulit terang) biasanya > harmoni mask; lesi sering daerah gelap
    if th.mean() > 127:
        th = cv2.bitwise_not(th)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(th, 8)
    if n <= 1:
        return img
    # ambil komponen terbesar yang bukan seluruh frame
    best, best_area = None, 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area > best_area:
            best_area = area
            best = (x, y, w, h)
    if best is None:
        return img
    x, y, w, h = best
    py, px = int(h * pad_frac), int(w * pad_frac)
    x0, x1 = max(0, x - px), min(img.shape[1], x + w + px)
    y0, y1 = max(0, y - py), min(img.shape[0], y + h + py)
    if y1 <= y0 or x1 <= x0:
        return img
    return img[y0:y1, x0:x1]


def build_clean_df(mode):
    df = build_bcn_df("raw")
    if mode in ("raw", "crop"):
        return build_bcn_df(mode)
    cache = CACHES[mode]
    os.makedirs(cache, exist_ok=True)
    fn = center_crop if mode == "center" else otsu_bbox_crop
    paths = {}
    for iid in df["image_id"]:
        out = os.path.join(cache, f"{iid}.jpg")
        if not os.path.exists(out):
            src = df[df["image_id"] == iid]["img_path"].iloc[0]
            img = cv2.imread(src)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(out, cv2.cvtColor(fn(img), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths[iid] = out
    df = df.copy()
    df["img_path"] = df["image_id"].map(paths)
    return df


def ckpt_suffix(mode):
    return f"_clean_{mode}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["raw", "crop", "center", "otsu"], default="center")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lambda_supcon", type=float, default=0.1)
    ap.add_argument("--out", default="birnet_retrain_bcn_clean.json")
    args = ap.parse_args()

    set_active_meta_fields("localization")
    cfg = bf.Config()
    os.makedirs(CKPT_DIR, exist_ok=True)

    bcn = build_clean_df(args.mode)
    print(f"[BCN {args.mode}] terlabel={len(bcn)} img non-null=",
          bcn["img_path"].notna().sum())
    train_df, val_df, test_df = lesion_grouped_split(bcn, args.seed)
    print(f"[lesion-grouped split seed={args.seed}] "
          f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # ckpt gen-only; suffix mode agar tidak tertimpa
    ckpt = os.path.join(CKPT_DIR,
                        f"cls_{BACKBONE}_multimodal_fold0_clean_{args.mode}_best.pth")

    res = train_retrained(train_df, val_df, test_df, f"GEN[{args.mode}]",
                          ckpt, cfg, args)
    results = {"gen": res, "_config": {
        "mode": args.mode, "seed": args.seed, "max_epochs": args.max_epochs,
        "patience": args.patience, "lambda_supcon": args.lambda_supcon}}
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nHasil disimpan: {args.out}\n{json.dumps(res, indent=2)}")


if __name__ == "__main__":
    main()