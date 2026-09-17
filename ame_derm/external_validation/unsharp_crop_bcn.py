#!/usr/bin/env python3
"""RAW-crop + unsharp mask untuk BCN20000.

Motivasi (percobaanReal2.md, konsensus 4-sumber Aug-15): pembeda akiec vs bcc =
WARNA + STRUKTUR HALUS (arborizing vessels, blue-gray globules, skuama, kontras).
Normalisasi "wp" MERATAKAN sinyal ini. Skrip ini justru MENONJOLKAN struktur
halus/tepi pada RAW crop (warna asli dipertahankan) via unsharp mask ringan.

Input : RAW Phase-1 crop  (CROP_DIR/{isic_id}.jpg)
Output: cache mode 'crop>unsharp' -> bcn_pipe/crop_unsharp/{nama_file_asli}
        (dinamai r['image'] agar terbaca build_bcn_df '>'-mode)

Usage:
  # preview 12 contoh (before/after), tak nulis cache:
  .venv/bin/python3 birnet_unsharp_crop_bcn.py --preview
  # proses penuh -> cache:
  .venv/bin/python3 birnet_unsharp_crop_bcn.py
  # lalu retrain:
  .venv/bin/python3 birnet_retrain_bcn.py --img_mode "crop>unsharp" --seed 42 \
      --split_seed 42 --gen_only --warm_start --rare_dup 6 \
      --out birnet_retrain_bcn_unsharp_r6_gen.json
"""
import argparse, os
import numpy as np
import cv2
import pandas as pd

import config

BCN_CSV = config.BCN_CSV
CROP_DIR = config.CROP_DIR
OUT_DIR = "BIRNet_Research/outputs/bcn_pipe/crop_unsharp"


def unsharp(img_bgr, amount=0.8, sigma=1.5):
    """Unsharp mask ringan: pertajam tepi/struktur halus, warna RAW dipertahankan.
    out = img + amount*(img - blur). amount kecil agar tak menambah noise berlebih."""
    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
    sharp = cv2.addWeighted(img_bgr, 1 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def process(img_bgr, img_size=224, amount=0.8, sigma=1.5):
    out = unsharp(img_bgr, amount, sigma)
    return cv2.resize(out, (img_size, img_size), interpolation=cv2.INTER_LINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=BCN_CSV)
    ap.add_argument("--crop_dir", default=CROP_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--amount", type=float, default=0.8)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    rows = df[["isic_id", "image"]].dropna()
    if args.limit:
        rows = rows.iloc[:args.limit]

    if args.preview:
        from PIL import Image
        sel = rows.iloc[::max(1, len(rows) // 12)][:12]
        TS = 220
        canvas = Image.new("RGB", (2 * (TS + 8) + 8, len(sel) * (TS + 8) + 8), (20, 20, 20))
        r = 0
        for _, row in sel.iterrows():
            cp = os.path.join(args.crop_dir, f"{row['isic_id']}.jpg")
            im = cv2.imread(cp)
            if im is None:
                continue
            out = process(im, args.img_size, args.amount, args.sigma)
            before = Image.fromarray(cv2.cvtColor(cv2.resize(im, (TS, TS)), cv2.COLOR_BGR2RGB))
            after = Image.fromarray(cv2.cvtColor(cv2.resize(out, (TS, TS)), cv2.COLOR_BGR2RGB))
            y = 8 + r * (TS + 8)
            canvas.paste(before, (8, y)); canvas.paste(after, (8 + TS + 8, y)); r += 1
        os.makedirs("hard_cases_vlm", exist_ok=True)
        outp = "hard_cases_vlm/_UNSHARP_PREVIEW.png"
        canvas.save(outp)
        print(f"PREVIEW (kiri=RAW crop, kanan=unsharp): {outp}")
        return

    os.makedirs(args.out, exist_ok=True)
    ok = miss = 0
    for i, (_, row) in enumerate(rows.iterrows()):
        cp = os.path.join(args.crop_dir, f"{row['isic_id']}.jpg")
        im = cv2.imread(cp)
        if im is None:
            miss += 1
            continue
        out = process(im, args.img_size, args.amount, args.sigma)
        cv2.imwrite(os.path.join(args.out, str(row["image"])), out)
        ok += 1
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(rows)} ...")
    print(f"Selesai: {ok} ditulis, {miss} crop tak ada -> {args.out}")


if __name__ == "__main__":
    main()
