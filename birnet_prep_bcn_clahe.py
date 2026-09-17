#!/usr/bin/env python3
"""Prep BCN full-frame → CLAHE (LAB) + bilateral filter (lensa Bagian 3.2).

Mirip birnet_eq_bcn_clean_eval.prep tapi TANPA hair-removal dan TANPA resize:
cache full-frame di BIRNet_Research/outputs/bcn_clahe supaya dataset
(birnet_retrain_bcn.py --img_mode clahe) tinggal resize ke 224.

Usage:
  .venv/bin/python3 birnet_prep_bcn_clahe.py [--skip_existing]
"""
import argparse
import multiprocessing
import os

import cv2
import numpy as np

import birnet_final as bf
from birnet_external_zero_shot import BCN_CSV, BCN_IMG

CLAHE_DIR = "BIRNet_Research/outputs/bcn_clahe"


def clahe_one(args):
    src, out = args
    if os.path.exists(out):
        return 0
    try:
        img = np.array(bf.Image.open(src).convert("RGB")).astype(np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        return 1
    except Exception as e:
        print(f"ERR {src}: {e}")
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(BCN_CSV)
    os.makedirs(CLAHE_DIR, exist_ok=True)
    jobs = [(os.path.join(BCN_IMG, str(r["image"])),
             os.path.join(CLAHE_DIR, str(r["image"]))) for _, r in df.iterrows()]
    jobs = [(s, o) for s, o in jobs if os.path.exists(s)]
    missing = [(s, o) for s, o in jobs if not os.path.exists(o)]
    print(f"[CLAHE] BCN total={len(jobs)} sisa-diproses={len(missing)}")
    if missing and not args.skip_existing:
        with multiprocessing.Pool(max(1, os.cpu_count() - 2)) as pool:
            done = 0
            for r in pool.imap_unordered(clahe_one, missing, chunksize=50):
                done += r
                if done and done % 500 == 0:
                    print(f"  {done} diproses")
    print(f"[CLAHE] selesai -> {len(os.listdir(CLAHE_DIR))} file di {CLAHE_DIR}")


if __name__ == "__main__":
    main()