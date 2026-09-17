#!/usr/bin/env python3
"""Prep BCN crop-lesi → CLAHE (LAB) + bilateral (lensa Bagian 3.2).

Urutan = crop dulu (baca cropped_phase1_final/{iid}.jpg), lalu aplikasikan
CLAHE+bilateral — persis lensa training champion pada crop. Hasil dicache di
BIRNet_Research/outputs/bcn_crop_clahe/{iid}.jpg supaya dataset retrain
(--img_mode crop_clahe di birnet_retrain_bcn.py) tinggal resize ke 224.

Contrast dgn birnet_prep_bcn_clahe.py (full-frame) dan CLEAN_DIR (GWDR).

Usage:
  .venv/bin/python3 birnet_prep_bcn_crop_clahe.py [--split test|None]
"""
import argparse
import multiprocessing
import os

import cv2
import numpy as np

from ame_derm.experts.train_classifier_phase2_final import CROP_DIR
from ame_derm.external_validation.external_zero_shot import BCN_CSV, EXT_TO_LABEL

OUT_DIR = "BIRNet_Research/outputs/bcn_crop_clahe"


def clahe_one(args):
    iid, src, out = args
    if os.path.exists(out):
        return 0
    try:
        img = cv2.imread(src)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        return 1
    except Exception as e:
        print(f"ERR {iid}: {e}")
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(BCN_CSV)
    if args.split:
        df = df[df["split"] == args.split]
    df = df[df["diagnosis"].isin(EXT_TO_LABEL)]
    df = df[df["isic_id"].notna()]

    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = []
    for _, r in df.iterrows():
        iid = str(r["isic_id"])
        src = os.path.join(CROP_DIR, f"{iid}.jpg")
        out = os.path.join(OUT_DIR, f"{iid}.jpg")
        if os.path.exists(src):
            jobs.append((iid, src, out))
    missing = [j for j in jobs if not os.path.exists(j[2])]
    print(f"[CROP+CLAHE] terlabel={len(df)} punya-crop={len(jobs)} "
          f"sisa-diproses={len(missing)}")
    if missing:
        with multiprocessing.Pool(max(1, os.cpu_count() - 2)) as pool:
            done = 0
            for r in pool.imap_unordered(clahe_one, missing, chunksize=50):
                done += r
                if done and done % 500 == 0:
                    print(f"  {done} diproses")
    print(f"[CROP+CLAHE] selesai -> {len(os.listdir(OUT_DIR))} file di {OUT_DIR}")


if __name__ == "__main__":
    main()