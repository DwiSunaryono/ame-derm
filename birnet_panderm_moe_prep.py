#!/usr/bin/env python3
"""Buat CSV per-zona dari rep_fold{F}.csv untuk melatih PanDerm-base experts.
Output: rep_fold{F}_{zone}.csv (train/val/test difilter ke lokalisasi zona itu)."""
import os, argparse, pandas as pd

ZONE_MAP = {
    "torso": ["back", "trunk", "abdomen", "chest"],
    "limb": ["lower extremity", "upper extremity", "foot", "hand", "acral"],
    "head": ["face", "neck", "scalp", "ear"],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds_dir", default="BIRNet_Research/outputs/rep_folds")
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    src = os.path.join(args.folds_dir, f"rep_fold{args.fold}.csv")
    d = pd.read_csv(src)
    for z, locs in ZONE_MAP.items():
        sub = d[d["localization"].isin(locs)].copy()
        out = os.path.join(args.folds_dir, f"rep_fold{args.fold}_{z}.csv")
        sub.to_csv(out, index=False)
        print(f"  {z}: {sub.groupby('split').size().to_dict()} -> {out}")

if __name__ == "__main__":
    main()
