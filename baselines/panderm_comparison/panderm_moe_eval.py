#!/usr/bin/env python3
"""MoE+PanDerm: PanDerm-Large gen + PanDerm-base per-zone experts, routing by
localization, alpha tuned on VAL -> TEST. Every probability comes from PanDerm's
val.csv/test.csv (filename + probability_class_* columns), aligned by filename.

Fills in the missing 2x2 cell: MoE+PanDerm vs MoE+DINOv2.
Usage: .venv/bin/python birnet_panderm_moe_eval.py --fold 0 --seed 122
"""
import os, argparse
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

import config
ROOT = config.AME_DERM_ROOT
PDRES = f"{ROOT}/PanDerm/results"
ZONE_MAP = {"torso": ["back", "trunk", "abdomen", "chest"],
            "limb": ["lower extremity", "upper extremity", "foot", "hand", "acral"],
            "head": ["face", "neck", "scalp", "ear"]}
PCOLS = [f"probability_class_{i}" for i in range(7)]


def get_zone(loc):
    for z, locs in ZONE_MAP.items():
        if loc in locs:
            return z
    return "gen"


def probs_by_filename(csv):
    d = pd.read_csv(csv)
    key = d["filename"].apply(lambda p: os.path.basename(str(p)))
    return dict(zip(key, d[PCOLS].values)), dict(zip(key, d["true_label"].values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    F, S = args.fold, args.seed

    fold_csv = pd.read_csv(f"{ROOT}/BIRNet_Research/outputs/rep_folds/rep_fold{F}.csv")
    fold_csv["fn"] = fold_csv["image"].apply(lambda p: os.path.basename(str(p)))
    loc_of = dict(zip(fold_csv["fn"], fold_csv["localization"]))

    # gen (PanDerm-Large, rigorous run) — val/test probabilities in full
    gen_dir = f"{PDRES}/rigorous_5fold/fold{F}_seed{S}"
    gen_val_p, yv_map = probs_by_filename(f"{gen_dir}/val.csv")
    gen_te_p, gen_te_y = probs_by_filename(f"{gen_dir}/test.csv")

    # experts (PanDerm-base) — zone val/test probabilities
    exp_val, exp_te = {}, {}
    for z in ZONE_MAP:
        ed = f"{PDRES}/moe_panderm/fold{F}_seed{S}/{z}"
        if not os.path.exists(f"{ed}/test.csv"):
            print(f"❌ expert {z} test.csv tak ada: {ed}\n   Latih dulu: bash birnet_panderm_experts.sh {F} {S}"); return
        exp_val[z], _ = probs_by_filename(f"{ed}/val.csv")
        exp_te[z], _ = probs_by_filename(f"{ed}/test.csv")

    def routed(gen_p, exp_p, y_ref):
        """Build (routed_expert, gen, y) for every filename in y_ref, in a consistent order."""
        rex, rgen, y = [], [], []
        for fn, gv in gen_p.items():
            z = get_zone(loc_of.get(fn, "unknown"))
            ex = exp_p[z].get(fn) if z in exp_p else None
            rex.append(ex if ex is not None else gv)   # fall back to gen when zone is 'gen' / absent
            rgen.append(gv); y.append(y_ref[fn])
        return np.array(rex), np.array(rgen), np.array(y)

    vex, vgen, yv = routed(gen_val_p, exp_val, yv_map)
    tex, tgen, yt = routed(gen_te_p, exp_te, gen_te_y)

    # sweep alpha on VAL
    best_a, best_m = 0.0, -1
    for a in np.arange(0.0, 0.951, 0.05):
        m = f1_score(yv, (a * vex + (1 - a) * vgen).argmax(1), average="macro", zero_division=0)
        if m > best_m:
            best_m, best_a = m, a

    moe = f1_score(yt, (best_a * tex + (1 - best_a) * tgen).argmax(1), average="macro", zero_division=0)
    gen_only = f1_score(yt, tgen.argmax(1), average="macro", zero_division=0)
    print("\n" + "=" * 60)
    print(f"MoE+PanDerm fold{F} (alpha={best_a:.2f} dari VAL, val_macro={best_m:.4f})")
    print(f"  PanDerm-Large gen (single)  macro-F1 = {gen_only:.4f}")
    print(f"  MoE + PanDerm               macro-F1 = {moe:.4f}   Δ={moe-gen_only:+.4f}")
    print("=" * 60)
    print(f"  Pembanding fold{F}: MoE+DINOv2 pure=0.8235 hybrid=0.8368 | single-PanDerm=0.7494")
    print(f"RESULT_SUMMARY moe_panderm fold={F} alpha={best_a:.2f} gen={gen_only:.4f} moe={moe:.4f}")
    import json
    outp = f"{PDRES}/moe_panderm/moe_panderm_fold{F}_seed{S}.json"
    with open(outp, "w") as fjs:
        json.dump({"fold": F, "seed": S, "alpha": float(best_a),
                   "gen_macro_f1": float(gen_only), "moe_macro_f1": float(moe),
                   "delta": float(moe - gen_only)}, fjs, indent=2)
    print(f"saved: {outp}")


if __name__ == "__main__":
    main()
