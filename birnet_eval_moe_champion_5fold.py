"""Validasi 5-FOLD "Sang Juara Sejati" (Anatomical MoE Statis 80:20).

Mengevaluasi, untuk tiap fold 0-4, konfigurasi yang di fold-0 mencetak F1=0.8235:
    Jenderal MURNI (gen_pure) + Torso v1 + Limb v1 + Head v2 (sintetis), Statis 80:20, TTA.

Untuk konteks, tiap fold juga melaporkan:
    - Generalist-only (gen_pure)         -> baseline pembanding yg JUJUR (single terbaik)
    - All-Pure Statis 80:20 (Head v1)    -> uji apakah "Head butuh sintetis" bertahan 5-fold
    - Head-v2 Statis 80:20               -> SANG JUARA
    - (opsional --with_router) versi Smart Router untuk uji klaim "router overfit"

Fungsi inti (extract_probs, eval_logic, build_exp_arrays, train_router_ensemble)
diimpor LANGSUNG dari birnet_eval_moe_pure_champion agar identik dgn run fold-0.

Fold yg checkpoint-nya belum lengkap otomatis di-skip (bisa dijalankan bertahap).

Usage:
    .venv/bin/python3 birnet_eval_moe_champion_5fold.py                # statis saja (cepat)
    .venv/bin/python3 birnet_eval_moe_champion_5fold.py --with_router  # + Smart Router
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score

import birnet_final as bf
from train_specialist_expert import MultimodalClassifier, set_active_meta_fields, make_loader, detect_metadata_fields
from train_classifier_phase2_final import CROP_DIR
from birnet_representative_val_split import stratified_ham_kfold_fully_representative
from birnet_eval_moe_pure_champion import (
    extract_probs, eval_logic, build_exp_arrays, train_router_ensemble,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GEN_DIR = "BIRNet_Research/outputs/checkpoints/phase2_multimodal_ablation"
EXP_DIR = "BIRNet_Research/outputs/checkpoints/phase2_multimodal"


def ckpt_paths(bb, fold):
    return {
        "gen_pure": f"{GEN_DIR}/cls_{bb}_multimodal_fold{fold}_best.pth",
        "torso":    f"{EXP_DIR}/cls_{bb}_expert_torso_fold{fold}_best.pth",
        "limb":     f"{EXP_DIR}/cls_{bb}_expert_limb_fold{fold}_best.pth",
        "head_v1":  f"{EXP_DIR}/cls_{bb}_expert_head_fold{fold}_best.pth",
        "head_v2":  f"{EXP_DIR}/cls_{bb}_expert_head_fold{fold}_active_noise_best.pth",
    }


def load_models(bb, fold, cfg):
    paths = ckpt_paths(bb, fold)
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        return None, missing
    models = {}
    for key, path in paths.items():
        m = MultimodalClassifier(bb, num_classes=cfg.num_classes, meta_dim=15, img_size=cfg.img_size).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE)["model_state"], strict=True)
        m.eval()
        models[key] = m
    return models, []


def eval_fold(bb, fold, cfg, ham_meta, ham_df, val_aug, with_router):
    models, missing = load_models(bb, fold, cfg)
    if models is None:
        print(f"[FOLD {fold}] SKIP — checkpoint belum ada: {missing}")
        return None

    # split identik dgn semua skrip lain
    val_df = test_df = None
    for _, fi, _, val_idx, tst_idx in stratified_ham_kfold_fully_representative(
            ham_meta, cfg, ["dataset"], n_splits=5, n_repeats=1):
        if fi == fold:
            val_df = ham_df.iloc[val_idx].reset_index(drop=True)
            test_df = ham_df.iloc[tst_idx].reset_index(drop=True)
            break

    test_loader = make_loader(test_df, cfg, val_aug, shuffle=False)
    test_tta, test_labels, test_metas = extract_probs(models, test_loader, cfg)
    test_gen = np.array(test_tta["gen_pure"])

    valid = test_labels >= 0
    lbl = test_labels[valid]
    gen_pred = np.argmax(test_gen, axis=1)[valid]
    row = {
        "fold": fold,
        "gen_only_f1": float(f1_score(lbl, gen_pred, average="macro")),
        "gen_only_bal": float(balanced_accuracy_score(lbl, gen_pred)),
    }

    # Statis 80:20 untuk dua kombinasi
    exp_pure = build_exp_arrays(test_tta, test_df, head_ver="head_v1")
    exp_champ = build_exp_arrays(test_tta, test_df, head_ver="head_v2")
    row["allpure_static_f1"] = float(eval_logic(test_gen, exp_pure, test_metas, test_labels, routers=None))
    row["champ_static_f1"] = float(eval_logic(test_gen, exp_champ, test_metas, test_labels, routers=None))

    if with_router:
        val_loader = make_loader(val_df, cfg, val_aug, shuffle=False)
        val_tta, val_labels, val_metas = extract_probs(models, val_loader, cfg)
        val_gen = np.array(val_tta["gen_pure"])
        for tag, head_key in [("allpure", "head_v1"), ("champ", "head_v2")]:
            val_exp = build_exp_arrays(val_tta, val_df, head_ver=head_key)
            test_exp = exp_pure if head_key == "head_v1" else exp_champ
            routers = train_router_ensemble(val_gen, val_exp, val_metas, val_labels, n_routers=10)
            row[f"{tag}_router_f1"] = float(eval_logic(test_gen, test_exp, test_metas, test_labels, routers=routers))

    print(f"[FOLD {fold}] gen_only={row['gen_only_f1']:.4f} | "
          f"all_pure_static={row['allpure_static_f1']:.4f} | "
          f"CHAMP_static={row['champ_static_f1']:.4f}"
          + (f" | champ_router={row.get('champ_router_f1', float('nan')):.4f}" if with_router else ""))
    del models
    torch.cuda.empty_cache()
    return row


def summarize(rows, key, label):
    vals = np.array([r[key] for r in rows if key in r], dtype=float)
    if len(vals) == 0:
        return
    print(f"  {label:32s} mean={vals.mean():.4f}  std={vals.std():.4f}  "
          f"folds={['%.4f' % v for v in vals]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_base_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--with_router", action="store_true", help="Ikut latih+uji Smart Router (lebih lambat)")
    ap.add_argument("--out_json", default="moe_champion_5fold_results.json")
    args = ap.parse_args()

    set_active_meta_fields("localization")
    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    ham_df = bf.build_ham10000_dataframe(paths)
    has_crop = ham_df["image_id"].apply(lambda iid: os.path.exists(os.path.join(CROP_DIR, f"{iid}.jpg")))
    ham_df = ham_df[has_crop].reset_index(drop=True)
    bucketed, _ = detect_metadata_fields(ham_df, n_bins_numeric=9)
    overlap = [c for c in bucketed.columns if c in ham_df.columns]
    ham_meta = pd.concat([ham_df.drop(columns=overlap), bucketed], axis=1)
    _, val_aug = bf.build_augmentations(cfg.img_size, seed=42)

    rows = []
    for fold in range(5):
        r = eval_fold(args.backbone, fold, cfg, ham_meta, ham_df, val_aug, args.with_router)
        if r is not None:
            rows.append(r)

    print("\n" + "=" * 78)
    print(f"RINGKASAN 5-FOLD ANATOMICAL MoE (n={len(rows)} fold tersedia)")
    print("=" * 78)
    summarize(rows, "gen_only_f1",       "Generalist-only (baseline)")
    summarize(rows, "allpure_static_f1", "All-Pure Static 80:20")
    summarize(rows, "champ_static_f1",   "CHAMPION (Head-v2) Static 80:20")
    if args.with_router:
        summarize(rows, "allpure_router_f1", "All-Pure + Smart Router")
        summarize(rows, "champ_router_f1",   "CHAMPION (Head-v2) + Router")

    if rows:
        json.dump(rows, open(args.out_json, "w"), indent=2)
        print(f"\nDetail per-fold disimpan: {args.out_json}")
        if len(rows) < 5:
            print(f"(Baru {len(rows)}/5 fold; jalankan lagi setelah training fold sisanya selesai.)")


if __name__ == "__main__":
    main()
