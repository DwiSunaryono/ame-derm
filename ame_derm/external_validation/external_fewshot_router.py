#!/usr/bin/env python3
"""STRATEGI 3 — FEW-SHOT ADAPTATION (Fine-Tuning Cepat Router MoE) di BCN20000.

Model dilatih 100% HAM10000 (frozen). Hanya lapisan SmartRouter MoE yang
dilatih ulang pada 10% data BCN20000 (adaptasi), lalu diuji pada sisa 90%.

Narasi: "Rumah sakit baru mengkalibrasi model hanya dengan 10% data lokal
mereka, tanpa menyentuh kecerdasan utama model generalis."

Protokol per fold 0-4 (konsisten dgn champion & birnet_external_zero_shot.py):
  - Model FROZEN: gen_pure + eksper torso/limb/head_v2 (TTA 6-view, meta loc).
  - Adapt set = 10% BCN (stratified per label), test set = sisa 90%.
  - Router dilatih HANYA dari probabilitas adapt set (SmartRouter, NLL on
    merged). Semua weight backbone EKSPLISIT tidak di-update.

Perbandingan pada test 90% BCN (makro-7):
  - gen_only        : DINOv2 standar, tanpa adaptasi
  - static_080      : champion statis 80:20, tanpa adaptasi (glass-ceiling)
  - router_adapt    : MoE + router yang di-fine-tune pada 10% BCN  <-- BIRODA

Reuse (IMPORT, tanpa memodifikasi): build_external_df / extract_probs dari
birnet_external_zero_shot.py; build_exp_arrays & train_router_ensemble dari
birnet_eval_moe_pure_champion.py.

Usage:
  .venv/bin/python3 birnet_external_fewshot_router.py                     # BCN, 10%, fold 0-4, raw
  .venv/bin/python3 birnet_external_fewshot_router.py --fold 0 --adapt_frac 0.10 --img_mode crop
  .venv/bin/python3 birnet_external_fewshot_router.py --n_routers 5 --epochs 60 --seed 2024
"""
import argparse
import json
import os
import numpy as np
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split

from ame_derm.common import birnet_final as bf
from ame_derm.experts.train_specialist_expert import set_active_meta_fields
from ame_derm.experts.train_moe_smart_router import SmartRouter

from ame_derm.external_validation.external_zero_shot import (
    build_external_df, make_loader, extract_probs, LAB_NAMES, DEVICE, BACKBONE,
    GEN_DIR, EXP_DIR,
)
from ame_derm.moe_fusion.eval_moe_pure_champion import build_exp_arrays, train_router_ensemble

CKPT_FOLD_BB = BACKBONE


def ckpt_paths(fold):
    return {
        "gen_pure": f"{GEN_DIR}/cls_{CKPT_FOLD_BB}_multimodal_fold{fold}_best.pth",
        "torso": f"{EXP_DIR}/cls_{CKPT_FOLD_BB}_expert_torso_fold{fold}_best.pth",
        "limb": f"{EXP_DIR}/cls_{CKPT_FOLD_BB}_expert_limb_fold{fold}_best.pth",
        "head_v1": f"{EXP_DIR}/cls_{CKPT_FOLD_BB}_expert_head_fold{fold}_best.pth",
        "head_v2": f"{EXP_DIR}/cls_{CKPT_FOLD_BB}_expert_head_fold{fold}_active_noise_best.pth",
    }


def load_models(fold, cfg):
    paths = ckpt_paths(fold)
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        return None, missing
    models = {}
    for key, path in paths.items():
        m = __import__("train_specialist_expert", fromlist=["MultimodalClassifier"]).MultimodalClassifier(
            BACKBONE, num_classes=cfg.num_classes, meta_dim=15, img_size=cfg.img_size).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False)["model_state"],
                          strict=False)
        m.eval()
        models[key] = m
    return models, []


def merge_with_router_ensemble(gen, exp, metas, routers):
    """Kembalikan probabilitas merged rata-rata dari ensembel router."""
    merged = np.zeros_like(gen, dtype=np.float32)
    with torch.no_grad():
        for r in routers:
            w = r(torch.tensor(gen).float().to(DEVICE),
                  torch.tensor(exp).float().to(DEVICE),
                  torch.tensor(metas).float().to(DEVICE)).cpu().numpy()
            merged += w * exp + (1 - w) * gen
    return merged / len(routers)


def macro(y, preds, subset=None):
    if subset is not None:
        mask = np.isin(y, subset)
        return float(f1_score(y[mask], preds[mask], labels=subset,
                              average="macro", zero_division=0))
    classes = np.unique(y)
    return float(f1_score(y, preds, labels=classes, average="macro", zero_division=0))


def run_fold(fold, adapt_df, test_df, cfg, val_aug, args):
    models, missing = load_models(fold, cfg)
    if models is None:
        print(f"[FOLD {fold}] SKIP — checkpoint belum ada: {missing}")
        return None

    adapt_loader = make_loader(adapt_df, cfg, val_aug, args.img_mode, batch_size=args.batch_size)
    test_loader = make_loader(test_df, cfg, val_aug, args.img_mode, batch_size=args.batch_size)

    print(f"[FOLD {fold}] extract probs: adapt(n={len(adapt_df)}) & test(n={len(test_df)}) ...")
    a_tta, a_lab, a_meta = extract_probs(models, adapt_loader)
    t_tta, t_lab, t_meta = extract_probs(models, test_loader)

    a_gen = np.array(a_tta["gen_pure"])[: len(a_lab)]
    t_gen = np.array(t_tta["gen_pure"])[: len(t_lab)]
    a_exp = build_exp_arrays(a_tta, adapt_df, head_ver="head_v2")[: len(a_lab)]
    t_exp = build_exp_arrays(t_tta, test_df, head_ver="head_v2")[: len(t_lab)]

    # 1) baseline: gen only (frozen)
    gen_macro = macro(t_lab, t_gen.argmax(1))
    # 2) champion statis 80:20 (frozen)
    static_macro = macro(t_lab, (0.8 * t_exp + 0.2 * t_gen).argmax(1))

    # 3) few-shot: fine-tune router PADA adapt (10%) saja
    routers = train_router_ensemble(a_gen, a_exp, a_meta, a_lab,
                                    n_routers=args.n_routers, epochs=args.epochs)
    t_merged = merge_with_router_ensemble(t_gen, t_exp, t_meta, routers)
    router_macro = macro(t_lab, t_merged.argmax(1))

    row = {
        "fold": fold, "n_adapt": len(adapt_df), "n_test": len(test_df),
        "gen_macro": gen_macro, "static_macro": static_macro, "router_macro": router_macro,
        "router_delta_om_static": router_macro - static_macro,
        "router_delta_om_gen": router_macro - gen_macro,
        "router_bacc": float(balanced_accuracy_score(t_lab, t_merged.argmax(1))),
        "static_bacc": float(balanced_accuracy_score(t_lab, (0.8 * t_exp + 0.2 * t_gen).argmax(1))),
        "router_acc": float(accuracy_score(t_lab, t_merged.argmax(1))),
    }
    print(f"[FOLD {fold}] gen={gen_macro:.4f} | static={static_macro:.4f} | "
          f"router_finetune={router_macro:.4f} (Δstatic={row['router_delta_om_static']:+.4f}, "
          f"Δgen={row['router_delta_om_gen']:+.4f})")
    del models
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["bcn", "msk"], default="bcn")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--img_mode", choices=["raw", "crop", "auto"], default="raw",
                    help="BCN crop tersedia; MSK test wajib raw")
    ap.add_argument("--adapt_frac", type=float, default=0.10)
    ap.add_argument("--n_routers", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="birnet_external_fewshot_router_results.json")
    args = ap.parse_args()

    set_active_meta_fields("localization")
    cfg = bf.Config()
    _, val_aug = bf.build_augmentations(cfg.img_size, seed=42)

    ext_df, _ = build_external_df(args.source, args.img_mode)
    if len(ext_df) == 0:
        print("df eksternal kosong — cek img_mode & source.")
        return
    print(f"[{args.source}] terlabel={len(ext_df)} | adapt_frac={args.adapt_frac} | "
          f"dist={ext_df['label'].value_counts().sort_index().to_dict()}")

    adapt_df, test_df = train_test_split(ext_df, test_size=1 - args.adapt_frac,
                                         stratify=ext_df["label"], random_state=args.seed)
    adapt_df = adapt_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"adapt={len(adapt_df)} test={len(test_df)}")

    rows = []
    for fold in (range(5) if args.fold is None else [args.fold]):
        r = run_fold(fold, adapt_df, test_df, cfg, val_aug, args)
        if r is not None:
            rows.append(r)

    if not rows:
        print("Tidak ada hasil — semua fold di-skip.")
        return

    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nDetail per-fold disimpan: {args.out}")

    print("\n" + "=" * 76)
    print(f"STRATEGI 3 — FEW-SHOT ROUTER @ BCN, adapt={args.adapt_frac:.0%} "
          f"({rows[0]['n_adapt']} samples), test={rows[0]['n_test']} (macro-7)")
    print("=" * 76)
    for key, label in [("gen_macro", "Generalist-only (frozen)"),
                       ("static_macro", "MoE Static 80:20 (frozen)"),
                       ("router_macro", "MoE + Router fine-tuned 10%")]:
        v = np.array([r[key] for r in rows])
        print(f"  {label:28s} macro-F1 = {v.mean():.4f} ± {v.std():.4f}"
              f"   folds={['%.4f' % x for x in v]}")
    d1 = np.array([r["router_delta_om_static"] for r in rows])
    d2 = np.array([r["router_delta_om_gen"] for r in rows])
    print(f"  ROUTER Δ vs static      = {d1.mean():+.4f} ± {d1.std():+.4f}")
    print(f"  ROUTER Δ vs gen         = {d2.mean():+.4f} ± {d2.std():+.4f}")


if __name__ == "__main__":
    main()