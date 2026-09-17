#!/usr/bin/env python3
"""ZERO-SHOT EXTERNAL VALIDATION (Strategi 1) — MoE champion fold 0-4 di BCN/MSK.

Model telah dilatih penuh 100% HAM10000 5-Fold (TIDAK pernah menyaksikan BCN/MSK).
Skrip ini memakai checkpoint fold-f yang SAMA (gen_pure + eksper torso/limb/head_v2)
pada kohort eksternal sebagai BLIND TEST, lantas membandingkan:

    - Generalist-only  (gen_pure)          -> baseline DINOv2 standar
    - Anatomical MoE   (head_v2, statis 0.8) -> BirNET

protokol routing & blending IDENTIK dgn birnet_eval_moe_champion_5fold.py
(weight_expert=0.8, TTA 6-view, meta = localization saja), sehingga angka internal
0.8235 (fold-0 HAM test) tetap sebanding dengan angka eksternal di sini.

Protokol label eksternal (terverifikasi konsisten dgn taksonomi HAM 7-kelas):
    - BCN : makro-7 (smua kelas HAM ada); SCC & unmappable dibuang
    - MSK : makro-3 headline (mel/nv/bkl) + makro-5 suplemen (mel/nv/bkl/bcc/vasc)
      kelas ultra-tipis df(9)/akiec(8) diekstrak dgn catatan

Gambar: DEFAULT raw full-image dari panderm_datasets/ (MSK test tak punya crop;
raw menjaga protokol seragam antar-kohort). Opsional --img_mode crop memakai
CROP_DIR (hanya tersedia utk BCN).

Usage:
  .venv/bin/python3 birnet_external_zero_shot.py                     # BCN+MSK, fold 0-4, raw
  .venv/bin/python3 birnet_external_zero_shot.py --source bcn --img_mode crop
  .venv/bin/python3 birnet_external_zero_shot.py --fold 0 --out /tmp/zero_shot_bcn.json
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score

from ame_derm.common import birnet_final as bf
from ame_derm.experts.train_specialist_expert import (
    MultimodalClassifier, set_active_meta_fields, encode_metadata,
    EXPERT_MAPPING, LOCALIZATION_CATEGORIES,
)
from ame_derm.experts.train_classifier_phase2_final import CROP_DIR
from ame_derm.moe_fusion.eval_moe_pure_champion import build_exp_arrays, get_expert_name, tta_views

import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "vit_base_patch14_reg4_dinov2.lvd142m"
BACKBONE_SHORT = "vit_base_patch14_reg4_dinov2"
GEN_DIR = "BIRNet_Research/outputs/checkpoints/phase2_multimodal_ablation"
EXP_DIR = "BIRNet_Research/outputs/checkpoints/phase2_multimodal"

BCN_CSV = config.BCN_CSV
BCN_IMG = config.BCN_IMG
MSK_CSV = "panderm_datasets/MSKCC/MSKCC/mskcc.csv"
MSK_IMG = "panderm_datasets/MSKCC/MSKCC/images"

# pemetaan diagnosis -> label HAM (0-6, HAM_LABEL_MAP di birnet_final)
EXT_TO_LABEL = {
    "melanoma": 0, "nevus": 1, "basal cell carcinoma": 2,
    "seborrheic keratosis": 4, "solar lentigo": 4, "lichenoid keratosis": 4,
    "lentigo NOS": 4, "lentigo simplex": 4, "actinic keratosis": 3,
    "dermatofibroma": 5, "vascular lesion": 6, "angioma": 6, "angiokeratoma": 6,
}
DROP_DX = {"squamous cell carcinoma", "melanoma metastasis", "atypical melanocytic proliferation"}

# anatom_site_general eksternal -> kategori localization HAM (untuk meta & routing)
EXT_LOC_MAP = {
    "anterior torso": "abdomen", "posterior torso": "back", "lateral torso": "trunk",
    "head/neck": "neck", "lower extremity": "lower extremity",
    "upper extremity": "upper extremity", "palms/soles": "acral",
    "oral/genital": "genital",
}
LAB_NAMES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
MSK_MACRO3 = [0, 1, 4]      # mel/nv/bkl
MSK_MACRO5 = [0, 1, 2, 4, 6]  # mel/nv/bkl/bcc/vasc


def build_external_df(source, img_mode):
    """Hasilkan df eksternal yang bersih: image_id, path, label, localization."""
    if source == "bcn":
        df = pd.read_csv(BCN_CSV)
        df = df[df["split"] == "test"]
        img_dir = BCN_IMG
    else:
        df = pd.read_csv(MSK_CSV)
        df = df[df["split"] == "test"]
        img_dir = MSK_IMG

    df = df[df["diagnosis"].isin(EXT_TO_LABEL)].copy()
    df["label"] = df["diagnosis"].map(EXT_TO_LABEL)
    df["localization"] = df["anatom_site_general"].map(EXT_LOC_MAP).fillna("unknown")

    rows = []
    for _, r in df.iterrows():
        iid = str(r["isic_id"])
        if iid in ("UNKNOWN", "unknown", "nan") or pd.isna(iid):
            continue
        crop_p = os.path.join(CROP_DIR, f"{iid}.jpg")
        raw_p = os.path.join(img_dir, str(r["image"]))
        if img_mode == "crop" and os.path.exists(crop_p):
            rows.append({"image_id": iid, "img_path": crop_p, "label": int(r["label"]),
                         "localization": r["localization"]})
        elif img_mode == "raw" and os.path.exists(raw_p):
            rows.append({"image_id": iid, "img_path": raw_p, "label": int(r["label"]),
                         "localization": r["localization"]})
        elif img_mode == "auto":
            if os.path.exists(crop_p):
                rows.append({"image_id": iid, "img_path": crop_p, "label": int(r["label"]),
                             "localization": r["localization"]})
            elif os.path.exists(raw_p):
                rows.append({"image_id": iid, "img_path": raw_p, "label": int(r["label"]),
                             "localization": r["localization"]})
    return pd.DataFrame(rows), img_mode


class ExternalSkinLesionDataset(torch.utils.data.Dataset):
    """Memuat citra raw/crop + encoding metadata localization (serupa
    SkinLesionDatasetMultimodal, namun path gambar eksternalnya eksplisit)."""

    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.meta_vecs = [encode_metadata(np.nan, "unknown", loc, "unknown")
                          for loc in self.df["localization"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import cv2
        img = cv2.imread(self.df.loc[idx, "img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = self.transform(image=img, mask=np.zeros((img.shape[0], img.shape[1]),
                                                       dtype=np.float32))
        return aug["image"], torch.from_numpy(self.meta_vecs[idx]), int(self.df.loc[idx, "label"])


def make_loader(df, cfg, aug, img_mode, batch_size=64):
    ds = ExternalSkinLesionDataset(df, aug)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                       num_workers=4, pin_memory=True)


def extract_probs(models, loader):
    probs_tta = {k: [] for k in models.keys()}
    all_labels, all_metas = [], []
    with torch.no_grad():
        for imgs, metas, labels in loader:
            imgs, metas = imgs.to(DEVICE), metas.to(DEVICE)
            all_labels.extend(labels.numpy())
            all_metas.extend(metas.cpu().numpy())
            views = tta_views(imgs)
            for m_name, model in models.items():
                p_sum = torch.zeros((imgs.shape[0], 7), device=DEVICE)
                for view in views:
                    p_sum += torch.softmax(model(view, metas), dim=1)
                probs_tta[m_name].extend((p_sum / len(views)).cpu().numpy())
    return probs_tta, np.array(all_labels), np.array(all_metas)


def static_eval(gen, exp, y, weight_expert=0.8):
    merged = weight_expert * exp + (1 - weight_expert) * gen
    preds = merged.argmax(1)
    return preds, merged


def macro_on_subset(y, preds, subset=None):
    if subset is not None:
        mask = np.isin(y, subset)
        return float(f1_score(y[mask], preds[mask], labels=subset,
                              average="macro", zero_division=0)), int(mask.sum())
    return float(f1_score(y, preds, average="macro", zero_division=0)), len(y)


def report(source, img_mode, rows):
    print("\n" + "=" * 72)
    print(f"ZERO-SHOT EXTERNAL: {source.upper()} | img={img_mode} | n={len(rows)} | "
          f"fold mean±std over {len(rows)} rows")
    print("=" * 72)
    keys_map = {
        "bcn": [("macro7", None, "MAKRO-7 (semua kelas)"),
                ("macro3", MSK_MACRO3, "MAKRO-3 (mel/nv/bkl)"),
                ("macro5", MSK_MACRO5, "MAKRO-5 (mel/nv/bkl/bcc/vasc)")],
        "msk": [("macro7", None, "MAKRO-7 (semua kelas)"),
                ("macro3", MSK_MACRO3, "MAKRO-3 (mel/nv/bkl)"),
                ("macro5", MSK_MACRO5, "MAKRO-5 (mel/nv/bkl/bcc/vasc)")],
    }
    for tag, subset, label in keys_map[source]:
        print(f"\n  [{label}]")
        for mode, k in [("gen", "gen_macro"), ("moe", "moe_macro")]:
            vals = np.array([r[f"{k}_{tag}"] for r in rows])
            print(f"    {mode.upper():5s} macro-F1 = {vals.mean():.4f} ± {vals.std():.4f}"
                  f"   folds={['%.4f' % v for v in vals]}")
        d = np.array([r[f"moe_macro_{tag}"] - r[f"gen_macro_{tag}"] for r in rows])
        print(f"    DELTA (MoE-gen) = {d.mean():+.4f} ± {d.std():+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["bcn", "msk", "both"], default="both")
    ap.add_argument("--fold", type=int, default=None, help="Satu fold (0-4) saja; default semua")
    ap.add_argument("--img_mode", choices=["raw", "crop", "auto"], default="raw")
    ap.add_argument("--out", default="birnet_external_zero_shot_results.json")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    set_active_meta_fields("localization")
    cfg = bf.Config()
    _, val_aug = bf.build_augmentations(cfg.img_size, seed=42)

    sources = ["bcn", "msk"] if args.source == "both" else [args.source]
    folds = range(5) if args.fold is None else [args.fold]

    all_rows = []
    for src in sources:
        ext_df, used_mode = build_external_df(src, args.img_mode)
        if len(ext_df) == 0:
            print(f"[{src}] tidak ada gambar utk img_mode={args.img_mode} → SKIP")
            continue
        print(f"\n[{src}] test terlabel={len(ext_df)} | img={used_mode} | "
              f"dist={ext_df['label'].value_counts().sort_index().to_dict()}")
        loader = make_loader(ext_df, cfg, val_aug, used_mode, batch_size=args.batch_size)

        for fold in folds:
            ckpts = {
                "gen_pure": f"{GEN_DIR}/cls_{BACKBONE}_multimodal_fold{fold}_best.pth",
                "torso": f"{EXP_DIR}/cls_{BACKBONE}_expert_torso_fold{fold}_best.pth",
                "limb": f"{EXP_DIR}/cls_{BACKBONE}_expert_limb_fold{fold}_best.pth",
                "head_v1": f"{EXP_DIR}/cls_{BACKBONE}_expert_head_fold{fold}_best.pth",
                "head_v2": f"{EXP_DIR}/cls_{BACKBONE}_expert_head_fold{fold}_active_noise_best.pth",
            }
            missing = [k for k, p in ckpts.items() if not os.path.exists(p)]
            if missing:
                print(f"[{src} fold{fold}] SKIP — checkpoint belum ada: {missing}")
                continue

            models = {}
            for key, path in ckpts.items():
                m = MultimodalClassifier(BACKBONE, num_classes=cfg.num_classes,
                                         meta_dim=15, img_size=cfg.img_size).to(DEVICE)
                m.load_state_dict(torch.load(path, map_location=DEVICE,
                                             weights_only=False)["model_state"], strict=False)
                m.eval()
                models[key] = m

            print(f"[{src} fold{fold}] extracting TTA probs...")
            tta, y, metas = extract_probs(models, loader)
            gen = np.array(tta["gen_pure"])
            exp_champ = build_exp_arrays(tta, ext_df, head_ver="head_v2")
            y = np.array(y)[: len(gen)]

            preds_gen, _ = static_eval(gen, gen, y, weight_expert=1.0)
            preds_moe, _ = static_eval(gen, exp_champ, y, weight_expert=0.8)

            row = {"source": src, "fold": fold, "img_mode": used_mode, "n": len(y)}
            row["gen_macro_macro7"], _ = macro_on_subset(y, preds_gen)
            row["moe_macro_macro7"], _ = macro_on_subset(y, preds_moe)
            for tag, subset in [("macro3", MSK_MACRO3), ("macro5", MSK_MACRO5)]:
                row[f"gen_macro_{tag}"], row[f"n_{tag}"] = macro_on_subset(y, preds_gen, subset)
                row[f"moe_macro_{tag}"], _ = macro_on_subset(y, preds_moe, subset)
            # metrik global (semua kelas berlabel)
            row["gen_bacc"] = float(balanced_accuracy_score(y, preds_gen))
            row["moe_bacc"] = float(balanced_accuracy_score(y, preds_moe))
            row["moe_acc"] = float(accuracy_score(y, preds_moe))
            all_rows.append(row)
            print(f"  fold{fold}: gen_macro7={row.get('gen_macro_macro7', row.get('gen_macro_macro3','-')):.4f} "
                  f"moe_macro7={row.get('moe_macro_macro7', row.get('moe_macro_macro3','-')):.4f} "
                  f"moe_bacc={row['moe_bacc']:.4f}")
            del models
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    if not all_rows:
        print("Tidak ada hasil (semua fold di-skip?).")
        return

    json.dump(all_rows, open(args.out, "w"), indent=2)
    print(f"\nDetail per-fold disimpan: {args.out}")

    for src in sources:
        rows_src = [r for r in all_rows if r["source"] == src]
        if rows_src:
            report(src, rows_src[0]["img_mode"], rows_src)


if __name__ == "__main__":
    main()