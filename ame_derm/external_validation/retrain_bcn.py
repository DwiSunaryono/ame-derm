#!/usr/bin/env python3
"""STRATEGI 4 — RETRAIN PENUH framework BirNET di BCN20000 (fold-0 / split resmi).

Filosofi: BirNET ialah FRAMEWORK, bukan model statis. Ketika kohort baru muncul
(rumah sakit / dataset baru), framework di-RETRAIN ulang dari backbone pretrained
pada data lokal, tidak cuma zero/few-shot adaptasi. Skrip ini memperlihatkan
headroom-nya: gen_pure + eksper torso/limb/head dilatih pada BCN train 100%,
early-stop via BCN val, kemudian dievaluasi di BCN test (makro-7) dgn protokol
IDENTIK champion (TTA 6-view, static 0.8, meta localization).

Protokol perbandingan pada BCN test:
  - gen_only           : generalis retrained (bukan frozen)
  - static_080         : MoE retrained, blend statis 0.8
  - (referensi)        : zero/few-shot frozen dari birnet_external_*_results.json

Resep training IDENTIK train_specialist_expert.run_fold:
  backbone DINOv2 base, meta_dim=15 (localization), MixUp 0.2 + CutMix 1.0,
  inverse-freq weights + oversample target=1500/class, SupCon lambda=0.1,
  AdamW 3-groups (BACKBONE_LR_SCALE=0.1), warmup 2ep + CosineAnnealing T0=20,
  early stop by val_f1 (patience=12, max 80 ep), BATCH=32.

Reuse (IMPORT tanpa modifikasi):
  - train_specialist_expert: MultimodalClassifier, encode_metadata,
    EXPERT_MAPPING, LOCALIZATION_CATEGORIES, SupConLoss, set_active_meta_fields
  - train_classifier_phase2_mixup: SoftTargetCrossEntropyWeighted, konstanta
  - birnet_external_zero_shot: EXT_LOC_MAP / EXT_TO_LABEL (mapping label BCN)
  - birnet_eval_moe_pure_champion: get_expert_name, build_exp_arrays, tta_views

Usage:
  .venv/bin/python3 birnet_retrain_bcn.py                          # gen + 3 expert + eval, seed 42
  .venv/bin/python3 birnet_retrain_bcn.py --seed 7 --out retrain_seed7.json
  .venv/bin/python3 birnet_retrain_bcn.py --max_epochs 60 --patience 10
  .venv/bin/python3 birnet_retrain_bcn.py --gen_only               # hanya GEN (kontrol cepat)
  .venv/bin/python3 birnet_retrain_bcn.py --gen_gate 0.55          # skip expert jika GEN < 0.55
  .venv/bin/python3 birnet_retrain_bcn.py --warm_start             # init dari champion HAM fold0
"""
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.data import Dataset, DataLoader
from timm.data import Mixup
from sklearn.metrics import balanced_accuracy_score, f1_score

from ame_derm.common import birnet_final as bf
from ame_derm.experts.train_specialist_expert import (
    MultimodalClassifier, encode_metadata, EXPERT_MAPPING,
    SupConLoss, set_active_meta_fields,
)
from ame_derm.experts.train_classifier_phase2_mixup import (
    SoftTargetCrossEntropyWeighted, BACKBONE_LR_SCALE, WARMUP_EPOCHS,
)
from ame_derm.external_validation.external_zero_shot import (
    EXT_LOC_MAP, EXT_TO_LABEL, BCN_CSV, BCN_IMG, DEVICE, BACKBONE,
)
from ame_derm.moe_fusion.eval_moe_pure_champion import get_expert_name, build_exp_arrays, tta_views

BATCH_SIZE = 32
CKPT_DIR = "BIRNet_Research/outputs/checkpoints/retrain_bcn"
# Sekaligus meniru eksperimen dari birnet_retrain_bcn_lpft: agar ckpt dari
# img_mode berbeda (raw/crop/clean) TIDAK saling menimpa, gunakan suffix via
# env BCN_IMGMODE_SUFFIX. Kosong = nama asli (raw), sehingga evaluasi raw lama
# tetap sah.
BCN_SUFFIX = os.environ.get("BCN_IMGMODE_SUFFIX", "")
_suf = f"_{BCN_SUFFIX}" if BCN_SUFFIX else ""
BACKBONE = os.environ.get("BCN_BACKBONE", BACKBONE)  # override untuk ensemble lintas-backbone
GEN_CKPT = os.path.join(CKPT_DIR, f"cls_{BACKBONE}_multimodal_fold0{_suf}_best.pth")
EXP_CKPT = {
    "torso": os.path.join(CKPT_DIR, f"cls_{BACKBONE}_expert_torso_fold0{_suf}_best.pth"),
    "limb": os.path.join(CKPT_DIR, f"cls_{BACKBONE}_expert_limb_fold0{_suf}_best.pth"),
    "head": os.path.join(CKPT_DIR, f"cls_{BACKBONE}_expert_head_fold0{_suf}_best.pth"),
}
CLEAN_DIR = "BIRNet_Research/outputs/bcn_clean_gwdr"
CLAHE_DIR = "BIRNet_Research/outputs/bcn_clahe"
CROP_CLAHE_DIR = "BIRNet_Research/outputs/bcn_crop_clahe"
JURA_DIR = "BIRNet_Research/outputs/bcn_jura_clahe_crop_bil"
PIPE_BASE = "BIRNet_Research/outputs/bcn_pipe"
# Champion HAM fold0 (backbone sama) — dipakai warm-start supaya GEN retrained
# berawal dari representasi HAM (mendekati HAM champion), bukan DINOv2 fresh.
CHAMP_GEN_DIR = "BIRNet_Research/outputs/checkpoints/phase2_multimodal_ablation"
CHAMP_EXP_DIR = "BIRNet_Research/outputs/checkpoints/phase2_multimodal"
CHAMP_GEN_CKPT = os.path.join(
    CHAMP_GEN_DIR, f"cls_{BACKBONE}_multimodal_fold0_best.pth")
CHAMP_GEN_CKPT = os.environ.get("BCN_WARM_CKPT", CHAMP_GEN_CKPT)  # override path warm-start per backbone
CHAMP_EXP_CKPT = {
    "torso": os.path.join(CHAMP_EXP_DIR, f"cls_{BACKBONE}_expert_torso_fold0_best.pth"),
    "limb": os.path.join(CHAMP_EXP_DIR, f"cls_{BACKBONE}_expert_limb_fold0_best.pth"),
    "head": os.path.join(CHAMP_EXP_DIR, f"cls_{BACKBONE}_expert_head_fold0_best.pth"),
}


# ── Dataset (path eksplisit — raw BCN / cadangan crop) ─────────────────

class RetrainBCNDataset(Dataset):
    """SkinLesionDatasetMultimodal yang membaca path gambar EKSPLISIT (BCN)."""

    def __init__(self, df, img_size, transform=None, oversample=False,
                 target_per_class=None, num_classes=7):
        self.img_size = img_size
        self.transform = transform
        self.meta_map = {}
        for row in df.itertuples():
            self.meta_map[row.image_id] = encode_metadata(
                getattr(row, "age", np.nan),
                getattr(row, "sex", "unknown"),
                getattr(row, "localization", "unknown"),
                getattr(row, "dataset", "unknown"),
            )
        entries = [(row.image_id, row.img_path, row.label) for row in df.itertuples()]
        if oversample and target_per_class is not None:
            entries = bf.SkinLesionDataset._build_oversampled_index(
                entries, target_per_class, num_classes)
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        import cv2
        entry = self.entries[idx]
        if len(entry) == 4:
            image_id, img_path, label, do_aug = entry
        else:
            image_id, img_path, label = entry
            do_aug = False

        img = None
        cache_dir = os.environ.get("BCN_IMG_CACHE", "")
        if cache_dir:
            ck = os.path.join(cache_dir, f"{image_id}.npy")
            if os.path.exists(ck):
                img = np.load(ck, allow_pickle=False)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img is None:
            img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_np = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        if do_aug:
            if np.random.rand() > 0.5:
                img_np = cv2.flip(img_np, 1)
            if np.random.rand() > 0.5:
                img_np = cv2.flip(img_np, 0)
            img_np = np.rot90(img_np, np.random.randint(0, 4)).copy()

        if label is None or (isinstance(label, float) and np.isnan(label)):
            label = -1

        dummy_mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        if self.transform:
            aug = self.transform(image=img_np, mask=dummy_mask)
            img_t = aug["image"]
        else:
            img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0

        meta_t = torch.from_numpy(self.meta_map.get(image_id, np.zeros(15, dtype=np.float32)))
        return img_t, meta_t, int(label)


def make_loader(df, cfg, aug, shuffle=False, oversample=False):
    ds = RetrainBCNDataset(df, cfg.img_size, transform=aug, oversample=oversample,
                           target_per_class=cfg.oversample_target_per_class if oversample else None,
                           num_classes=cfg.num_classes)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=4,
                      pin_memory=True, drop_last=shuffle and len(ds) > BATCH_SIZE)


# ── Susun dataframe BCN (format mirip-HAM utk training) ────────────────

def build_bcn_df(img_mode="raw"):
    """Seluruh split BCN → df {image_id, img_path, label, localization, age, sex, dataset}."""
    df = pd.read_csv(BCN_CSV)
    df = df[df["diagnosis"].isin(EXT_TO_LABEL)].copy()
    df["label"] = df["diagnosis"].map(EXT_TO_LABEL)
    df["localization"] = df["anatom_site_general"].map(EXT_LOC_MAP).fillna("unknown")
    df["age"] = df["age_approx"]
    df["sex"] = df["sex"].fillna("unknown")
    df["sex"] = df["sex"].where(df["sex"].isin(["male", "female"]), "unknown")
    df["dataset"] = "bcn"

    rows = []
    for _, r in df.iterrows():
        iid = str(r["isic_id"])
        if iid in ("UNKNOWN", "unknown", "nan") or pd.isna(iid):
            continue
        from ame_derm.experts.train_classifier_phase2_final import CROP_DIR
        crop_p = os.path.join(CROP_DIR, f"{iid}.jpg")
        raw_p = os.path.join(BCN_IMG, str(r["image"]))
        clean_p = os.path.join(CLEAN_DIR, str(r["image"]))
        clahe_p = os.path.join(CLAHE_DIR, str(r["image"]))
        crop_clahe_p = os.path.join(CROP_CLAHE_DIR, f"{iid}.jpg")
        if img_mode == "crop" and os.path.exists(crop_p):
            rows.append({"image_id": iid, "img_path": crop_p, "label": int(r["label"]),
                         "localization": r["localization"], "age": r["age"],
                         "sex": r["sex"], "dataset": "bcn", "lesion_id": str(r["lesion_id"])})
        elif img_mode == "clean" and os.path.exists(clean_p):
            rows.append({"image_id": iid, "img_path": clean_p, "label": int(r["label"]),
                         "localization": r["localization"], "age": r["age"],
                         "sex": r["sex"], "dataset": "bcn", "lesion_id": str(r["lesion_id"])})
        elif img_mode == "clahe" and os.path.exists(clahe_p):
            rows.append({"image_id": iid, "img_path": clahe_p, "label": int(r["label"]),
                         "localization": r["localization"], "age": r["age"],
                         "sex": r["sex"], "dataset": "bcn", "lesion_id": str(r["lesion_id"])})
        elif img_mode == "crop_clahe" and os.path.exists(crop_clahe_p):
            rows.append({"image_id": iid, "img_path": crop_clahe_p, "label": int(r["label"]),
                         "localization": r["localization"], "age": r["age"],
                         "sex": r["sex"], "dataset": "bcn", "lesion_id": str(r["lesion_id"])})
        elif img_mode == "clahe_crop_bil":
            jura_p = os.path.join(JURA_DIR, str(r["image"]))
            if os.path.exists(jura_p):
                rows.append({"image_id": iid, "img_path": jura_p, "label": int(r["label"]),
                             "localization": r["localization"], "age": r["age"],
                             "sex": r["sex"], "dataset": "bcn", "lesion_id": str(r["lesion_id"])})
        elif ">" in img_mode:
            # pipeline umum: bcn_pipe/<op1>_<op2>_.../{image}
            pipe_dir = os.path.join(PIPE_BASE, img_mode.replace(">", "_"))
            pipe_p = os.path.join(pipe_dir, str(r["image"]))
            if os.path.exists(pipe_p):
                rows.append({"image_id": iid, "img_path": pipe_p, "label": int(r["label"]),
                             "localization": r["localization"], "age": r["age"],
                             "sex": r["sex"], "dataset": "bcn", "lesion_id": str(r["lesion_id"])})
        elif img_mode == "raw" and os.path.exists(raw_p):
            rows.append({"image_id": iid, "img_path": raw_p, "label": int(r["label"]),
                         "localization": r["localization"], "age": r["age"],
                         "sex": r["sex"], "dataset": "bcn", "lesion_id": str(r["lesion_id"])})
    out = pd.DataFrame(rows)
    return out


def lesion_grouped_split(df, seed, n_splits=5):
    """PENGGANTI split BCN bawaan yg BOCOR (train∩test lesi=915). Lesion-grouped +
    stratified label: test=1 fold, val=1 fold dari sisa, sisanya train.
    Verifikasi 0 kebocoran lesi antar-split."""
    from sklearn.model_selection import StratifiedGroupKFold
    groups = df["lesion_id"].astype(str).values
    y = df["label"].values
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(sgkf.split(df, y, groups))
    sub = df.iloc[trainval_idx].reset_index(drop=True)
    sgkf2 = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_idx, val_idx = next(sgkf2.split(sub, sub["label"].values,
                                       sub["lesion_id"].astype(str).values))
    train = sub.iloc[tr_idx].reset_index(drop=True)
    val = sub.iloc[val_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    g = lambda d: set(d["lesion_id"].astype(str))
    assert g(train).isdisjoint(g(test)), "LEAK train/test!"
    assert g(train).isdisjoint(g(val)), "LEAK train/val!"
    assert g(val).isdisjoint(g(test)), "LEAK val/test!"
    return train, val, test


# ── Loop training (identik run_fold di train_specialist_expert) ──────

def train_retrained(train_df, val_df, test_df, tag, ckpt_path, cfg, args, init_ckpt=None):
    bf.set_deterministic(args.seed)
    os.makedirs(CKPT_DIR, exist_ok=True)
    train_aug, val_aug = bf.build_augmentations(cfg.img_size, seed=args.seed)
    train_loader = make_loader(train_df, cfg, train_aug, shuffle=True, oversample=True)
    val_loader = make_loader(val_df, cfg, val_aug, shuffle=False)
    test_loader = make_loader(test_df, cfg, val_aug, shuffle=False)

    print(f"\n{'='*70}\n[RETRAIN] {tag}: train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}\n{'='*70}")

    class_weights = bf.compute_inverse_freq_weights(
        train_df["label"].values, num_classes=cfg.num_classes).to(DEVICE)

    mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,
                     mode="batch", label_smoothing=0.1, num_classes=cfg.num_classes)

    model = MultimodalClassifier(BACKBONE, num_classes=cfg.num_classes,
                                 meta_dim=15, head_dropout=0.3,
                                 img_size=cfg.img_size).to(DEVICE)

    if init_ckpt and os.path.exists(init_ckpt):
        sd = torch.load(init_ckpt, map_location="cpu", weights_only=False)["model_state"]
        if args.img_size:
            for k in [k for k in sd if "pos_embed" in k]:
                sd.pop(k, None)
                print(f"[warm-start] drop {k} (resolusi beda -> interpolasi DINOv2 dynamic_img_size)")
        # buang key beda-bentuk (mis. head num_classes berbeda: refiner k-kelas vs champion 7)
        msd = model.state_dict()
        shape_drop = [k for k in list(sd) if k in msd and sd[k].shape != msd[k].shape]
        for k in shape_drop:
            sd.pop(k)
        if shape_drop:
            print(f"[warm-start] drop {len(shape_drop)} key beda-bentuk (reinit): {shape_drop}")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[warm-start] init dari {os.path.basename(init_ckpt)}: "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    elif init_ckpt:
        print(f"[warm-start] SKIP — ckpt tidak ada: {init_ckpt}")

    criterion = SoftTargetCrossEntropyWeighted(
        class_weights=class_weights, gamma=getattr(args, "focal_gamma", 0.0)).to(DEVICE)
    if getattr(args, "focal_gamma", 0.0) > 0:
        print(f"[focal] gamma={args.focal_gamma} (emphasize contoh sulit/salah-prediksi)")
    supcon_criterion = SupConLoss(temperature=0.1).to(DEVICE)

    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 5e-5 * BACKBONE_LR_SCALE},
        {"params": list(model.meta_embed.parameters()) + list(model.fusion.parameters()),
         "lr": 5e-5, "weight_decay": 1e-3},
        {"params": list(model.head.parameters()) + list(model.proj_head.parameters()), "lr": 5e-5},
    ], weight_decay=1e-4)

    steps_per_epoch = len(train_loader)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / max(1, warmup_steps)))
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)

    max_epochs, patience = args.max_epochs, args.patience
    best_f1, no_improve, global_step = 0.0, 0, 0
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        in_warmup = global_step < warmup_steps
        model.train()
        total_loss, n = 0.0, 0
        for i, (imgs, metas, labels) in enumerate(train_loader):
            imgs, metas = imgs.to(DEVICE), metas.to(DEVICE)
            labels = labels.to(DEVICE)
            valid = labels >= 0
            if valid.sum() < 2:
                continue
            imgs_v, metas_v, labels_v = imgs[valid], metas[valid], labels[valid]
            if imgs_v.shape[0] % 2 == 1:
                imgs_v, metas_v, labels_v = imgs_v[:-1], metas_v[:-1], labels_v[:-1]
            if imgs_v.shape[0] < 2:
                continue

            loss_supcon = 0.0
            if args.lambda_supcon > 0:
                _, features = model(imgs_v, metas_v, return_features=True)
                loss_supcon = supcon_criterion(features, labels_v)

            imgs_mixed, soft_targets = mixup_fn(imgs_v.clone(), labels_v)
            logits = model(imgs_mixed, metas_v, return_features=False)
            loss = criterion(logits, soft_targets) + (args.lambda_supcon * loss_supcon)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if in_warmup:
                warmup_scheduler.step()
            total_loss += loss.item() * imgs_mixed.shape[0]
            n += imgs_mixed.shape[0]
        global_step += steps_per_epoch
        if not in_warmup:
            cosine_scheduler.step()

        model.eval()
        val_preds, val_labs = [], []
        with torch.no_grad():
            for imgs, metas, labels in val_loader:
                imgs, metas = imgs.to(DEVICE), metas.to(DEVICE)
                logits = model(imgs, metas)
                valid = labels >= 0
                val_preds.extend(logits[valid].argmax(1).cpu().numpy())
                val_labs.extend(labels[valid].numpy())
        val_preds, val_labs = np.array(val_preds), np.array(val_labs)
        val_f1 = f1_score(val_labs, val_preds, average="macro", zero_division=0)
        val_bal = balanced_accuracy_score(val_labs, val_preds)

        is_best = val_f1 > best_f1
        flag = ""
        if is_best:
            best_f1, no_improve, flag = val_f1, 0, "*"
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_f1": val_f1, "tag": tag}, ckpt_path)
        else:
            no_improve += 1
        print(f"[{tag}] Ep {epoch:3d}/{max_epochs} | train_loss={total_loss/max(n,1):.4f} "
              f"| val_bal={val_bal:.4f} val_f1={val_f1:.4f} (best={best_f1:.4f}) "
              f"| {time.time()-t0:.0f}s {flag}")
        if no_improve >= patience:
            print(f"[{tag}] Early stopping pada epoch {epoch}.")
            break

    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    test_preds, test_labs = [], []
    with torch.no_grad():
        for imgs, metas, labels in test_loader:
            imgs, metas = imgs.to(DEVICE), metas.to(DEVICE)
            logits = model(imgs, metas)
            valid = labels >= 0
            test_preds.extend(logits[valid].argmax(1).cpu().numpy())
            test_labs.extend(labels[valid].numpy())
    test_preds, test_labs = np.array(test_preds), np.array(test_labs)
    test_f1 = f1_score(test_labs, test_preds, average="macro", zero_division=0)
    test_bal = balanced_accuracy_score(test_labs, test_preds)
    print(f"[{tag}] TEST: acc={(test_preds==test_labs).mean():.4f} "
          f"bal_acc={test_bal:.4f} f1_macro={test_f1:.4f} (epoch={ck['epoch']})")
    return dict(tag=tag, test_acc=float((test_preds==test_labs).mean()),
                test_bal=float(test_bal), test_f1=float(test_f1), best_epoch=int(ck["epoch"]))


# ── Evaluasi MoE hasil retrain (protokol identik champion) ───────────────

def eval_moe_retrained(test_df, cfg, args):
    set_active_meta_fields("localization")
    paths = {"gen_pure": GEN_CKPT, **EXP_CKPT}
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        print(f"SKIP eval — checkpoint belum ada: {missing}")
        return None

    _, val_aug = bf.build_augmentations(cfg.img_size, seed=args.seed)
    loader = make_loader(test_df, cfg, val_aug, shuffle=False)

    models = {}
    for key, path in paths.items():
        m = MultimodalClassifier(BACKBONE, num_classes=cfg.num_classes,
                                 meta_dim=15, img_size=cfg.img_size).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False)["model_state"],
                          strict=False)
        m.eval()
        models[key] = m

    print("Extracting TTA probs (BCN test)...")
    probs_tta = {k: [] for k in models}
    y = []
    with torch.no_grad():
        for imgs, metas, labels in loader:
            imgs, metas = imgs.to(DEVICE), metas.to(DEVICE)
            y.extend(labels.numpy())
            views = tta_views(imgs)
            for m_name, model in models.items():
                p_sum = torch.zeros((imgs.shape[0], cfg.num_classes), device=DEVICE)
                for v in views:
                    p_sum += torch.softmax(model(v, metas), dim=1)
                probs_tta[m_name].extend((p_sum / len(views)).cpu().numpy())
    y = np.array(y)
    gen = np.array(probs_tta["gen_pure"])
    exp = build_exp_arrays(probs_tta, test_df, head_ver="head")  # expert hasil retrain disimpan dengan key "head"

    preds_gen = gen.argmax(1)
    preds_moe = (0.8 * exp + 0.2 * gen).argmax(1)
    preds_moe07 = (0.7 * exp + 0.3 * gen).argmax(1)

    def macro(preds):
        return float(f1_score(y, preds, average="macro", zero_division=0))
    res = {
        "n": len(y),
        "gen_macro": macro(preds_gen),
        "moe_macro_080": macro(preds_moe),
        "moe_macro_070": macro(preds_moe07),
        "gen_bacc": float(balanced_accuracy_score(y, preds_gen)),
        "moe_bacc_080": float(balanced_accuracy_score(y, preds_moe)),
        "moe_acc_080": float((preds_moe == y).mean()),
        "delta_moe_gen": macro(preds_moe) - macro(preds_gen),
    }
    print(f"\n[EVAL RETRAINED @ BCN test n={res['n']}]")
    print(f"  gen_macro          = {res['gen_macro']:.4f}")
    print(f"  MoE static 0.8     = {res['moe_macro_080']:.4f} (Δ+{res['delta_moe_gen']:+.4f} vs gen)")
    print(f"  MoE static 0.7     = {res['moe_macro_070']:.4f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_seed", type=int, default=0,
                    help="Seed split lesion-grouped (0 = ikuti --seed). Wajib 42 utk model ensemble yang dibandingkan di test seed-42")
    ap.add_argument("--img_mode", default="raw",
                    help="raw | crop | clean | clahe | crop_clahe | clahe_crop_bil | pipeline generik spt 'clahe>crop>bil' (lihat birnet_prep_bcn_pipelines.py)")
    ap.add_argument("--max_epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lambda_supcon", type=float, default=0.1)
    ap.add_argument("--out", default="birnet_retrain_bcn_results.json")
    ap.add_argument("--skip_train", action="store_true",
                    help="Lewati training, langsung eval dari checkpoint yg ada")
    ap.add_argument("--gen_gate", type=float, default=0.0,
                    help="GEN test_f1 di bawah nilai ini -> SKIP expert & MoE (hemat ~75 pct kompute). 0 = off")
    ap.add_argument("--warm_start", action="store_true",
                    help="Init GEN/expert dari champion HAM fold0 (mendekati HAM champion), lalu retrain BCN")
    ap.add_argument("--gen_only", action="store_true",
                    help="Hanya GEN (tanpa expert, tanpa MoE eval)")
    ap.add_argument("--skip_gen", action="store_true",
                    help="GEN sudah ada di GEN_CKPT -> latih expert saja + eval MoE")
    ap.add_argument("--noise_mask", default="",
                    help="JSON {image_id, noise} utk membuang sampel train curiga (bukan evaluasi)")
    ap.add_argument("--rare_dup", type=int, default=1,
                    help="Ulangi baris kelas langka (df/vasc) N kali di train (N=1 off) utk lawan imbalance ekstrem")
    ap.add_argument("--boost_mid", action="store_true",
                    help="Juga ulangi akiec/bkl (2x) di train utk membantu kelas ambiguity")
    ap.add_argument("--img_size", type=int, default=0,
                    help="Resolusi input (0 = default cfg, e.g. 224). 392 utk tekstur granular akiec/bcc/bkl")
    ap.add_argument("--batch_size", type=int, default=0,
                    help="Override BATCH_SIZE (diperlukan utk img_size 392 pada VRAM 31GB)")
    ap.add_argument("--pool_ham", action="store_true",
                    help="Tambahkan HAM10000 RAW (image asli) ke train BCN (test BCN seed-42 tetap). "
                         "HAM menyediakan sampel kelas langka df/vasc yg kurang di BCN train.")
    ap.add_argument("--focal_gamma", type=float, default=0.0,
                    help="Focal loss gamma (>0 = emphasize contoh sulit/salah-prediksi; 2.0 lazim). 0=CE biasa.")
    ap.add_argument("--pool_rare", default="",
                    help="Suplai TERTARGET kelas langka dari sumber dermoskopi eksternal. "
                         "Label ids koma, mis. '3,5,6' (akiec,df,vasc). Kosong=off.")
    ap.add_argument("--pool_rare_src", default="mskcc,hiba,ham",
                    help="Sumber suplai (koma): mskcc,hiba,ham. Default ketiganya (dermoskopi).")
    args = ap.parse_args()

    set_active_meta_fields("localization")
    cfg = bf.Config()
    if args.img_size:
        cfg.img_size = args.img_size
        print(f"[img_size override] resolusi input = {cfg.img_size}")
    global BATCH_SIZE
    if args.batch_size:
        BATCH_SIZE = args.batch_size
        print(f"[batch_size override] BATCH_SIZE = {BATCH_SIZE}")

    bcn = build_bcn_df(args.img_mode)
    print(f"[BCN] terlabel={len(bcn)} img_mode={args.img_mode} "
          f"dist={bcn['label'].value_counts().sort_index().to_dict()}")
    # split BCN bawaan BOCOR lesi (train∩test=915) -> gunakan lesion-grouped
    split_seed = args.seed if args.split_seed == 0 else args.split_seed
    train_df, val_df, test_df = lesion_grouped_split(bcn, split_seed)
    print(f"[lesion-grouped split, seed={split_seed}] "
          f"train={len(train_df)} val={len(val_df)} test={len(test_df)} (0 kebocoran lesi terverifikasi)")

    if args.pool_ham:
        paths = bf.DatasetPaths(cfg)
        ham = bf.build_ham10000_dataframe(paths)
        ham = ham[["image_id", "img_path", "label", "localization", "age", "sex", "dataset", "lesion_id"]]
        before = len(train_df)
        train_df = pd.concat([train_df, ham], ignore_index=True)
        print(f"[pool-ham] HAM10000 RAW +{len(ham)} -> train {before} -> {len(train_df)} "
              f"(test BCN seed-42 TETAP utk banding adil)")

    if args.pool_rare:
        from ame_derm.external_validation.pool_external_rare import build_external_rare_df
        rare_ids = [int(x) for x in args.pool_rare.split(",")]
        srcs = tuple(s.strip() for s in args.pool_rare_src.split(","))
        ext = build_external_rare_df(rare_ids, srcs, cfg=cfg)
        # ANTI-LEAK: lesion_id eksternal memakai prefix sumber -> pasti disjoint dari BCN test
        test_les = set(test_df["lesion_id"].astype(str))
        assert set(ext["lesion_id"].astype(str)).isdisjoint(test_les), "LEAK: lesion eksternal di test!"
        before = len(train_df)
        train_df = pd.concat([train_df, ext], ignore_index=True)
        dist = ext.groupby("label").size().to_dict()
        print(f"[pool-rare] suplai TERTARGET {rare_ids} dari {srcs}: +{len(ext)} "
              f"(per-label {dist}) -> train {before} -> {len(train_df)} (test BCN seed-42 TETAP)")

    if args.noise_mask:
        import json as _json
        nm = _json.load(open(args.noise_mask))
        drop = set(str(i) for i, flg in zip(nm["image_id"], nm["noise"]) if flg)
        before = len(train_df)
        train_df = train_df[~train_df["image_id"].astype(str).isin(drop)].reset_index(drop=True)
        print(f"[noise-mask {args.noise_mask}] drop {before - len(train_df)} sampel train "
              f"({100*(before-len(train_df))/before:.1f}%) -> keep={len(train_df)}")

    if args.rare_dup > 1 or args.boost_mid:
        extra = []
        rare = {5, 6}  # df, vasc
        mid = {3, 4}   # akiec, bkl
        for _, r in train_df.iterrows():
            if r["label"] in rare:
                extra.extend([r] * (args.rare_dup - 1))
            elif args.boost_mid and r["label"] in mid:
                extra.extend([r] * 1)
        train_df = pd.concat([train_df, pd.DataFrame(extra)], ignore_index=True)
        print(f"[oversample-rare] dup_langka={args.rare_dup}x "
              f"boost_mid={args.boost_mid} -> train={len(train_df)} "
              f"dist={train_df['label'].value_counts().sort_index().to_dict()}")

    results = {}

    if not args.skip_train:
        # 1) Generalist
        if not args.skip_gen:
            results["gen"] = train_retrained(train_df, val_df, test_df, "GEN",
                                             GEN_CKPT, cfg, args,
                                             init_ckpt=CHAMP_GEN_CKPT if args.warm_start else None)
            json.dump(results, open(args.out, "w"), indent=2)
            print(f"[gate] GEN test_f1={results['gen']['test_f1']:.4f} | "
                  f"gen_gate={args.gen_gate} | warm_start={args.warm_start}")
        else:
            print("[skip_gen] pakai GEN existing di GEN_CKPT (tanpa retrain GEN).")

        # 2) Gate: GEN kurang menjanjikan -> hentikan sebelum expert (hemat 75%).
        gen_f1 = results.get("gen", {}).get("test_f1")
        if (not args.gen_only and gen_f1 is not None
                and args.gen_gate > 0 and gen_f1 < args.gen_gate):
            print(f"[gate] SKIP expert & MoE — GEN {results['gen']['test_f1']:.4f} < "
                  f"{args.gen_gate}. Cukup GEN saja.")
            results["_skipped"] = "expert,moe by gen_gate"
            results["_config"] = {"seed": args.seed, "split_seed": split_seed,
                                  "img_mode": args.img_mode,
                                  "max_epochs": args.max_epochs, "patience": args.patience,
                                  "lambda_supcon": args.lambda_supcon,
                                  "gen_gate": args.gen_gate, "warm_start": args.warm_start}
            json.dump(results, open(args.out, "w"), indent=2)
            print(f"Hasil (gate) disimpan: {args.out}")
            return

        # 3) Experts (jika bukan gen_only)
        if not args.gen_only:
            for expert, locs in EXPERT_MAPPING.items():
                tdf = train_df[train_df["localization"].isin(locs)].reset_index(drop=True)
                vdf = val_df[val_df["localization"].isin(locs)].reset_index(drop=True)
                tsf = test_df[test_df["localization"].isin(locs)].reset_index(drop=True)
                print(f"\n[{expert.upper()}] subset train={len(tdf)} val={len(vdf)} test={len(tsf)}")
                if len(tdf) == 0:
                    print(f"SKIP {expert} — subset kosong.")
                    continue
                results[expert] = train_retrained(
                    tdf, vdf, tsf, expert.upper(), EXP_CKPT[expert], cfg, args,
                    init_ckpt=CHAMP_EXP_CKPT.get(expert) if args.warm_start else None)
            json.dump(results, open(args.out, "w"), indent=2)
            print(f"\nTraining results disimpan: {args.out}")

    # 4) Evaluasi MoE hasil retrain (dilewati saat gen_only)
    if args.gen_only:
        results["_config"] = {"seed": args.seed, "split_seed": split_seed, "img_mode": args.img_mode,
                              "max_epochs": args.max_epochs, "patience": args.patience,
                              "lambda_supcon": args.lambda_supcon, "gen_only": True,
                              "warm_start": args.warm_start}
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\nHasil (gen_only) disimpan: {args.out}")
        return
    moe = eval_moe_retrained(test_df, cfg, args)
    if moe is not None:
        results["moe_eval"] = moe
        results["_config"] = {"seed": args.seed, "split_seed": split_seed, "img_mode": args.img_mode,
                              "max_epochs": args.max_epochs, "patience": args.patience,
                              "lambda_supcon": args.lambda_supcon, "warm_start": args.warm_start}
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\nHasil lengkap disimpan: {args.out}")


if __name__ == "__main__":
    main()
