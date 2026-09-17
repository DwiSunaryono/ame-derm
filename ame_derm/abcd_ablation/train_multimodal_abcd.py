"""Latih generalist DINOv2 multimodal: localization (15D) + ABCD features (8D) = 23D meta.

Strategi: reuse seluruh pipeline train_multimodal_ablation (arsitektur cross-attention,
MixUp, oversample, class-weight, discriminative LR, early-stop F1) — hanya metadata
yang diperluas dari 15D (loc-only) → 23D (loc + ABCD).

8 Fitur ABCD (dari birnet_extract_abcd_features.py, sudah tersimpan di CSV):
  asymmetry_h, asymmetry_v  : asimetri geometri (MEL > NV)
  border_irreg, border_jagged: kekasaran border (AKIEC >> BKL, MEL > NV)
  color_std_r/g/b           : variegasi warna (MEL multi-warna)
  compactness               : kerapatan bentuk (NV lebih kompak)

Sinyal terukur dari data (HAM10000 per-kelas):
  MEL  : asymmetry_h=0.460, border_irreg=9.08, color_std_r=0.089
  NV   : asymmetry_h=0.351, border_irreg=7.23, color_std_r=0.075
  AKIEC: asymmetry_h=0.676, border_irreg=16.0  ← tertinggi!

Meta vector (urutan TETAP, kompatibel dengan train_multimodal_ablation):
  [0]     = loc (15D one-hot)  — identik dengan champion loc-only
  [15:23] = ABCD (8D float32) — BARU, normalized per-fitur ke [0,1]

Usage:
  .venv/bin/python3 train_multimodal_abcd.py --fold 0
  .venv/bin/python3 train_multimodal_abcd.py --fold 0 --backbone vit_large_patch14_reg4_dinov2
  .venv/bin/python3 train_multimodal_abcd.py --fold 0 --no_abcd  # baseline loc-only (ablasi)
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ame_derm.common import birnet_final as bf
from ame_derm.abcd_ablation import train_multimodal_ablation as tma
from ame_derm.abcd_ablation.train_multimodal_ablation import (
    SkinLesionDatasetMultimodal, MultimodalClassifier,
    train_one_epoch_multimodal,
    SoftTargetCrossEntropyWeighted,
)
from ame_derm.experts.train_classifier_generic import evaluate_full
from ame_derm.experts.train_classifier_phase2_final import CROP_DIR
from ame_derm.experts.train_classifier_phase2_mixup import BATCH_SIZE, BACKBONE_LR_SCALE, WARMUP_EPOCHS
from ame_derm.data_splits.representative_val_split import stratified_ham_kfold_fully_representative
from ame_derm.data_splits.diagnose_metadata_imbalance import detect_metadata_fields
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from timm.data import Mixup
import time

ABCD_CSV = "BIRNet_Research/outputs/abcd_features.csv"
CKPT_DIR_BASE = "BIRNet_Research/outputs/checkpoints/multimodal_abcd"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Normalisasi ABCD ke [0,1] berdasarkan statistik dataset (dari output ekstraksi)
# Diambil dari output: percentile 1% - 99% untuk robustness terhadap outlier
ABCD_NORM = {
    "asymmetry_h":  (0.0049, 1.0),
    "asymmetry_v":  (0.0052, 1.0),
    "border_irreg": (1.137,  50.0),   # clip outlier (max=327 ada extreme cases)
    "border_jagged":(3.223,  30.0),   # clip outlier (max=58)
    "color_std_r":  (0.0,    0.267),
    "color_std_g":  (0.0,    0.191),
    "color_std_b":  (0.0,    0.210),
    "compactness":  (0.028,  0.949),
}
ABCD_COLS = list(ABCD_NORM.keys())
ABCD_DIM = len(ABCD_COLS)  # 8

# Grup fitur ABCD untuk ablasi (nilai di luar grup di-nol, arsitektur tetap 23D)
GEOM_COLS = {"asymmetry_h", "asymmetry_v", "border_irreg", "border_jagged", "compactness"}
COLOR_COLS = {"color_std_r", "color_std_g", "color_std_b"}

def abcd_group_mask(group):
    if group == "none":       keep = set()
    elif group == "geometry": keep = GEOM_COLS
    elif group == "color":    keep = COLOR_COLS
    else:                     keep = set(ABCD_COLS)  # all
    return np.array([1.0 if c in keep else 0.0 for c in ABCD_COLS], dtype=np.float32)


def load_abcd_map(abcd_csv):
    """Muat CSV ABCD → dict {image_id: np.array(8,)}."""
    df = pd.read_csv(abcd_csv)
    abcd_map = {}
    for row in df.itertuples(index=False):
        vec = np.zeros(ABCD_DIM, dtype=np.float32)
        for i, col in enumerate(ABCD_COLS):
            val = getattr(row, col, 0.0)
            lo, hi = ABCD_NORM[col]
            vec[i] = float(np.clip((val - lo) / max(hi - lo, 1e-8), 0.0, 1.0))
        abcd_map[row.image_id] = vec
    print(f"ABCD map loaded: {len(abcd_map)} gambar")
    return abcd_map


class SkinLesionDatasetABCD(SkinLesionDatasetMultimodal):
    """Dataset multimodal dengan metadata LOC (15D) + ABCD (8D) = 23D.
    Jika no_abcd=True, hanya pakai LOC (15D) — untuk ablasi bersih."""

    def __init__(self, df, img_size, abcd_map, transform=None,
                 oversample=False, target_per_class=None, num_classes=7,
                 abcd_mask=None):
        super().__init__(df, img_size, transform=transform,
                         oversample=oversample, target_per_class=target_per_class,
                         num_classes=num_classes)
        self.abcd_map = abcd_map
        # mask 8D (0/1): fitur di luar grup di-nol. None = semua aktif.
        self.abcd_mask = abcd_mask if abcd_mask is not None else np.ones(ABCD_DIM, dtype=np.float32)

    def __getitem__(self, idx):
        img_t, _, label = super().__getitem__(idx)
        image_id = self.entries[idx][0]
        # SELALU 23D: loc (15) + abcd (8, grup non-aktif di-nol). Arsitektur identik -> ablasi bersih.
        loc_vec = self.meta_map.get(image_id, np.zeros(tma.META_DIM, dtype=np.float32))
        abcd_vec = self.abcd_map.get(image_id, np.zeros(ABCD_DIM, dtype=np.float32)) * self.abcd_mask
        meta_t = torch.from_numpy(np.concatenate([loc_vec, abcd_vec]))
        return img_t, meta_t, label


# === FIX BUG-1 (RAW): patch getitem induk agar BYPASS CLAHE+bilateral (baca crop RAW),
# konsisten dgn champion. super().__getitem__ di SkinLesionDatasetABCD memakai versi ini. ===
import cv2 as _cv2

def _raw_multimodal_getitem(self, idx):
    entry = self.entries[idx]
    if len(entry) == 4:
        image_id, _, label, do_aug = entry
    else:
        image_id, _, label = entry; do_aug = False
    img = np.array(bf.Image.open(os.path.join(CROP_DIR, f"{image_id}.jpg")).convert("RGB")).astype(np.uint8)
    img_np = _cv2.resize(img, (self.img_size, self.img_size), interpolation=_cv2.INTER_LINEAR)
    if do_aug:
        if np.random.rand() > 0.5: img_np = _cv2.flip(img_np, 1)
        if np.random.rand() > 0.5: img_np = _cv2.flip(img_np, 0)
        img_np = np.rot90(img_np, np.random.randint(0, 4)).copy()
    if label is None or (isinstance(label, float) and np.isnan(label)):
        label = 0
    if self.transform:
        img_t = self.transform(image=img_np, mask=np.zeros((self.img_size, self.img_size), dtype=np.float32))["image"]
    else:
        img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
    base_id = image_id.split("_synth")[0] if "_synth" in image_id else image_id
    meta_vec = self.meta_map.get(image_id, self.meta_map.get(base_id, np.zeros(tma.META_DIM, dtype=np.float32)))
    return img_t, torch.from_numpy(meta_vec), int(label)

tma.SkinLesionDatasetMultimodal.__getitem__ = _raw_multimodal_getitem  # aktifkan RAW


def run_fold_abcd(fold_i, train_df, val_df, test_df, backbone, seed, cfg,
                  abcd_map, abcd_group="all", ckpt_dir=None):
    bf.set_deterministic(seed)
    suffix = f"abcd_{abcd_group}"
    abcd_mask = abcd_group_mask(abcd_group)
    meta_dim = tma.META_DIM + ABCD_DIM  # SELALU 23 (placebo=zeros -> arsitektur identik utk ablasi bersih)
    if ckpt_dir is None:
        ckpt_dir = CKPT_DIR_BASE
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"cls_{backbone}_{suffix}_fold{fold_i}_best.pth")
    done_marker = os.path.join(ckpt_dir, f"cls_{backbone}_{suffix}_fold{fold_i}_DONE")

    train_aug, val_aug = bf.build_augmentations(cfg.img_size, seed=seed)

    def make_ds(df, aug, oversample=False):
        return SkinLesionDatasetABCD(
            df, cfg.img_size, abcd_map=abcd_map, transform=aug,
            oversample=oversample, target_per_class=cfg.oversample_target_per_class,
            num_classes=cfg.num_classes, abcd_mask=abcd_mask)

    train_ds = make_ds(train_df, train_aug, oversample=True)
    val_ds = make_ds(val_df, val_aug)
    test_ds = make_ds(test_df, val_aug)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    print(f"\n{'='*70}")
    print(f"[FOLD {fold_i}] backbone={backbone} meta_dim={meta_dim} ({suffix})")
    print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"{'='*70}")

    if os.path.exists(done_marker):
        print(f"[F{fold_i}] Sudah SELESAI (DONE marker). Langsung eval TEST.")
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = MultimodalClassifier(backbone, num_classes=cfg.num_classes,
                                     meta_dim=meta_dim).to(DEVICE)
        model.load_state_dict(ck["model_state"])
        test_acc, test_bal, test_f1 = evaluate_full(model, test_loader, DEVICE)
        print(f"[FOLD {fold_i}] TEST: acc={test_acc:.4f} f1_macro={test_f1:.4f}")
        return dict(fold=fold_i, test_acc=test_acc, test_bal=test_bal, test_f1=test_f1)

    class_weights = bf.compute_inverse_freq_weights(
        train_df["label"].values, num_classes=cfg.num_classes).to(DEVICE)
    print(f"Class weights: {class_weights.cpu().numpy().round(3)}")

    mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,
                     mode="batch", label_smoothing=0.1, num_classes=cfg.num_classes)
    model = MultimodalClassifier(backbone, num_classes=cfg.num_classes,
                                 meta_dim=meta_dim).to(DEVICE)
    criterion = SoftTargetCrossEntropyWeighted(class_weights=class_weights).to(DEVICE)

    is_transformer = any(x in backbone for x in ["vit", "swin", "eva", "dinov2"])
    base_lr = 5e-5 if is_transformer else 1e-4
    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": base_lr * BACKBONE_LR_SCALE},
        {"params": list(model.meta_embed.parameters()) +
                   list(model.fusion.parameters()) +
                   list(model.head.parameters()), "lr": base_lr},
    ], weight_decay=1e-4)

    steps_per_epoch = len(train_loader)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    warmup_sched = LambdaLR(optimizer,
                            lr_lambda=lambda s: min(1.0, (s + 1) / max(1, warmup_steps)))
    cosine_sched = CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)

    max_epochs, patience = cfg.cls_max_epochs, cfg.cls_patience
    best_f1, no_improve, global_step = 0.0, 0, 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        in_warmup = global_step < warmup_steps
        train_loss = train_one_epoch_multimodal(
            model, train_loader, optimizer, criterion, mixup_fn, DEVICE,
            warmup_scheduler=warmup_sched if in_warmup else None)
        global_step += steps_per_epoch
        val_acc, val_bal, val_f1 = evaluate_full(model, val_loader, DEVICE)
        if not in_warmup:
            cosine_sched.step()
        is_best = val_f1 > best_f1
        flag = ""
        if is_best:
            best_f1, no_improve, flag = val_f1, 0, "*"
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_f1": val_f1}, ckpt_path)
        else:
            no_improve += 1
        print(f"[F{fold_i}] Ep {epoch:3d}/{max_epochs} | loss={train_loss:.4f} "
              f"| val_f1={val_f1:.4f} (best={best_f1:.4f}) | {time.time()-t0:.0f}s {flag}")
        if no_improve >= patience:
            print(f"[F{fold_i}] Early stopping epoch {epoch}.")
            break

    with open(done_marker, "w") as f:
        f.write("done\n")

    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state"])
    test_acc, test_bal, test_f1 = evaluate_full(model, test_loader, DEVICE)
    print(f"[FOLD {fold_i}] TEST: acc={test_acc:.4f} bal_acc={test_bal:.4f} "
          f"f1_macro={test_f1:.4f} (ckpt ep={ck['epoch']})")
    return dict(fold=fold_i, test_acc=test_acc, test_bal=test_bal, test_f1=test_f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_base_patch14_reg4_dinov2")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--abcd_group", choices=["all", "geometry", "color", "none"], default="all",
                    help="Grup ABCD aktif (sisanya di-nol, arsitektur tetap 23D). none=placebo")
    ap.add_argument("--no_abcd", action="store_true", help="Alias --abcd_group none (placebo)")
    ap.add_argument("--abcd_csv", default=ABCD_CSV)
    ap.add_argument("--ckpt_dir", default=None)
    args = ap.parse_args()
    if args.no_abcd:
        args.abcd_group = "none"

    # Aktifkan metadata: HANYA localization (konsisten dengan champion)
    tma.set_active_meta_fields("localization")
    print(f"Meta LOC {tma.META_DIM}D + ABCD {ABCD_DIM}D = {tma.META_DIM + ABCD_DIM}D (SELALU) "
          f"| grup ABCD aktif: {args.abcd_group}")

    # Load ABCD map
    abcd_map = load_abcd_map(args.abcd_csv)

    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    ham_df = bf.build_ham10000_dataframe(paths)
    has_crop = ham_df["image_id"].apply(
        lambda iid: os.path.exists(os.path.join(CROP_DIR, f"{iid}.jpg")))
    ham_df = ham_df[has_crop].reset_index(drop=True)
    print(f"{len(ham_df)} gambar HAM10000 dengan crop tersedia")

    # Split SAMA dgn champion (fully_representative, stratify label+dataset) -> angka comparable
    bucketed, _ = detect_metadata_fields(ham_df, 9)
    overlap = [c for c in bucketed.columns if c in ham_df.columns]
    ham_meta = pd.concat([ham_df.drop(columns=overlap), bucketed], axis=1)

    results = []
    for _rep, fold_i, train_idx, val_idx, test_idx in stratified_ham_kfold_fully_representative(
            ham_meta, cfg, ["dataset"], n_splits=5, n_repeats=1):
        if fold_i != args.fold:
            continue
        train_df = ham_df.iloc[train_idx].reset_index(drop=True)
        val_df = ham_df.iloc[val_idx].reset_index(drop=True)
        test_df = ham_df.iloc[test_idx].reset_index(drop=True)

        result = run_fold_abcd(fold_i, train_df, val_df, test_df,
                               args.backbone, args.seed, cfg,
                               abcd_map=abcd_map, abcd_group=args.abcd_group,
                               ckpt_dir=args.ckpt_dir)
        results.append(result)
        print(f"RESULT_SUMMARY backbone={args.backbone} variant=abcd_{args.abcd_group} "
              f"fold={fold_i} test_acc={result['test_acc']:.4f} "
              f"test_bal={result['test_bal']:.4f} test_f1={result['test_f1']:.4f}")


if __name__ == "__main__":
    main()
