"""FINAL Phase 2 classification pipeline, built on top of the finalized
Phase 1 segmentation (birnet_seg_phase1_final.py, dice=0.9099) and every
methodological fix established but not yet fully assembled earlier this
session:

  1. Pre-cropped images (via birnet_generate_phase1_crops.py) instead of
     on-the-fly mask prediction -- the new Phase 1 ensemble is too heavy
     to run per-image per-epoch.
  2. Proper 5-fold GroupKFold cross-validation (bf.stratified_ham_kfold),
     not the old single-fold stratified_ham_split (which silently only
     ever used fold 0 of a 7-fold split).
  3. Checkpoint selection by macro-F1, not balanced accuracy (balanced
     accuracy is recall-only and blind to precision damage from a class
     being over-predicted at another's expense).
  4. Inverse-frequency class weighting (bf.compute_inverse_freq_weights),
     computed fresh from each fold's OWN train-set label distribution.
  5. Augmentation applied only to the train split (val/test use the
     deterministic val_aug).
  6. Full determinism (bf.set_deterministic) including the albumentations-
     seed and cuDNN fixes discovered earlier this session.
  7. Discriminative LR + warmup (kept from train_classifier_cropped_v2.py --
     a real, already-verified fix for the "peaks at epoch 2-3 then
     degrades" fine-tuning instability pattern).

Reports mean +/- std of test F1/accuracy/balanced-accuracy ACROSS ALL 5
FOLDS (every image used as test exactly once), not a single lucky/unlucky
split.

Usage:
    python train_classifier_phase2_final.py --backbone vit_base_patch16_224
    python train_classifier_phase2_final.py --backbone efficientnet_b5
    python train_classifier_phase2_final.py --backbone vit_base_patch16_224 --fold 2   # single fold only
"""
import argparse
import os
import time
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.data import Dataset, DataLoader

import birnet_final as bf
from train_classifier_generic import GenericClassifier, evaluate_full

CROP_DIR = "BIRNet_Research/outputs/cropped_phase1_final"
CKPT_DIR = "BIRNet_Research/outputs/checkpoints/phase2_final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
BACKBONE_LR_SCALE = 0.1
WARMUP_EPOCHS = 2


class SkinLesionDatasetPrecomputedCrop(Dataset):
    """Loads ALREADY-CROPPED images (from birnet_generate_phase1_crops.py),
    applies the same CLAHE + bilateral-filter preprocessing as every prior
    crop experiment, then resizes. No mask prediction at load time."""

    def __init__(self, df, img_size, transform=None, oversample=False,
                 target_per_class=None, num_classes=7, preprocess=True):
        self.img_size = img_size
        self.transform = transform
        self.preprocess = preprocess  # False = RAW crop (ablasi tunjukkan CLAHE+bilateral MERUGIKAN)
        entries = [(row.image_id, None, row.label) for row in df.itertuples()]
        if oversample and target_per_class is not None:
            entries = bf.SkinLesionDataset._build_oversampled_index(entries, target_per_class, num_classes)
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        if len(entry) == 4:
            image_id, _, label, do_aug = entry
        else:
            image_id, _, label = entry
            do_aug = False

        img_path = os.path.join(CROP_DIR, f"{image_id}.jpg")
        img = np.array(bf.Image.open(img_path).convert("RGB")).astype(np.uint8)
        if self.preprocess:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
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
        return img_t, torch.zeros(1), int(label)


def train_one_epoch_discriminative(model, loader, optimizer, criterion, device, warmup_scheduler=None):
    model.train()
    total_loss, n = 0.0, 0
    for imgs, _, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        valid = labels >= 0
        if valid.sum() == 0:
            continue
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits[valid], labels[valid])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        total_loss += loss.item() * valid.sum().item()
        n += valid.sum().item()
    return total_loss / max(n, 1)


def run_fold(fold_i, train_df, val_df, test_df, backbone, seed, cfg):
    bf.set_deterministic(seed)
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"cls_{backbone}_fold{fold_i}_best.pth")

    train_aug, val_aug = bf.build_augmentations(cfg.img_size, seed=seed)
    train_ds = SkinLesionDatasetPrecomputedCrop(train_df, cfg.img_size, transform=train_aug,
                                                 oversample=True, target_per_class=cfg.oversample_target_per_class,
                                                 num_classes=cfg.num_classes)
    val_ds = SkinLesionDatasetPrecomputedCrop(val_df, cfg.img_size, transform=val_aug)
    test_ds = SkinLesionDatasetPrecomputedCrop(test_df, cfg.img_size, transform=val_aug)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"\n{'='*70}\n[FOLD {fold_i}] backbone={backbone} seed={seed} "
          f"train={len(train_ds)} (oversampled dr {len(train_df)}) val={len(val_ds)} test={len(test_ds)}\n{'='*70}")

    class_weights = bf.compute_inverse_freq_weights(train_df["label"].values, num_classes=cfg.num_classes).to(DEVICE)
    print(f"Inverse-freq class weights: {class_weights.cpu().numpy().round(3)}")

    model = GenericClassifier(backbone, num_classes=cfg.num_classes).to(DEVICE)
    criterion = bf.LabelSmoothingFocalLoss(gamma=2.0, smoothing=0.1, weight=class_weights).to(DEVICE)
    is_transformer = "vit" in backbone or "swin" in backbone or "eva" in backbone
    base_lr = 5e-5 if is_transformer else 1e-4

    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": base_lr * BACKBONE_LR_SCALE},
        {"params": model.head.parameters(), "lr": base_lr},
    ], weight_decay=1e-4)

    steps_per_epoch = len(train_loader)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / max(1, warmup_steps)))
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)

    max_epochs, patience = cfg.cls_max_epochs, cfg.cls_patience
    best_f1, no_improve = 0.0, 0
    global_step = 0
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        in_warmup = global_step < warmup_steps
        train_loss = train_one_epoch_discriminative(
            model, train_loader, optimizer, criterion, DEVICE,
            warmup_scheduler=warmup_scheduler if in_warmup else None)
        global_step += steps_per_epoch
        val_acc, val_bal, val_f1 = evaluate_full(model, val_loader, DEVICE)
        if not in_warmup:
            cosine_scheduler.step()
        is_best = val_f1 > best_f1
        flag = ""
        if is_best:
            best_f1, no_improve, flag = val_f1, 0, "*"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_f1": val_f1}, ckpt_path)
        else:
            no_improve += 1
        print(f"[F{fold_i}] Ep {epoch:3d}/{max_epochs} | train_loss={train_loss:.4f} "
              f"| val_acc={val_acc:.4f} val_bal={val_bal:.4f} val_f1={val_f1:.4f} (best={best_f1:.4f}) "
              f"| {time.time()-t0:.0f}s {flag}")
        if no_improve >= patience:
            print(f"[F{fold_i}] Early stopping pada epoch {epoch}.")
            break

    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state"])
    test_acc, test_bal, test_f1 = evaluate_full(model, test_loader, DEVICE)
    print(f"[FOLD {fold_i}] TEST: acc={test_acc:.4f} bal_acc={test_bal:.4f} f1_macro={test_f1:.4f} "
          f"(checkpoint epoch={ck['epoch']})")
    return dict(fold=fold_i, test_acc=test_acc, test_bal=test_bal, test_f1=test_f1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=[
        "vit_base_patch16_224", "efficientnet_b5", "resnet50", "convnext_tiny", "swin_tiny_patch4_window7_224"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None, help="Run only this fold index (0-based); omit to run all.")
    args = parser.parse_args()

    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    ham_df = bf.build_ham10000_dataframe(paths)
    print(f"Total HAM10000 (dengan crop tersedia): checking...")
    has_crop = ham_df["image_id"].apply(lambda iid: os.path.exists(os.path.join(CROP_DIR, f"{iid}.jpg")))
    print(f"  {has_crop.sum()}/{len(ham_df)} punya crop pre-computed")
    ham_df = ham_df[has_crop].reset_index(drop=True)

    results = []
    for fold_i, train_idx, val_idx, test_idx in bf.stratified_ham_kfold(ham_df, cfg, n_splits=args.n_splits):
        if args.fold is not None and fold_i != args.fold:
            continue
        train_df = ham_df.iloc[train_idx].reset_index(drop=True)
        val_df = ham_df.iloc[val_idx].reset_index(drop=True)
        test_df = ham_df.iloc[test_idx].reset_index(drop=True)
        result = run_fold(fold_i, train_df, val_df, test_df, args.backbone, args.seed, cfg)
        results.append(result)
        print(f"RESULT_SUMMARY backbone={args.backbone} fold={fold_i} "
              f"test_acc={result['test_acc']:.4f} test_bal={result['test_bal']:.4f} test_f1={result['test_f1']:.4f}")

    if len(results) > 1:
        accs = np.array([r["test_acc"] for r in results])
        bals = np.array([r["test_bal"] for r in results])
        f1s = np.array([r["test_f1"] for r in results])
        print(f"\n{'='*70}\n=== RINGKASAN {len(results)}-FOLD (backbone={args.backbone}) ===\n{'='*70}")
        print(f"acc     : mean={accs.mean():.4f} std={accs.std():.4f}")
        print(f"bal_acc : mean={bals.mean():.4f} std={bals.std():.4f}")
        print(f"f1_macro: mean={f1s.mean():.4f} std={f1s.std():.4f}")
        print(f"RESULT_SUMMARY_FINAL backbone={args.backbone} n_folds={len(results)} "
              f"mean_acc={accs.mean():.4f} mean_bal={bals.mean():.4f} mean_f1={f1s.mean():.4f} std_f1={f1s.std():.4f}")


if __name__ == "__main__":
    main()
