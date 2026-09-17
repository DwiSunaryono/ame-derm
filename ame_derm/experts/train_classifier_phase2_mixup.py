"""Phase 2 classifier + MixUp/CutMix, confirmed as truly untested in this
project and sensible for HAM10000's extreme class imbalance (it imposes a
continuous decision boundary between majority classes like NV/MEL and
minority classes like DF/VASC, rather than letting the model memorize the
handful of minority samples outright).

Relies on timm.data.Mixup (mixup_alpha=0.2, cutmix_alpha=1.0, switch_prob=0.5,
built-in label_smoothing=0.1), which yields soft-label targets, combined
with a custom weighted soft-target cross-entropy that keeps this project's
inverse-frequency class weighting intact: a mixed sample's loss weight is
the same lambda-interpolation of the two source classes' weights as the soft
target itself, so class-weighting and MixUp are not merely stacked naively
but are mathematically consistent with one another.

In all other respects identical to train_classifier_phase2_final.py (same
crops, same 5-fold GroupKFold, same F1-macro checkpoint selection, same
discriminative LR+warmup), so the MixUp/CutMix effect stands out as the
only new variable.

Usage:
    python train_classifier_phase2_mixup.py --backbone vit_base_patch16_224 --fold 0
"""
import argparse

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.data import DataLoader
from timm.data import Mixup

from ame_derm.common import birnet_final as bf
from ame_derm.experts.train_classifier_generic import GenericClassifier, evaluate_full
from ame_derm.experts.train_classifier_phase2_final import SkinLesionDatasetPrecomputedCrop, CROP_DIR

CKPT_DIR = "BIRNet_Research/outputs/checkpoints/phase2_mixup"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
BACKBONE_LR_SCALE = 0.1
WARMUP_EPOCHS = 2


class SoftTargetCrossEntropyWeighted(nn.Module):
    """Cross-entropy against soft (MixUp/CutMix-blended) targets, where each
    sample's loss is weighted by the SAME lambda-interpolation of the class
    weights that generated its soft target -- this keeps inverse-frequency
    class weighting mathematically consistent with the MixUp blend instead
    of applying class weights as though targets were still one-hot."""

    def __init__(self, class_weights=None, gamma=0.0):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma  # >0 = focal loss (emphasize hard/misclassified); 0 = CE biasa (default)

    def forward(self, logits, soft_targets):
        log_probs = F.log_softmax(logits, dim=1)
        if self.gamma > 0:
            # per-class focal: (1-p_c)^gamma down-weights EASY examples, lifts HARD ones
            focal = (1.0 - log_probs.exp()).clamp(min=0.0).pow(self.gamma)
            loss_per_sample = -(soft_targets * focal * log_probs).sum(dim=1)
        else:
            loss_per_sample = -(soft_targets * log_probs).sum(dim=1)
        if self.class_weights is not None:
            sample_weight = (soft_targets * self.class_weights.unsqueeze(0)).sum(dim=1)
            loss_per_sample = loss_per_sample * sample_weight
        return loss_per_sample.mean()


def train_one_epoch_mixup(model, loader, optimizer, criterion, mixup_fn, device, warmup_scheduler=None,
                           manifold_mixup_alpha=0.0):
    """manifold_mixup_alpha>0 turns on Manifold Mixup (Verma et al. 2019):
    AFTER the usual pixel-space MixUp/CutMix (mixup_fn), the backbone
    features (pre-head) are interpolated once more with a random partner
    inside the same batch, a new lambda ~ Beta(alpha,alpha), and soft_targets
    are blended with that SAME lambda -- a latent-space regularizer, additive
    on top of the proven pixel-mixup (not a replacement), so its effect can
    be isolated cleanly."""
    model.train()
    total_loss, n = 0.0, 0
    for imgs, _, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        valid = labels >= 0
        if valid.sum() == 0 or valid.sum() < 2:
            continue
        imgs_v, labels_v = imgs[valid], labels[valid]
        if imgs_v.shape[0] % 2 == 1:
            imgs_v, labels_v = imgs_v[:-1], labels_v[:-1]
        if imgs_v.shape[0] < 2:
            continue
        imgs_mixed, soft_targets = mixup_fn(imgs_v, labels_v)

        optimizer.zero_grad()
        if manifold_mixup_alpha > 0:
            feat = model.backbone(imgs_mixed)
            lam2 = float(np.random.beta(manifold_mixup_alpha, manifold_mixup_alpha))
            perm = torch.randperm(feat.shape[0], device=feat.device)
            feat = lam2 * feat + (1 - lam2) * feat[perm]
            soft_targets = lam2 * soft_targets + (1 - lam2) * soft_targets[perm]
            logits = model.head(feat)
        else:
            logits = model(imgs_mixed)
        loss = criterion(logits, soft_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        total_loss += loss.item() * imgs_mixed.shape[0]
        n += imgs_mixed.shape[0]
    return total_loss / max(n, 1)


def run_fold(fold_i, train_df, val_df, test_df, backbone, seed, cfg,
             weight_decay=1e-4, drop_path_rate=None, head_dropout=0.3,
             label_smoothing=0.1, mixup_alpha=0.2, cutmix_alpha=1.0, manifold_mixup_alpha=0.0,
             preprocess=True):
    bf.set_deterministic(seed)
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"cls_{backbone}_mixup_fold{fold_i}_best.pth")
    laststate_path = os.path.join(CKPT_DIR, f"cls_{backbone}_mixup_fold{fold_i}_laststate.pth")
    done_marker = os.path.join(CKPT_DIR, f"cls_{backbone}_mixup_fold{fold_i}_DONE")

    train_aug, val_aug = bf.build_augmentations(cfg.img_size, seed=seed)
    train_ds = SkinLesionDatasetPrecomputedCrop(train_df, cfg.img_size, transform=train_aug,
                                                 oversample=True, target_per_class=cfg.oversample_target_per_class,
                                                 num_classes=cfg.num_classes, preprocess=preprocess)
    val_ds = SkinLesionDatasetPrecomputedCrop(val_df, cfg.img_size, transform=val_aug, preprocess=preprocess)
    test_ds = SkinLesionDatasetPrecomputedCrop(test_df, cfg.img_size, transform=val_aug, preprocess=preprocess)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"\n{'='*70}\n[FOLD {fold_i}] MIXUP backbone={backbone} seed={seed} "
          f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}\n{'='*70}")

    class_weights = bf.compute_inverse_freq_weights(train_df["label"].values, num_classes=cfg.num_classes).to(DEVICE)
    print(f"Inverse-freq class weights: {class_weights.cpu().numpy().round(3)}")

    mixup_fn = Mixup(mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, prob=1.0, switch_prob=0.5,
                      mode="batch", label_smoothing=label_smoothing, num_classes=cfg.num_classes)

    model = GenericClassifier(backbone, num_classes=cfg.num_classes,
                               drop_path_rate=drop_path_rate, head_dropout=head_dropout,
                               img_size=cfg.img_size).to(DEVICE)

    if os.path.exists(done_marker):
        print(f"[F{fold_i}] Sudah SELESAI sebelumnya (marker DONE ditemukan) -- langsung evaluasi TEST.")
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state"])
        test_acc, test_bal, test_f1 = evaluate_full(model, test_loader, DEVICE)
        print(f"[FOLD {fold_i}] TEST: acc={test_acc:.4f} bal_acc={test_bal:.4f} f1_macro={test_f1:.4f} "
              f"(checkpoint epoch={ck['epoch']})")
        return dict(fold=fold_i, test_acc=test_acc, test_bal=test_bal, test_f1=test_f1)

    criterion = SoftTargetCrossEntropyWeighted(class_weights=class_weights).to(DEVICE)
    is_transformer = "vit" in backbone or "swin" in backbone or "eva" in backbone
    base_lr = 5e-5 if is_transformer else 1e-4

    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": base_lr * BACKBONE_LR_SCALE},
        {"params": model.head.parameters(), "lr": base_lr},
    ], weight_decay=weight_decay)

    steps_per_epoch = len(train_loader)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / max(1, warmup_steps)))
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)

    max_epochs, patience = cfg.cls_max_epochs, cfg.cls_patience
    start_epoch, best_f1, no_improve, global_step = 1, 0.0, 0, 0
    if os.path.exists(laststate_path):
        state = torch.load(laststate_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        warmup_scheduler.load_state_dict(state["warmup_scheduler_state"])
        cosine_scheduler.load_state_dict(state["cosine_scheduler_state"])
        start_epoch = state["epoch"] + 1
        best_f1 = state["best_f1"]
        no_improve = state["no_improve"]
        global_step = state["global_step"]
        print(f"[F{fold_i}] RESUME dari epoch {start_epoch} (best_f1 sejauh ini={best_f1:.4f}, "
              f"no_improve={no_improve}/{patience})")

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()
        in_warmup = global_step < warmup_steps
        train_loss = train_one_epoch_mixup(model, train_loader, optimizer, criterion, mixup_fn, DEVICE,
                                            warmup_scheduler=warmup_scheduler if in_warmup else None,
                                            manifold_mixup_alpha=manifold_mixup_alpha)
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
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "warmup_scheduler_state": warmup_scheduler.state_dict(),
                    "cosine_scheduler_state": cosine_scheduler.state_dict(),
                    "best_f1": best_f1, "no_improve": no_improve, "global_step": global_step}, laststate_path)
        if no_improve >= patience:
            print(f"[F{fold_i}] Early stopping pada epoch {epoch}.")
            break
    else:
        print(f"[F{fold_i}] max_epochs={max_epochs} tercapai tanpa early-stop.")

    with open(done_marker, "w") as f:
        f.write("done\n")

    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state"])
    test_acc, test_bal, test_f1 = evaluate_full(model, test_loader, DEVICE)
    print(f"[FOLD {fold_i}] TEST: acc={test_acc:.4f} bal_acc={test_bal:.4f} f1_macro={test_f1:.4f} "
          f"(checkpoint epoch={ck['epoch']})")
    return dict(fold=fold_i, test_acc=test_acc, test_bal=test_bal, test_f1=test_f1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=[
        "vit_base_patch16_224", "efficientnet_b5", "resnet50", "convnext_tiny",
        "convnext_base", "swin_tiny_patch4_window7_224", "swin_base_patch4_window7_224",
        "eva02_base_patch16_clip_224.merged2b", "vit_base_patch14_reg4_dinov2",
        "vit_large_patch14_reg4_dinov2", "vit_base_patch16_siglip_224",
        "coatnet_2_rw_224.sw_in12k_ft_in1k", "maxvit_small_tf_224.in1k"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()

    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    ham_df = bf.build_ham10000_dataframe(paths)
    has_crop = ham_df["image_id"].apply(lambda iid: os.path.exists(os.path.join(CROP_DIR, f"{iid}.jpg")))
    ham_df = ham_df[has_crop].reset_index(drop=True)
    print(f"{len(ham_df)} gambar dgn crop tersedia")

    results = []
    for fold_i, train_idx, val_idx, test_idx in bf.stratified_ham_kfold(ham_df, cfg, n_splits=args.n_splits):
        if args.fold is not None and fold_i != args.fold:
            continue
        train_df = ham_df.iloc[train_idx].reset_index(drop=True)
        val_df = ham_df.iloc[val_idx].reset_index(drop=True)
        test_df = ham_df.iloc[test_idx].reset_index(drop=True)
        result = run_fold(fold_i, train_df, val_df, test_df, args.backbone, args.seed, cfg)
        results.append(result)
        print(f"RESULT_SUMMARY backbone={args.backbone} variant=mixup fold={fold_i} "
              f"test_acc={result['test_acc']:.4f} test_bal={result['test_bal']:.4f} test_f1={result['test_f1']:.4f}")

    if len(results) > 1:
        import numpy as np
        f1s = np.array([r["test_f1"] for r in results])
        print(f"\n=== RINGKASAN {len(results)}-FOLD MIXUP (backbone={args.backbone}) ===")
        print(f"f1_macro: mean={f1s.mean():.4f} std={f1s.std():.4f}")
        print(f"RESULT_SUMMARY_FINAL backbone={args.backbone} variant=mixup n_folds={len(results)} "
              f"mean_f1={f1s.mean():.4f} std_f1={f1s.std():.4f}")


if __name__ == "__main__":
    main()
