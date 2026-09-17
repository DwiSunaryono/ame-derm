"""Generic classifier training for any backbone (through timm), under a
protocol IDENTICAL to the one that produced cls_vit_best.pth/cls_b5_best.pth:
    - Data   : HAM10000, stratified 70/15/15 split seed=42 (exactly as in
               birnet_final.py, ham_test_df is never touched)
    - Oversampling: target 1500/class in the training partition
    - Loss   : Label Smoothing Focal Loss (gamma=2.0, smoothing=0.1)
    - Optim  : AdamW + CosineAnnealingWarmRestarts (T0=20)
    - Early stopping patience=12 based on val balanced accuracy

Used to answer "why only ViT-B16 & EfficientNet-B5" in a data-driven way:
train a few other backbone candidates under equally fair conditions, then
pick the best pair/ensemble from evidence, not from whichever checkpoints
happened to exist.

Pemakaian:
    python train_classifier_generic.py --backbone resnet50
    python train_classifier_generic.py --backbone convnext_tiny
    python train_classifier_generic.py --backbone swin_tiny_patch4_window7_224
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import balanced_accuracy_score, f1_score

from ame_derm.common import birnet_final as bf

CKPT_DIR = "BIRNet_Research/outputs/checkpoints/candidate_backbones"
# All downstream scripts (fusion_v2/v3_clinical/positive_control, ensemble_*,
# joint_ncl_finetune, verify_*) load their checkpoints from here under this name.
CANONICAL_CKPT_DIR = "BIRNet_Research/outputs/checkpoints/verify_ensemble"
CANONICAL_NAME = {"vit_base_patch16_224": "cls_vit_best.pth", "efficientnet_b5": "cls_b5_best.pth"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4
BATCH_SIZE = 32
# MAX_EPOCHS & PATIENCE are taken straight from cfg.cls_max_epochs/cls_patience
# inside main() -- EXACTLY as in the C5/C6/C7 protocol (80 epochs, patience 12),
# so backbone-to-backbone comparison is truly apple-to-apple.


class GenericClassifier(nn.Module):
    """Head structure identical to SkinClassifierB5/ViT in the original
    notebook, so head capacity is not a confounder across backbones."""

    def __init__(self, backbone_name: str, num_classes: int = 7,
                 drop_path_rate: float = None, head_dropout: float = 0.3, img_size: int = 224):
        super().__init__()
        is_transformer = ("vit" in backbone_name or "swin" in backbone_name or "deit" in backbone_name
                           or "eva" in backbone_name)
        backbone_kwargs = {}
        if drop_path_rate is not None:
            backbone_kwargs["drop_path_rate"] = drop_path_rate
        # DINOv2 was trained at 518px with fixed pos-embeds -- to use another
        # resolution (224 default, or 384/448 to exploit its native-res design)
        # the pos-embeds must be interpolated via dynamic_img_size + img_size per crop.
        if "dinov2" in backbone_name:
            backbone_kwargs["img_size"] = img_size
            backbone_kwargs["dynamic_img_size"] = True
        # BUG (spotted while retraining on the corrected split): this line
        # used to put global_pool="avg" in BOTH ternary branches, forcing
        # ViT/Swin/DeiT onto avg-pooling (timm then names the final layer norm
        # "fc_norm"), whereas SkinClassifierViT (used by
        # verify_standalone_checkpoints.py, birnet_fusion_v2.py, etc.) passes
        # no global_pool at all for ViT, so timm applies the standard
        # CLS-token pooling (final layer norm named "norm"). Checkpoints made
        # before this fix CANNOT be loaded into SkinClassifierViT (state_dict
        # key mismatch: fc_norm vs norm) -- transformer backbones therefore
        # receive no global_pool argument here at all, keeping the architecture identical.
        if is_transformer:
            self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, **backbone_kwargs)
        else:
            self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0,
                                               global_pool="avg", **backbone_kwargs)
        feat_dim = self.backbone.num_features
        if is_transformer:
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 512),
                nn.GELU(), nn.Dropout(head_dropout), nn.Linear(512, num_classes))
        else:
            self.head = nn.Sequential(
                nn.Linear(feat_dim, 512), nn.LayerNorm(512),
                nn.GELU(), nn.Dropout(head_dropout), nn.Linear(512, num_classes))

    def forward(self, x):
        return self.head(self.backbone(x))


def train_one_epoch(model, loader, optimizer, criterion, device):
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
        total_loss += loss.item() * valid.sum().item()
        n += valid.sum().item()
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labs = [], []
    for imgs, _, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        valid = labels >= 0
        preds.extend(logits[valid].argmax(1).cpu().numpy())
        labs.extend(labels[valid].numpy())
    preds, labs = np.array(preds), np.array(labs)
    acc = (preds == labs).mean()
    bal = balanced_accuracy_score(labs, preds)
    return acc, bal


def evaluate_full(model, loader, device):
    """Like evaluate(), but macro-F1 is reported as well. balanced_accuracy_score
    covers recall only (mean per-class recall) -- it does NOT punish a model
    that over-predicts one class while hurting others' precision (e.g. the
    pooled-BCN oversampling-imbalance bug: mel's own recall climbed as it
    swallowed misclassified bcc/akiec/bkl samples, quietly eroding mel's
    precision in a way balanced accuracy by itself understates). Macro-F1
    (the harmonic mean of precision+recall, averaged per class) reacts to
    that precision loss and ought to be reported next to balanced accuracy,
    not swapped in silently, so historical balanced-accuracy comparisons in
    this project stay interpretable."""
    model.eval()
    preds, labs = [], []
    for imgs, _, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        valid = labels >= 0
        preds.extend(logits[valid].argmax(1).cpu().numpy())
        labs.extend(labels[valid].numpy())
    preds, labs = np.array(preds), np.array(labs)
    acc = (preds == labs).mean()
    bal = balanced_accuracy_score(labs, preds)
    f1_macro = f1_score(labs, preds, average="macro", zero_division=0)
    return acc, bal, f1_macro


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, help="Nama model timm, misal resnet50, convnext_tiny")
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"cls_{args.backbone}_best.pth")
    canonical_path = None
    if args.backbone in CANONICAL_NAME:
        os.makedirs(CANONICAL_CKPT_DIR, exist_ok=True)
        canonical_path = os.path.join(CANONICAL_CKPT_DIR, CANONICAL_NAME[args.backbone])

    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    ham_df = bf.build_ham10000_dataframe(paths)
    train_idx, val_idx, test_idx = bf.stratified_ham_split(ham_df, cfg)
    ham_train_df = ham_df.iloc[train_idx].reset_index(drop=True)
    ham_val_df = ham_df.iloc[val_idx].reset_index(drop=True)

    train_aug, val_aug = bf.build_augmentations(cfg.img_size)
    train_ds = bf.SkinLesionDataset(ham_train_df, cfg.img_size, transform=train_aug,
                                     oversample=True, target_per_class=cfg.oversample_target_per_class,
                                     num_classes=cfg.num_classes)
    val_ds = bf.SkinLesionDataset(ham_val_df, cfg.img_size, transform=val_aug)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Backbone: {args.backbone} | train={len(train_ds)} val={len(val_ds)}")

    model = GenericClassifier(args.backbone, num_classes=cfg.num_classes).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameter: {n_params/1e6:.2f}M")

    # IMPORTANT: cls_weight inverse-frequency is NOT used here. The training
    # set is already oversampled to a flat 1500/class -- adding inverse-
    # frequency weights (computed from the distribution BEFORE oversampling)
    # on top of already-balanced data causes double-correction (over-correcting
    # toward minority classes, evident from the model's prediction distribution
    # skewing to MEL/BKL, short on NV). The original notebook
    # (cls_vit_best/cls_b5_best, cell 64) does use UNIFORM weights
    # (eq_w = torch.ones(NUM_CLASSES)) -- mirrored here.
    criterion = bf.LabelSmoothingFocalLoss(gamma=2.0, smoothing=0.1, weight=None).to(DEVICE)
    # Transformer/attention backbones (ViT, Swin) need a smaller LR than CNNs
    # -- matching the original notebook precedent: cls_vit used lr=5e-5,
    # cls_b5 (CNN) used lr=1e-4. Mirrored here per backbone type.
    is_transformer = ("vit" in args.backbone or "swin" in args.backbone or "deit" in args.backbone)
    lr = 5e-5 if is_transformer else LR
    print(f"Learning rate dipakai: {lr:.0e} ({'transformer' if is_transformer else 'CNN'})")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)

    max_epochs, patience = cfg.cls_max_epochs, cfg.cls_patience
    best_bal, no_improve = 0.0, 0
    print(f"\nMulai training {args.backbone} | max_epochs={max_epochs} patience={patience}")
    print("=" * 80)
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_acc, val_bal = evaluate(model, val_loader, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        is_best = val_bal > best_bal
        flag = ""
        if is_best:
            best_bal, no_improve, flag = val_bal, 0, "*"
            ckpt = {"epoch": epoch, "model_state": model.state_dict(), "val_bal": val_bal}
            torch.save(ckpt, ckpt_path)
            if canonical_path is not None:
                torch.save(ckpt, canonical_path)
        else:
            no_improve += 1

        print(f"Ep {epoch:3d}/{max_epochs} | train_loss={train_loss:.4f} "
              f"| val_acc={val_acc:.4f} val_bal={val_bal:.4f} (best={best_bal:.4f}) "
              f"| {elapsed:.0f}s {flag}")

        if no_improve >= patience:
            print(f"Early stopping pada epoch {epoch}.")
            break

    print(f"\nSelesai. Checkpoint terbaik: {ckpt_path} (val_bal={best_bal:.4f})")
    if canonical_path is not None:
        print(f"Juga disalin ke lokasi kanonik: {canonical_path}")


if __name__ == "__main__":
    main()
