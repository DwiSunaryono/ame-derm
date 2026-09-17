"""Training classifier generik untuk backbone apa pun (via timm), dengan
protokol IDENTIK dengan yang menghasilkan cls_vit_best.pth/cls_b5_best.pth:
    - Data   : HAM10000, stratified split 70/15/15 seed=42 (sama persis
               dengan birnet_final.py, ham_test_df tidak pernah disentuh)
    - Oversampling: target 1500/kelas di training partition
    - Loss   : Label Smoothing Focal Loss (gamma=2.0, smoothing=0.1)
    - Optim  : AdamW + CosineAnnealingWarmRestarts (T0=20)
    - Early stopping patience=12 berdasarkan val balanced accuracy

Dipakai untuk menjawab pertanyaan "kenapa cuma ViT-B16 & EfficientNet-B5"
secara berbasis-data: latih beberapa kandidat backbone lain dengan cara
yang sama-sama adil, baru pilih pasangan/ensemble terbaik berdasarkan
bukti, bukan berdasarkan checkpoint yang kebetulan ada.

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

import birnet_final as bf

CKPT_DIR = "BIRNet_Research/outputs/checkpoints/candidate_backbones"
# Semua skrip downstream (fusion_v2/v3_clinical/positive_control, ensemble_*,
# joint_ncl_finetune, verify_*) memuat checkpoint dari sini dengan nama ini.
CANONICAL_CKPT_DIR = "BIRNet_Research/outputs/checkpoints/verify_ensemble"
CANONICAL_NAME = {"vit_base_patch16_224": "cls_vit_best.pth", "efficientnet_b5": "cls_b5_best.pth"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4
BATCH_SIZE = 32
# MAX_EPOCHS & PATIENCE diambil langsung dari cfg.cls_max_epochs/cls_patience
# di dalam main() -- SAMA PERSIS dengan protokol C5/C6/C7 (80 epoch, patience 12),
# supaya perbandingan antar backbone benar-benar apple-to-apple.


class GenericClassifier(nn.Module):
    """Struktur head identik dengan SkinClassifierB5/ViT di notebook asli,
    supaya kapasitas head tidak jadi variabel perancu antar backbone."""

    def __init__(self, backbone_name: str, num_classes: int = 7,
                 drop_path_rate: float = None, head_dropout: float = 0.3, img_size: int = 224):
        super().__init__()
        is_transformer = ("vit" in backbone_name or "swin" in backbone_name or "deit" in backbone_name
                           or "eva" in backbone_name)
        backbone_kwargs = {}
        if drop_path_rate is not None:
            backbone_kwargs["drop_path_rate"] = drop_path_rate
        # DINOv2 dilatih di 518px dgn pos-embed tetap -- utk pakai di resolusi
        # lain (224 default, atau 384/448 utk buka desain native-res-nya) perlu
        # interpolasi pos-embed via dynamic_img_size + img_size sesuai crop.
        if "dinov2" in backbone_name:
            backbone_kwargs["img_size"] = img_size
            backbone_kwargs["dynamic_img_size"] = True
        # BUG (ditemukan saat retraining pada split terkoreksi): baris ini
        # sebelumnya menulis global_pool="avg" di KEDUA cabang ternary,
        # sehingga ViT/Swin/DeiT dipaksa memakai avg-pooling (timm menamai
        # layer norm akhirnya "fc_norm") padahal SkinClassifierViT (dipakai
        # verify_standalone_checkpoints.py, birnet_fusion_v2.py, dst.) tidak
        # mengoper global_pool sama sekali untuk ViT, sehingga timm memakai
        # CLS-token pooling standar (layer norm akhirnya bernama "norm").
        # Checkpoint yang dihasilkan sebelum fix ini TIDAK bisa dimuat ke
        # SkinClassifierViT (state_dict key mismatch: fc_norm vs norm) --
        # transformer backbone karenanya tidak diberi argumen global_pool
        # sama sekali di sini, supaya arsitekturnya identik.
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
    """Like evaluate(), but also reports macro-F1. balanced_accuracy_score
    is recall-only (average per-class recall) -- it does NOT penalize a
    model that over-predicts one class at the expense of others' precision
    (e.g. the pooled-BCN oversampling-imbalance bug: mel's own recall rose
    while it was absorbing misclassified bcc/akiec/bkl samples, silently
    degrading mel's precision in a way balanced accuracy alone understates).
    Macro-F1 (harmonic mean of precision+recall, averaged per class) is
    sensitive to that precision damage and should be reported alongside
    balanced accuracy, not as a silent replacement, so historical
    balanced-accuracy comparisons in this project remain interpretable."""
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

    # PENTING: TIDAK pakai cls_weight inverse-frequency di sini. Training set
    # sudah di-oversample rata 1500/kelas -- menambah bobot inverse-frequency
    # (dihitung dari distribusi SEBELUM oversampling) di atas data yang sudah
    # seimbang menyebabkan double-correction (over-koreksi ke kelas minoritas,
    # terbukti dari distribusi prediksi model bias ke MEL/BKL, kurang ke NV).
    # Notebook asli (cls_vit_best/cls_b5_best, cell 64) memang memakai bobot
    # SERAGAM (eq_w = torch.ones(NUM_CLASSES)) -- disamakan di sini.
    criterion = bf.LabelSmoothingFocalLoss(gamma=2.0, smoothing=0.1, weight=None).to(DEVICE)
    # Backbone berbasis transformer/attention (ViT, Swin) butuh LR lebih kecil
    # dari CNN -- sesuai preseden notebook asli: cls_vit pakai lr=5e-5,
    # cls_b5 (CNN) pakai lr=1e-4. Disamakan di sini per tipe backbone.
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
