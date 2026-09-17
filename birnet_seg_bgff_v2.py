"""birnet_seg_bgff.py with two changes to align more closely with the
reference hair-segmentation paper's setup, per explicit request to match
its loss function before testing grayscale vs RGB on lesion segmentation:

  1. Loss function replaced with the paper's own formulation:
       L1 = lambda1*BCE + lambda2*L_BD + lambda3*FocalTversky
       (lambda1=0.3, lambda2=0.1, lambda3=0.7; Tversky alpha=0.7, beta=0.3,
       focal gamma=0.75; L_BD = Dice computed on the Sobel boundary maps of
       prediction vs ground truth, not the full mask)
     used for the whole run here (the paper switches to a second weighting
     L2 in its stages 2-3; we keep one stage since our discriminative-LR
     recipe already serves the "stabilize then adapt" role that multi-stage
     schedule serves in the source paper).
  2. --color_space {rgb,grayscale}: grayscale mode converts to single-channel
     and replicates to 3 channels before the encoder (same approach the
     source paper uses), so the same EfficientNet-B4 encoder architecture
     applies unmodified.

Not matched (impractical for this pipeline, noted explicitly rather than
silently ignored): the paper's 512x512 input and 80+120+200-epoch three-
stage schedule. This pipeline's resolution is fixed at 224x224 by the
downstream classification models it must feed masks to, and single-stage
discriminative-LR training already fixed a real overfitting/instability
pattern observed earlier today -- reverting to the paper's LR schedule
would reintroduce it.

Usage:
    python birnet_seg_bgff_v2.py --color_space rgb
    python birnet_seg_bgff_v2.py --color_space grayscale
"""
import argparse
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import birnet_final as bf
from birnet_seg_bgff import UNetBGFF, SobelBoundary

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 16
MAX_EPOCHS = 60
PATIENCE = 10
LAMBDA1, LAMBDA2, LAMBDA3 = 0.3, 0.1, 0.7  # paper's L1 weights (BCE, boundary-Dice, Focal-Tversky)


class FocalTverskyLoss(nn.Module):
    """Paper's Tversky (alpha=0.7 penalizes FP more, beta=0.3) + focal
    exponent gamma=0.75, matching its stated configuration exactly."""

    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1.0):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        B = probs.shape[0]
        p, t = probs.view(B, -1), targets.view(B, -1)
        tp = (p * t).sum(1)
        fp = (p * (1 - t)).sum(1)
        fn = ((1 - p) * t).sum(1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return ((1.0 - tversky) ** self.gamma).mean()


class BoundaryDiceLoss(nn.Module):
    """L_BD: Dice loss computed on the Sobel boundary maps of prediction
    vs ground truth, rather than on the full masks."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.sobel = SobelBoundary()
        self.smooth = smooth

    def forward(self, logits, targets):
        pred_b = self.sobel(torch.sigmoid(logits))
        gt_b = self.sobel(targets)
        B = pred_b.shape[0]
        p, t = pred_b.view(B, -1), gt_b.view(B, -1)
        inter = (p * t).sum(1)
        dice = (2 * inter + self.smooth) / (p.sum(1) + t.sum(1) + self.smooth)
        return 1.0 - dice.mean()


class PaperMatchedSegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.l_bd = BoundaryDiceLoss()
        self.focal_tversky = FocalTverskyLoss()

    def forward(self, logits, targets):
        return (LAMBDA1 * self.bce(logits, targets)
                + LAMBDA2 * self.l_bd(logits, targets)
                + LAMBDA3 * self.focal_tversky(logits, targets))


def load_and_preprocess_seg_v2(img_path, mask_path, img_size, color_space):
    img = np.array(bf.Image.open(img_path).convert("RGB")).astype(np.uint8)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    if color_space == "grayscale":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.stack([gray, gray, gray], axis=-1)  # replicate to 3ch, same as source paper
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    img = (img.astype(np.float32) / 255.0 - mean) / std
    import os
    if mask_path and os.path.exists(mask_path):
        mask = np.array(bf.Image.open(mask_path).convert("L"))
        mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
    else:
        mask = np.zeros((img_size, img_size), dtype=np.float32)
    return img, mask


class SegDatasetV2(Dataset):
    def __init__(self, df, img_size, color_space):
        self.entries = list(zip(df["img_path"], df["mask_path"]))
        self.img_size = img_size
        self.color_space = color_space

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, mask_path = self.entries[idx]
        img, mask = load_and_preprocess_seg_v2(img_path, mask_path, self.img_size, self.color_space)
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return img_t, mask_t


@torch.no_grad()
def evaluate(model, loader, device, thr=0.5):
    model.eval()
    total_dice = total_iou = n = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits, _ = model(imgs)
        pred = (torch.sigmoid(logits) > thr).float()
        B = pred.shape[0]
        p, t = pred.view(B, -1), masks.view(B, -1)
        inter = (p * t).sum(1)
        union = p.sum(1) + t.sum(1) - inter
        dice = ((2 * inter + 1.0) / (p.sum(1) + t.sum(1) + 1.0)).mean().item()
        iou = ((inter + 1.0) / (union + 1.0)).mean().item()
        total_dice += dice; total_iou += iou; n += 1
    return total_dice / max(n, 1), total_iou / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--color_space", required=True, choices=["rgb", "grayscale"])
    args = parser.parse_args()
    ckpt_path = f"BIRNet_Research/outputs/checkpoints/verify_ensemble/unetpp_bgff_{args.color_space}_best.pth"

    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    isic_t1_train_df = bf.build_seg_dataframe(paths.isic_t1_train_img, paths.isic_t1_train_mask)
    isic16_df = bf.build_seg_dataframe(paths.isic16_img, paths.isic16_mask, mask_suffix="_Segmentation.png")
    isic17_df = bf.build_seg_dataframe(paths.isic17_img, paths.isic17_mask, mask_suffix="_segmentation.png")
    ph2_df = bf.build_ph2_dataframe(paths)
    train_df = bf.pd.concat([isic_t1_train_df, isic16_df, isic17_df, ph2_df], ignore_index=True)
    isic_t1_val_df = bf.build_seg_dataframe(paths.isic_t1_val_img, paths.isic_t1_val_mask)
    isic_t1_test_df = bf.build_seg_dataframe(paths.isic_t1_test_img, paths.isic_t1_test_mask)

    print(f"color_space={args.color_space} train(pooled)={len(train_df)} "
          f"val(ISIC18T1)={len(isic_t1_val_df)} test(ISIC18T1)={len(isic_t1_test_df)}")

    train_ds = SegDatasetV2(train_df, IMG_SIZE, args.color_space)
    val_ds = SegDatasetV2(isic_t1_val_df, IMG_SIZE, args.color_space)
    test_ds = SegDatasetV2(isic_t1_test_df, IMG_SIZE, args.color_space)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = UNetBGFF().to(DEVICE)
    seg_loss_fn = PaperMatchedSegLoss().to(DEVICE)
    sobel = SobelBoundary().to(DEVICE)

    optimizer = optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 1e-5},
        {"params": model.blocks.parameters(), "lr": 1e-4},
        {"params": model.head.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)

    best_dice, no_improve = 0.0, 0
    print(f"Mulai training UNet+BGFF v2 (paper-matched loss) | color_space={args.color_space} "
          f"max_epochs={MAX_EPOCHS} patience={PATIENCE}")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        t0 = time.time()
        total_loss, n = 0.0, 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            logits, boundary_logits = model(imgs)
            loss = seg_loss_fn(logits, masks)
            gt_boundary_full = sobel(masks)
            for b_logit in boundary_logits:
                gt_b = F.interpolate(gt_boundary_full, size=b_logit.shape[2:], mode="bilinear", align_corners=False)
                loss = loss + 0.3 * F.binary_cross_entropy_with_logits(b_logit, gt_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); n += 1
        scheduler.step()

        val_dice, val_iou = evaluate(model, val_loader, DEVICE)
        is_best = val_dice > best_dice
        flag = ""
        if is_best:
            best_dice, no_improve, flag = val_dice, 0, "*"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_dice": val_dice}, ckpt_path)
        else:
            no_improve += 1
        print(f"Ep {epoch:3d}/{MAX_EPOCHS} | loss={total_loss/max(n,1):.4f} "
              f"| val_dice={val_dice:.4f} val_iou={val_iou:.4f} (best={best_dice:.4f}) | {time.time()-t0:.0f}s {flag}")
        if no_improve >= PATIENCE:
            print(f"Early stopping pada epoch {epoch}.")
            break

    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state"])
    test_dice, test_iou = evaluate(model, test_loader, DEVICE)
    print(f"\n=== TEST (ISIC 2018 Task 1, UNet+BGFF v2, color_space={args.color_space}) ===")
    print(f"dice={test_dice:.4f} iou={test_iou:.4f} (checkpoint epoch={ck['epoch']})")
    print(f"RESULT_SUMMARY model=unetpp_bgff_v2 color_space={args.color_space} "
          f"test_dice={test_dice:.4f} test_iou={test_iou:.4f}")


if __name__ == "__main__":
    main()
