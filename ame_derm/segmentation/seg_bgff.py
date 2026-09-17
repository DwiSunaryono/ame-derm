"""Lesion segmentation using a Boundary-Guided Feature Fusion (BGFF) decoder,
derived from the boundary-guided skip-connection concept introduced for
hair segmentation in a companion paper (Mobile U-Net + BGFF: B = Dilate(y) -
Erode(y); F_BGFF = M_B(B_F) (x) F). In that paper the BGFF boundary map is
computed from the GROUND-TRUTH mask by means of morphological
dilation/erosion. That works for hair segmentation's own evaluation
setting, but cannot be applied directly at inference time on a NEW lesion
image, since no ground-truth mask is available there to dilate or erode.

Adaptation adopted here: rather than deriving a morphological boundary map
from GT, every decoder stage carries a small learned "boundary head" that
predicts a boundary-probability map straight from the fused encoder+decoder
features at that scale. An auxiliary loss supervises it against the Sobel
boundary of the true ground-truth mask (downsampled to that stage's
resolution, and applied only during training). During inference the
boundary head operates on the image's own predicted features -- ground
truth is not needed -- and the head's output gates (multiplicatively
emphasizes) the fused feature before it passes onward through the decoder,
mirroring the original BGFF idea while remaining usable on unseen images.

Training uses the exact same pooled segmentation data (ISIC 2018 Task 1 +
ISIC 2016 + ISIC 2017 + PH2) and evaluation follows the exact same protocol
(ISIC 2018 Task 1 test partition, Dice/IoU) as the existing UnetPlusPlus
(unetpp_final_best.pth), keeping the two directly comparable: does a
boundary-guided decoder yield a better lesion mask than the current plain
UNet++, and if so, does that convert into a larger crop-then-classify gain
downstream?

Usage:
    python birnet_seg_bgff.py
"""
import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

from ame_derm.common import birnet_final as bf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 16
MAX_EPOCHS = 60
PATIENCE = 10
LAMBDA_BOUNDARY = 0.3
CKPT_PATH = "BIRNet_Research/outputs/checkpoints/verify_ensemble/unetpp_bgff_best.pth"


class SobelBoundary(nn.Module):
    """A Sobel-based boundary extractor identical to the one in
    birnet_final.BoundaryLoss; it is reused to supervise the learned
    boundary heads."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, mask):
        gx = F.conv2d(mask, self.sobel_x, padding=1)
        gy = F.conv2d(mask, self.sobel_y, padding=1)
        return torch.clamp(torch.sqrt(gx ** 2 + gy ** 2 + 1e-8), 0, 1)


class BGFFDecoderBlock(nn.Module):
    """A single U-Net decoder stage equipped with a boundary-guided skip
    connection. The encoder skip feature (E) and the upsampled decoder
    feature (D) are combined; a learned boundary head then predicts a
    1-channel boundary-probability map from that combination, and this map
    gates the combined features (1 + M_B) ahead of the standard conv block
    -- the deployable counterpart to BGFF's morphological boundary
    gating."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.boundary_head = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 2, 1, 1))
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, d, skip):
        d = F.interpolate(d, size=skip.shape[2:], mode="bilinear", align_corners=False)
        fused = self.reduce(torch.cat([d, skip], dim=1))
        boundary_logit = self.boundary_head(fused)
        boundary_prob = torch.sigmoid(boundary_logit)
        gated = fused * (1.0 + boundary_prob)
        out = self.conv(gated)
        return out, boundary_logit


class UNetBGFF(nn.Module):
    """An EfficientNet-B4 encoder (ImageNet-pretrained, the same one used by
    the existing UnetPlusPlus baseline) paired with a plain U-Net-depth
    decoder whose skip connections are BGFF-gated; a boundary-prediction
    head sits at each stage for auxiliary supervision."""

    def __init__(self, encoder_name="efficientnet-b4"):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(encoder_name, weights="imagenet")
        ch = self.encoder.out_channels  # e.g. (3, 48, 32, 56, 160, 448) ordered shallow->deep, input included
        dec_ch = [256, 128, 64, 32, 16]
        self.blocks = nn.ModuleList()
        in_ch = ch[-1]
        skip_chs = list(reversed(ch[1:-1]))  # skip connections, ordered deep->shallow, leaving out the deepest (bottleneck) and the raw input
        for i, out_ch in enumerate(dec_ch):
            skip_ch = skip_chs[i] if i < len(skip_chs) else 0
            self.blocks.append(BGFFDecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch
        self.head = nn.Conv2d(dec_ch[-1], 1, 1)

    def forward(self, x):
        feats = self.encoder(x)  # ordered shallow->deep, where feats[0] is the input itself
        d = feats[-1]
        skips = list(reversed(feats[1:-1]))
        boundary_logits = []
        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else torch.zeros(d.shape[0], 0, *d.shape[2:], device=d.device)
            d, b_logit = block(d, skip)
            boundary_logits.append(b_logit)
        out = F.interpolate(self.head(d), size=x.shape[2:], mode="bilinear", align_corners=False)
        return out, boundary_logits


def load_and_preprocess_seg(img_path, mask_path, img_size):
    img = np.array(bf.Image.open(img_path).convert("RGB")).astype(np.uint8)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    img = (img.astype(np.float32) / 255.0 - mean) / std
    if mask_path and os.path.exists(mask_path):
        mask = np.array(bf.Image.open(mask_path).convert("L"))
        mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
    else:
        mask = np.zeros((img_size, img_size), dtype=np.float32)
    return img, mask


class SegDataset(Dataset):
    def __init__(self, df, img_size):
        self.entries = list(zip(df["img_path"], df["mask_path"]))
        self.img_size = img_size

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, mask_path = self.entries[idx]
        img, mask = load_and_preprocess_seg(img_path, mask_path, self.img_size)
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
    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)

    isic_t1_train_df = bf.build_seg_dataframe(paths.isic_t1_train_img, paths.isic_t1_train_mask)
    isic16_df = bf.build_seg_dataframe(paths.isic16_img, paths.isic16_mask, mask_suffix="_Segmentation.png")
    isic17_df = bf.build_seg_dataframe(paths.isic17_img, paths.isic17_mask, mask_suffix="_segmentation.png")
    ph2_df = bf.build_ph2_dataframe(paths)
    train_df = bf.pd.concat([isic_t1_train_df, isic16_df, isic17_df, ph2_df], ignore_index=True)

    isic_t1_val_df = bf.build_seg_dataframe(paths.isic_t1_val_img, paths.isic_t1_val_mask)
    isic_t1_test_df = bf.build_seg_dataframe(paths.isic_t1_test_img, paths.isic_t1_test_mask)

    print(f"train(pooled)={len(train_df)} val(ISIC18T1)={len(isic_t1_val_df)} test(ISIC18T1)={len(isic_t1_test_df)}")

    train_ds = SegDataset(train_df, IMG_SIZE)
    val_ds = SegDataset(isic_t1_val_df, IMG_SIZE)
    test_ds = SegDataset(isic_t1_test_df, IMG_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = UNetBGFF().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet+BGFF total params: {n_params/1e6:.2f}M")

    seg_loss_fn = bf.SegmentationLoss()
    sobel = SobelBoundary().to(DEVICE)

    optimizer = optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 1e-5},
        {"params": model.blocks.parameters(), "lr": 1e-4},
        {"params": model.head.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)

    best_dice, no_improve = 0.0, 0
    print(f"Mulai training UNet+BGFF | max_epochs={MAX_EPOCHS} patience={PATIENCE}")
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
                loss = loss + LAMBDA_BOUNDARY * F.binary_cross_entropy_with_logits(b_logit, gt_b)
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
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_dice": val_dice}, CKPT_PATH)
        else:
            no_improve += 1
        print(f"Ep {epoch:3d}/{MAX_EPOCHS} | loss={total_loss/max(n,1):.4f} "
              f"| val_dice={val_dice:.4f} val_iou={val_iou:.4f} (best={best_dice:.4f}) | {time.time()-t0:.0f}s {flag}")
        if no_improve >= PATIENCE:
            print(f"Early stopping pada epoch {epoch}.")
            break

    ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state"])
    test_dice, test_iou = evaluate(model, test_loader, DEVICE)
    print(f"\n=== TEST (ISIC 2018 Task 1, UNet+BGFF) ===")
    print(f"dice={test_dice:.4f} iou={test_iou:.4f} (checkpoint epoch={ck['epoch']})")
    print(f"RESULT_SUMMARY model=unetpp_bgff test_dice={test_dice:.4f} test_iou={test_iou:.4f}")


if __name__ == "__main__":
    main()
