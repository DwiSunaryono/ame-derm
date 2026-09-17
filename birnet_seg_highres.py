"""Item 4 of the "maximize Phase 1" list (most expensive): retrains
UnetPlusPlus at 384x384 instead of 224x224 -- more input detail should, in
principle, let the model capture finer lesion boundary precision.

Fairness note: since the downstream crop-then-classify pipeline needs
224x224-compatible masks regardless, this script evaluates Dice TWICE:
  (a) at native 384x384 (informative on its own, but not directly
      comparable to the 224x224 foundation's 0.8773 number since resizing
      changes pixel counts)
  (b) predictions downsampled to 224x224 and compared against the SAME
      224x224 ground truth used everywhere else in this project -- this is
      the fair, apples-to-apples number against the existing foundation.

Same encoder (efficientnet-b4), same paper-matched loss, same pooled
dataset, fully deterministic. Batch size reduced (8 instead of 16) since
384x384 images use ~3x the memory of 224x224 per sample.

Usage:
    python birnet_seg_highres.py --seed 42
"""
import argparse
import time
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

import birnet_final as bf
from birnet_seg_bgff_v2 import PaperMatchedSegLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_SIZE = 384
EVAL_SIZE = 224  # for fair comparison against the existing foundation
BATCH_SIZE = 8
MAX_EPOCHS = 60
PATIENCE = 10
ENCODER = "efficientnet-b4"


class SegDatasetHighRes(Dataset):
    """Like SegDatasetV2, but ALSO returns the mask resized to EVAL_SIZE
    (224) alongside the TRAIN_SIZE (384) version, so a single dataset can
    serve both the training loop and the fair-comparison evaluation."""

    def __init__(self, df, train_size, eval_size):
        self.entries = list(zip(df["img_path"], df["mask_path"]))
        self.train_size = train_size
        self.eval_size = eval_size

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, mask_path = self.entries[idx]
        img = np.array(bf.Image.open(img_path).convert("RGB")).astype(np.uint8)
        img_native_h, img_native_w = img.shape[:2]
        img_train = cv2.resize(img, (self.train_size, self.train_size), interpolation=cv2.INTER_LINEAR)
        mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
        img_train = (img_train.astype(np.float32) / 255.0 - mean) / std

        mask = np.array(bf.Image.open(mask_path).convert("L"))
        mask_train = cv2.resize(mask, (self.train_size, self.train_size), interpolation=cv2.INTER_NEAREST)
        mask_train = (mask_train > 127).astype(np.float32)
        mask_eval = cv2.resize(mask, (self.eval_size, self.eval_size), interpolation=cv2.INTER_NEAREST)
        mask_eval = (mask_eval > 127).astype(np.float32)

        img_t = torch.from_numpy(img_train.transpose(2, 0, 1)).float()
        mask_train_t = torch.from_numpy(mask_train).unsqueeze(0).float()
        mask_eval_t = torch.from_numpy(mask_eval).unsqueeze(0).float()
        return img_t, mask_train_t, mask_eval_t


def dice_iou_batch(pred, gt):
    B = pred.shape[0]
    p, t = pred.view(B, -1), gt.view(B, -1)
    inter = (p * t).sum(1)
    union = p.sum(1) + t.sum(1) - inter
    dice = ((2 * inter + 1.0) / (p.sum(1) + t.sum(1) + 1.0)).mean().item()
    iou = ((inter + 1.0) / (union + 1.0)).mean().item()
    return dice, iou


@torch.no_grad()
def evaluate(model, loader, device, thr=0.5):
    model.eval()
    total_dice_native = total_iou_native = 0.0
    total_dice_224 = total_iou_224 = 0.0
    n = 0
    for imgs, masks_train, masks_eval in loader:
        imgs, masks_train, masks_eval = imgs.to(device), masks_train.to(device), masks_eval.to(device)
        logits = model(imgs)
        prob = torch.sigmoid(logits)
        pred_native = (prob > thr).float()
        d_native, i_native = dice_iou_batch(pred_native, masks_train)

        prob_224 = torch.nn.functional.interpolate(prob, size=(EVAL_SIZE, EVAL_SIZE), mode="bilinear", align_corners=False)
        pred_224 = (prob_224 > thr).float()
        d_224, i_224 = dice_iou_batch(pred_224, masks_eval)

        total_dice_native += d_native; total_iou_native += i_native
        total_dice_224 += d_224; total_iou_224 += i_224
        n += 1
    return (total_dice_native / max(n, 1), total_iou_native / max(n, 1),
            total_dice_224 / max(n, 1), total_iou_224 / max(n, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder", type=str, default=ENCODER,
                         help="e.g. efficientnet-b4, efficientnet-b5, efficientnet-b6, resnet50")
    args = parser.parse_args()
    bf.set_deterministic(args.seed)
    encoder = args.encoder
    encoder_tag = encoder.replace("efficientnet-", "")

    import os
    ckpt_dir = "BIRNet_Research/outputs/checkpoints/foundation"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = f"{ckpt_dir}/unetpp_{encoder_tag}_highres384_seed{args.seed}_best.pth"

    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    isic_t1_train_df = bf.build_seg_dataframe(paths.isic_t1_train_img, paths.isic_t1_train_mask)
    isic16_df = bf.build_seg_dataframe(paths.isic16_img, paths.isic16_mask, mask_suffix="_Segmentation.png")
    isic17_df = bf.build_seg_dataframe(paths.isic17_img, paths.isic17_mask, mask_suffix="_segmentation.png")
    ph2_df = bf.build_ph2_dataframe(paths)
    train_df = bf.pd.concat([isic_t1_train_df, isic16_df, isic17_df, ph2_df], ignore_index=True)
    isic_t1_val_df = bf.build_seg_dataframe(paths.isic_t1_val_img, paths.isic_t1_val_mask)
    isic_t1_test_df = bf.build_seg_dataframe(paths.isic_t1_test_img, paths.isic_t1_test_mask)

    print(f"[HIGHRES384 seed={args.seed}] train(pooled)={len(train_df)} "
          f"val(ISIC18T1)={len(isic_t1_val_df)} test(ISIC18T1)={len(isic_t1_test_df)}")

    train_ds = SegDatasetHighRes(train_df, TRAIN_SIZE, EVAL_SIZE)
    val_ds = SegDatasetHighRes(isic_t1_val_df, TRAIN_SIZE, EVAL_SIZE)
    test_ds = SegDatasetHighRes(isic_t1_test_df, TRAIN_SIZE, EVAL_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = smp.UnetPlusPlus(encoder_name=encoder, encoder_weights="imagenet",
                              in_channels=3, classes=1, activation=None).to(DEVICE)
    seg_loss_fn = PaperMatchedSegLoss().to(DEVICE)

    optimizer = optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 1e-5},
        {"params": model.decoder.parameters(), "lr": 1e-4},
        {"params": model.segmentation_head.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)

    best_dice_224, no_improve = 0.0, 0
    print(f"Mulai training HIGHRES384 (encoder={encoder}) | max_epochs={MAX_EPOCHS} patience={PATIENCE}")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        t0 = time.time()
        total_loss, n = 0.0, 0
        for imgs, masks_train, _ in train_loader:
            imgs, masks_train = imgs.to(DEVICE), masks_train.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = seg_loss_fn(logits, masks_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); n += 1
        scheduler.step()

        val_dice_native, val_iou_native, val_dice_224, val_iou_224 = evaluate(model, val_loader, DEVICE)
        is_best = val_dice_224 > best_dice_224
        flag = ""
        if is_best:
            best_dice_224, no_improve, flag = val_dice_224, 0, "*"
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_dice_224": val_dice_224, "val_dice_native": val_dice_native}, ckpt_path)
        else:
            no_improve += 1
        print(f"Ep {epoch:3d}/{MAX_EPOCHS} | loss={total_loss/max(n,1):.4f} "
              f"| val_dice_native384={val_dice_native:.4f} val_dice@224={val_dice_224:.4f} "
              f"(best@224={best_dice_224:.4f}) | {time.time()-t0:.0f}s {flag}")
        if no_improve >= PATIENCE:
            print(f"Early stopping pada epoch {epoch}.")
            break

    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state"])
    test_dice_native, test_iou_native, test_dice_224, test_iou_224 = evaluate(model, test_loader, DEVICE)
    print(f"\n=== TEST (ISIC 2018 Task 1, HIGHRES384, encoder={encoder}, seed={args.seed}) ===")
    print(f"Native 384x384   : dice={test_dice_native:.4f} iou={test_iou_native:.4f}")
    print(f"Downsampled ke 224x224 (pembanding adil vs versi 224-native): dice={test_dice_224:.4f} iou={test_iou_224:.4f}")
    print(f"RESULT_SUMMARY model=seg_highres384 encoder={encoder_tag} seed={args.seed} "
          f"test_dice_native={test_dice_native:.4f} test_dice_224={test_dice_224:.4f}")


if __name__ == "__main__":
    main()
