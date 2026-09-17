"""Tests another genuinely new architecture axis for the Phase 1 ensemble:
UPerNet decoder + ConvNeXt encoder (via timm-universal, "tu-convnext_tiny"),
prompted by an external claim that Mask2Former/UperNet+ConvNeXt/InternImage
beat SegFormer on general segmentation benchmarks. Unlike Mask2Former/SAM/
UNETR (which need entirely new libraries -- detectron2, segment_anything,
monai -- none installed here), UPerNet+ConvNeXt works directly through
segmentation_models_pytorch's existing timm-universal encoder support, so
it's the same tier of effort as the mit_b3/Segformer test.

Same paper-matched loss, same pooled dataset, same deterministic setup as
every other Phase 1 script.

Usage:
    python birnet_seg_convnext_upernet.py --seed 42
"""
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from ame_derm.common import birnet_final as bf
from ame_derm.segmentation.seg_bgff_v2 import PaperMatchedSegLoss, SegDatasetV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 16
MAX_EPOCHS = 60
PATIENCE = 10
ENCODER = "tu-convnext_tiny"
CKPT_DIR = "BIRNet_Research/outputs/checkpoints/foundation"


@torch.no_grad()
def evaluate(model, loader, device, thr=0.5):
    model.eval()
    total_dice = total_iou = n = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bf.set_deterministic(args.seed)

    import os
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = f"{CKPT_DIR}/upernet_convnext_foundation_seed{args.seed}_best.pth"

    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    isic_t1_train_df = bf.build_seg_dataframe(paths.isic_t1_train_img, paths.isic_t1_train_mask)
    isic16_df = bf.build_seg_dataframe(paths.isic16_img, paths.isic16_mask, mask_suffix="_Segmentation.png")
    isic17_df = bf.build_seg_dataframe(paths.isic17_img, paths.isic17_mask, mask_suffix="_segmentation.png")
    ph2_df = bf.build_ph2_dataframe(paths)
    train_df = bf.pd.concat([isic_t1_train_df, isic16_df, isic17_df, ph2_df], ignore_index=True)
    isic_t1_val_df = bf.build_seg_dataframe(paths.isic_t1_val_img, paths.isic_t1_val_mask)
    isic_t1_test_df = bf.build_seg_dataframe(paths.isic_t1_test_img, paths.isic_t1_test_mask)

    print(f"[UPERNET-CONVNEXT seed={args.seed}] train(pooled)={len(train_df)} "
          f"val(ISIC18T1)={len(isic_t1_val_df)} test(ISIC18T1)={len(isic_t1_test_df)}")

    train_ds = SegDatasetV2(train_df, IMG_SIZE, "rgb")
    val_ds = SegDatasetV2(isic_t1_val_df, IMG_SIZE, "rgb")
    test_ds = SegDatasetV2(isic_t1_test_df, IMG_SIZE, "rgb")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = smp.UPerNet(encoder_name=ENCODER, encoder_weights="imagenet",
                         in_channels=3, classes=1, activation=None).to(DEVICE)
    seg_loss_fn = PaperMatchedSegLoss().to(DEVICE)

    optimizer = optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 1e-5},
        {"params": model.decoder.parameters(), "lr": 1e-4},
        {"params": model.segmentation_head.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)

    best_dice, no_improve = 0.0, 0
    print(f"Mulai training UPERNET (encoder={ENCODER}, deterministic) "
          f"| max_epochs={MAX_EPOCHS} patience={PATIENCE}")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        t0 = time.time()
        total_loss, n = 0.0, 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = seg_loss_fn(logits, masks)
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
    print(f"\n=== TEST (ISIC 2018 Task 1, UPERNET encoder={ENCODER}, seed={args.seed}) ===")
    print(f"dice={test_dice:.4f} iou={test_iou:.4f} (checkpoint epoch={ck['epoch']})")
    print(f"RESULT_SUMMARY model=seg_upernet_convnext seed={args.seed} "
          f"test_dice={test_dice:.4f} test_iou={test_iou:.4f}")


if __name__ == "__main__":
    main()
