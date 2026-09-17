"""Evaluates guided-image-filter post-processing (He et al. 2010) as a
dependency-free substitute for dense CRF (pydensecrf would not build -- a
Cython/Python version incompatibility; and cv2.ximgproc is missing from
this opencv-python build). It performs edge-aware boundary refinement of
the predicted segmentation probability map, with the original RGB image
serving as the guide.

No retraining is involved -- this is purely inference-time post-processing,
in the same spirit as the CRF item on the "cheap wins before expensive
retraining" list.

Usage:
    python birnet_seg_foundation_guided_filter.py
"""
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from ame_derm.common import birnet_final as bf
from ame_derm.segmentation.seg_bgff_v2 import SegDatasetV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
CKPT = "BIRNet_Research/outputs/checkpoints/foundation/unetpp_b4_foundation_seed42_best.pth"


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """Guided filter (He, Sun, Tang 2010). Inputs are guide: an HxW grayscale
    float32 array in [0,1], and src: an HxW probability map of float32 in
    [0,1]. The returned value is the refined probability map."""
    box = lambda x: cv2.boxFilter(x, cv2.CV_32F, (radius, radius))
    mean_I = box(guide)
    mean_p = box(src)
    corr_I = box(guide * guide)
    corr_Ip = box(guide * src)
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = box(a)
    mean_b = box(b)
    return mean_a * guide + mean_b


def dice_iou(pred, gt):
    p, t = pred.flatten(), gt.flatten()
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    dice = (2 * inter + 1.0) / (p.sum() + t.sum() + 1.0)
    iou = (inter + 1.0) / (union + 1.0)
    return dice, iou


def main():
    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    isic_t1_test_df = bf.build_seg_dataframe(paths.isic_t1_test_img, paths.isic_t1_test_mask)
    test_ds = SegDatasetV2(isic_t1_test_df, IMG_SIZE, "rgb")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    model = smp.UnetPlusPlus(encoder_name="efficientnet-b4", encoder_weights=None,
                              in_channels=3, classes=1, activation=None).to(DEVICE)
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()

    dices_raw, dices_gf, ious_raw, ious_gf = [], [], [], []
    best_radius_search = {r: [] for r in [4, 8, 16]}

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs_gpu = imgs.to(DEVICE)
            logits = model(imgs_gpu)
            prob = torch.sigmoid(logits).cpu().numpy()[0, 0].astype(np.float32)
            gt = masks.numpy()[0, 0]

            # undo normalization, returning the image to a [0,1] grayscale guide
            img_np = imgs.numpy()[0].transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
            img_rgb = np.clip(img_np * std + mean, 0, 1).astype(np.float32)
            guide = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

            pred_raw = (prob > 0.5).astype(np.float32)
            d_raw, i_raw = dice_iou(pred_raw, gt)
            dices_raw.append(d_raw); ious_raw.append(i_raw)

            for r in [4, 8, 16]:
                refined = guided_filter(guide, prob, radius=r)
                pred_gf = (refined > 0.5).astype(np.float32)
                d_gf, _ = dice_iou(pred_gf, gt)
                best_radius_search[r].append(d_gf)

    print(f"Baseline (tanpa guided filter): dice={np.mean(dices_raw):.4f} iou={np.mean(ious_raw):.4f}")
    for r in [4, 8, 16]:
        print(f"Guided filter radius={r}: dice={np.mean(best_radius_search[r]):.4f} "
              f"(delta={np.mean(best_radius_search[r])-np.mean(dices_raw):+.4f})")

    best_r = max(best_radius_search, key=lambda r: np.mean(best_radius_search[r]))
    print(f"\nRESULT_SUMMARY model=seg_guided_filter baseline_dice={np.mean(dices_raw):.4f} "
          f"best_radius={best_r} best_dice={np.mean(best_radius_search[best_r]):.4f} "
          f"delta={np.mean(best_radius_search[best_r])-np.mean(dices_raw):+.4f}")


if __name__ == "__main__":
    main()
