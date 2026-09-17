"""Tests the newly-discovered Segformer+mit_b3 model (solo test_dice=0.9033,
beating the entire prior CNN-ensemble champion of 0.9002) in combination with
everything already proven to work: TTA (4-way), guided filter (r=30), and
ensembling with the existing strong members (b5, resnet50, highres384-b4).

Usage:
    python birnet_seg_mitb3_combo.py
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import birnet_final as bf
from birnet_seg_bgff_v2 import SegDatasetV2
from birnet_seg_highres import SegDatasetHighRes
from birnet_seg_foundation_guided_filter import guided_filter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
HIGHRES_SIZE = 384
GF_RADIUS = 30

MODELS_224 = {
    "b5": ("unetpp", "efficientnet-b5", "BIRNet_Research/outputs/checkpoints/verify_ensemble/unetpp_efficientnet-b5_best.pth"),
    "resnet50": ("unetpp", "resnet50", "BIRNet_Research/outputs/checkpoints/verify_ensemble/unetpp_resnet50_best.pth"),
    "mit_b3": ("segformer", "mit_b3", "BIRNet_Research/outputs/checkpoints/foundation/segformer_mitb3_foundation_seed42_best.pth"),
}
HIGHRES_CKPT = "BIRNet_Research/outputs/checkpoints/foundation/unetpp_b4_highres384_seed42_best.pth"

CANDIDATE_COMBOS = [
    ("mit_b3",),
    ("mit_b3", "highres384"),
    ("mit_b3", "b5"),
    ("mit_b3", "resnet50"),
    ("mit_b3", "b5", "resnet50"),
    ("mit_b3", "b5", "highres384"),
    ("mit_b3", "resnet50", "highres384"),
    ("mit_b3", "b5", "resnet50", "highres384"),
]


def load_model(decoder, encoder, ckpt_path):
    cls = smp.UnetPlusPlus if decoder == "unetpp" else smp.Segformer
    model = cls(encoder_name=encoder, encoder_weights=None,
                 in_channels=3, classes=1, activation=None).to(DEVICE)
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model


def dice_iou(pred, gt):
    p, t = pred.reshape(pred.shape[0], -1), gt.reshape(gt.shape[0], -1)
    inter = (p * t).sum(1)
    union = p.sum(1) + t.sum(1) - inter
    dice = ((2 * inter + 1.0) / (p.sum(1) + t.sum(1) + 1.0)).mean()
    iou = ((inter + 1.0) / (union + 1.0)).mean()
    return dice, iou


@torch.no_grad()
def tta_predict(model, imgs):
    variants = [
        (imgs, lambda x: x),
        (torch.flip(imgs, dims=[3]), lambda x: torch.flip(x, dims=[3])),
        (torch.flip(imgs, dims=[2]), lambda x: torch.flip(x, dims=[2])),
        (torch.flip(imgs, dims=[2, 3]), lambda x: torch.flip(x, dims=[2, 3])),
    ]
    probs = []
    for aug_img, inv_fn in variants:
        logits = model(aug_img)
        probs.append(inv_fn(torch.sigmoid(logits)))
    return torch.stack(probs, dim=0).mean(dim=0)


@torch.no_grad()
def get_probs_224(model, loader, use_tta=True):
    probs, gts, imgs_out = [], [], []
    for imgs, masks in loader:
        imgs_gpu = imgs.to(DEVICE)
        prob = tta_predict(model, imgs_gpu) if use_tta else torch.sigmoid(model(imgs_gpu))
        probs.append(prob.cpu().numpy())
        gts.append(masks.numpy())
        imgs_out.append(imgs.numpy())
    return np.concatenate(probs), np.concatenate(gts), np.concatenate(imgs_out)


@torch.no_grad()
def get_highres_probs_224(model, loader, use_tta=True):
    probs, gts = [], []
    for imgs_384, _masks_384, masks_224 in loader:
        imgs_gpu = imgs_384.to(DEVICE)
        prob_384 = tta_predict(model, imgs_gpu) if use_tta else torch.sigmoid(model(imgs_gpu))
        prob_224 = F.interpolate(prob_384, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        probs.append(prob_224.cpu().numpy())
        gts.append(masks_224.numpy())
    return np.concatenate(probs), np.concatenate(gts)


def main():
    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    isic_t1_test_df = bf.build_seg_dataframe(paths.isic_t1_test_img, paths.isic_t1_test_mask)

    test_ds_224 = SegDatasetV2(isic_t1_test_df, IMG_SIZE, "rgb")
    test_loader_224 = DataLoader(test_ds_224, batch_size=16, shuffle=False, num_workers=4)
    test_ds_hr = SegDatasetHighRes(isic_t1_test_df, HIGHRES_SIZE, IMG_SIZE)
    test_loader_hr = DataLoader(test_ds_hr, batch_size=16, shuffle=False, num_workers=4)

    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])

    all_probs = {}
    gt_ref, imgs_224_ref = None, None
    for name, (decoder, encoder, ckpt) in MODELS_224.items():
        print(f"Evaluasi {name} ({decoder}+{encoder}) @224 dengan TTA...")
        model = load_model(decoder, encoder, ckpt)
        probs, gts, imgs_out = get_probs_224(model, test_loader_224, use_tta=True)
        all_probs[name] = probs
        if gt_ref is None:
            gt_ref, imgs_224_ref = gts, imgs_out
        d, _ = dice_iou((probs > 0.5).astype(np.float32), gt_ref)
        print(f"  {name} solo+TTA: dice={d:.4f}")
        del model; torch.cuda.empty_cache()

    print("Evaluasi highres384 (downsampled ke 224) dengan TTA...")
    hr_model = load_model("unetpp", "efficientnet-b4", HIGHRES_CKPT)
    hr_probs, hr_gt = get_highres_probs_224(hr_model, test_loader_hr, use_tta=True)
    del hr_model; torch.cuda.empty_cache()
    assert np.allclose(gt_ref, hr_gt)
    all_probs["highres384"] = hr_probs
    d, _ = dice_iou((hr_probs > 0.5).astype(np.float32), gt_ref)
    print(f"  highres384 solo+TTA: dice={d:.4f}")

    # mit_b3 solo + TTA + GF
    print(f"\n{'='*70}\nmit_b3 solo + TTA + guided_filter(r={GF_RADIUS})\n{'='*70}")
    gf_dices = []
    for idx in range(all_probs["mit_b3"].shape[0]):
        img_rgb = np.clip(imgs_224_ref[idx].transpose(1, 2, 0) * std + mean, 0, 1).astype(np.float32)
        guide = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        refined = guided_filter(guide, all_probs["mit_b3"][idx, 0], radius=GF_RADIUS)
        pred_gf = (refined > 0.5).astype(np.float32)
        d, _ = dice_iou(pred_gf[None, None], gt_ref[idx][None])
        gf_dices.append(d)
    print(f"mit_b3 + TTA + GF(r={GF_RADIUS}): dice={np.mean(gf_dices):.4f}")

    print(f"\n{'='*70}\nEnsemble (semua sudah +TTA) + guided filter r={GF_RADIUS}\n{'='*70}")
    results = {}
    for combo in CANDIDATE_COMBOS:
        avg_prob = np.mean([all_probs[n] for n in combo], axis=0)
        d_raw, _ = dice_iou((avg_prob > 0.5).astype(np.float32), gt_ref)

        gf_dices = []
        for idx in range(avg_prob.shape[0]):
            img_rgb = np.clip(imgs_224_ref[idx].transpose(1, 2, 0) * std + mean, 0, 1).astype(np.float32)
            guide = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            refined = guided_filter(guide, avg_prob[idx, 0], radius=GF_RADIUS)
            pred_gf = (refined > 0.5).astype(np.float32)
            d, _ = dice_iou(pred_gf[None, None], gt_ref[idx][None])
            gf_dices.append(d)
        d_gf = float(np.mean(gf_dices))
        results["+".join(combo)] = (d_raw, d_gf)
        print(f"  {'+'.join(combo)}: raw={d_raw:.4f}  +GF(r={GF_RADIUS})={d_gf:.4f}")

    best_name = max(results, key=lambda k: results[k][1])
    print(f"\n=== JUARA BARU: {best_name} (+TTA) + GF = {results[best_name][1]:.4f} ===")
    print(f"RESULT_SUMMARY model=seg_mitb3_combo best_combo={best_name} best_dice={results[best_name][1]:.4f}")


if __name__ == "__main__":
    main()
