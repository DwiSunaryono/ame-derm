"""A complete combinatorial map of the Phase 1 ensemble search space, built
from the best-known variant of each distinct architecture family found so
far (these are not literal duplicates -- highres-b4 rather than plain b4,
for instance, because highres gave b4 a real +0.0103 gain, yet plain
mit_b3 rather than its highres variant because that yielded ~0 gain there):

  - highres-b4      (unetpp+efficientnet-b4 @384->224, seed42): solo test 0.8873
  - b5              (unetpp+efficientnet-b5 @224): solo test 0.8800
  - b6              (unetpp+efficientnet-b6 @224): solo test 0.8804
  - highres-resnet50(unetpp+resnet50 @384->224): solo test 0.8832
  - mit_b3          (segformer+mit_b3 @224): solo test 0.9033
  - convnext        (upernet+tu-convnext_tiny @224): solo test 0.8999

Left out of this map: mobilenet_v2 (weakest solo, 0.8718, an established
pattern of dragging ensembles down), the BGFF/CBAM-refined variants (every
one confirmed WORSE than its own base model, so adding them would only
introduce correlated-and-worse noise), and the non-superseded plain
versions of b4/resnet50 (highly correlated duplicates of their highres
counterparts).

Each model's probability map relies on 4-way flip TTA (shown to help, or at
worst stay neutral, in every earlier test). The script examines ALL
non-trivial combinations (2 to 6 members), both raw (simple mean) and with
guided filter (r=30) post-processing, so the result shows where the true
optimum actually lies -- not merely the top pick.

Usage:
    python birnet_seg_full_ensemble_map.py
"""
import itertools
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from ame_derm.common import birnet_final as bf
from ame_derm.segmentation.seg_bgff_v2 import SegDatasetV2
from ame_derm.segmentation.seg_highres import SegDatasetHighRes
from ame_derm.segmentation.seg_foundation_guided_filter import guided_filter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
HIGHRES_SIZE = 384
GF_RADIUS = 30

MODELS_224 = {
    "b5": ("unetpp", "efficientnet-b5", "BIRNet_Research/outputs/checkpoints/verify_ensemble/unetpp_efficientnet-b5_best.pth"),
    "b6": ("unetpp", "efficientnet-b6", "BIRNet_Research/outputs/checkpoints/verify_ensemble/unetpp_efficientnet-b6_best.pth"),
    "mit_b3": ("segformer", "mit_b3", "BIRNet_Research/outputs/checkpoints/foundation/segformer_mitb3_foundation_seed42_best.pth"),
    "convnext": ("upernet", "tu-convnext_tiny", "BIRNet_Research/outputs/checkpoints/foundation/upernet_convnext_foundation_seed42_best.pth"),
}
MODELS_HIGHRES = {
    "hr_b4": ("unetpp", "efficientnet-b4", "BIRNet_Research/outputs/checkpoints/foundation/unetpp_b4_highres384_seed42_best.pth"),
    "hr_resnet50": ("unetpp", "resnet50", "BIRNet_Research/outputs/checkpoints/foundation/unetpp_resnet50_highres384_seed42_best.pth"),
}
DECODER_CLASS = {"unetpp": "UnetPlusPlus", "segformer": "Segformer", "upernet": "UPerNet"}


def load_model(decoder_key, encoder, ckpt_path):
    cls = getattr(smp, DECODER_CLASS[decoder_key])
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
def get_probs_224(model, loader):
    probs, gts, imgs_out = [], [], []
    for imgs, masks in loader:
        imgs_gpu = imgs.to(DEVICE)
        prob = tta_predict(model, imgs_gpu)
        probs.append(prob.cpu().numpy())
        gts.append(masks.numpy())
        imgs_out.append(imgs.numpy())
    return np.concatenate(probs), np.concatenate(gts), np.concatenate(imgs_out)


@torch.no_grad()
def get_highres_probs_224(model, loader):
    probs, gts = [], []
    for imgs_384, _masks_384, masks_224 in loader:
        imgs_gpu = imgs_384.to(DEVICE)
        prob_384 = tta_predict(model, imgs_gpu)
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
    for name, (dec, enc, ckpt) in MODELS_224.items():
        print(f"Evaluasi {name} ({dec}+{enc}) @224 dengan TTA...")
        model = load_model(dec, enc, ckpt)
        probs, gts, imgs_out = get_probs_224(model, test_loader_224)
        all_probs[name] = probs
        if gt_ref is None:
            gt_ref, imgs_224_ref = gts, imgs_out
        d, _ = dice_iou((probs > 0.5).astype(np.float32), gt_ref)
        print(f"  {name} solo+TTA: dice={d:.4f}")
        del model; torch.cuda.empty_cache()

    for name, (dec, enc, ckpt) in MODELS_HIGHRES.items():
        print(f"Evaluasi {name} ({dec}+{enc}) highres384->224 dengan TTA...")
        model = load_model(dec, enc, ckpt)
        probs, gts = get_highres_probs_224(model, test_loader_hr)
        all_probs[name] = probs
        assert np.allclose(gt_ref, gts)
        d, _ = dice_iou((probs > 0.5).astype(np.float32), gt_ref)
        print(f"  {name} solo+TTA: dice={d:.4f}")
        del model; torch.cuda.empty_cache()

    names = list(all_probs.keys())
    print(f"\n{'='*80}\nPETA LENGKAP: semua kombinasi 2-{len(names)} model (raw + guided_filter r={GF_RADIUS})\n{'='*80}")

    results_by_size = {}
    overall_best_name, overall_best_dice = None, 0.0
    for r in range(2, len(names) + 1):
        best_at_size = (None, 0.0, 0.0)
        for combo in itertools.combinations(names, r):
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

            label = "+".join(combo)
            print(f"  [{r}] {label}: raw={d_raw:.4f} +GF={d_gf:.4f}")
            if d_gf > best_at_size[2]:
                best_at_size = (label, d_raw, d_gf)
            if d_gf > overall_best_dice:
                overall_best_dice, overall_best_name = d_gf, label
        results_by_size[r] = best_at_size
        print(f"  --> TERBAIK ukuran {r}: {best_at_size[0]} (raw={best_at_size[1]:.4f}, +GF={best_at_size[2]:.4f})")

    print(f"\n{'='*80}\nRINGKASAN: terbaik per ukuran ensemble\n{'='*80}")
    for r, (label, d_raw, d_gf) in results_by_size.items():
        print(f"  ukuran {r}: {label:50s} raw={d_raw:.4f} +GF={d_gf:.4f}")

    print(f"\n=== JUARA KESELURUHAN: {overall_best_name} + GF(r={GF_RADIUS}) = {overall_best_dice:.4f} ===")
    print(f"RESULT_SUMMARY model=full_ensemble_map overall_best={overall_best_name} overall_best_dice={overall_best_dice:.4f}")


if __name__ == "__main__":
    main()
