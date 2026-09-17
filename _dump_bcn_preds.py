#!/usr/bin/env python3
"""Dump prediksi per-sampel BCN test (preds/probs/labs/image_id) utk model terbaik,
aligned ke split lesion-grouped seed-42. Dipakai VLM-cascade agar tak salah model.

Usage (kamu jalankan; ~5 menit GPU):
  .venv/bin/python3 _dump_bcn_preds.py \
    --ckpt BIRNet_Research/outputs/checkpoints/retrain_bcn/cls_vit_base_patch14_reg4_dinov2.lvd142m_multimodal_fold0_wp_crop_ws_denoise_rare6_best.pth \
    --img_mode "wp>crop" --out birnet_diag_bcn_best_test.json
"""
import argparse, json
import numpy as np, torch
import birnet_final as bf
import birnet_retrain_bcn as R
from train_specialist_expert import MultimodalClassifier, set_active_meta_fields
from birnet_retrain_bcn import make_loader
from birnet_external_zero_shot import DEVICE, BACKBONE
from birnet_eval_moe_pure_champion import tta_views
from sklearn.metrics import f1_score

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--img_mode", default="wp>crop")
ap.add_argument("--split_seed", type=int, default=42, help="split lesion-grouped (harus sama dg saat retrain fold ini)")
ap.add_argument("--img_size", type=int, default=0)
ap.add_argument("--tta", action="store_true", help="6-view TTA (protokol champion)")
ap.add_argument("--out", default="birnet_diag_bcn_best_test.json")
args = ap.parse_args()

set_active_meta_fields("localization")
cfg = bf.Config()
if args.img_size: cfg.img_size = args.img_size

bcn = R.build_bcn_df(args.img_mode)
_, _, test_df = R.lesion_grouped_split(bcn, args.split_seed)
test_df = test_df.reset_index(drop=True)
ids = test_df["image_id"].astype(str).tolist()

_, val_aug = bf.build_augmentations(cfg.img_size, seed=42)
loader = make_loader(test_df, cfg, val_aug, shuffle=False)

m = MultimodalClassifier(BACKBONE, num_classes=7, meta_dim=15, img_size=cfg.img_size).to(DEVICE)
m.load_state_dict(torch.load(args.ckpt, map_location=DEVICE, weights_only=False)["model_state"], strict=False)
m.eval()

probs, labs = [], []
with torch.no_grad():
    for imgs, metas, labels in loader:
        imgs, metas = imgs.to(DEVICE), metas.to(DEVICE)
        if args.tta:
            ps = torch.zeros((imgs.shape[0], 7), device=DEVICE)
            for v in tta_views(imgs):
                ps += torch.softmax(m(v, metas), 1)
            ps /= len(tta_views(imgs))
        else:
            ps = torch.softmax(m(imgs, metas), 1)
        probs.extend(ps.cpu().numpy()); labs.extend(labels.numpy())

probs = np.array(probs); labs = np.array(labs); preds = probs.argmax(1)
macro = f1_score(labs, preds, average="macro", zero_division=0)
json.dump({"img_mode": args.img_mode, "ckpt": args.ckpt, "tta": args.tta, "n": len(labs),
           "image_id": ids, "preds": preds.tolist(), "labs": labs.tolist(),
           "probs": probs.tolist()}, open(args.out, "w"))
print(f"macro-F1={macro:.4f} n={len(labs)} -> {args.out} (image_id disertakan)")
