#!/usr/bin/env python3
"""Analisis GEN hasil retrain (wp>crop) pada BCN test: PRF per-kelas,
confusion matrix mentah, serta sebaran confidence (benar vs salah)."""
import os
import json
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from ame_derm.common import birnet_final as bf
from ame_derm.external_validation import retrain_bcn as rb
from ame_derm.experts.train_specialist_expert import MultimodalClassifier, set_active_meta_fields

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_MODE = "wp>crop"
SAFE = IMG_MODE.replace(">", "_")
CKPT = os.path.join(rb.CKPT_DIR, f"cls_{rb.BACKBONE}_multimodal_fold0_{SAFE}_best.pth")
LAB_NAMES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]

def main():
    set_active_meta_fields("localization")
    cfg = bf.Config()
    bcn = rb.build_bcn_df(IMG_MODE)
    train_df, val_df, test_df = rb.lesion_grouped_split(bcn, 42)
    _, val_aug = bf.build_augmentations(cfg.img_size, seed=42)
    loader = rb.make_loader(test_df, cfg, val_aug, shuffle=False)

    model = MultimodalClassifier(rb.BACKBONE, num_classes=cfg.num_classes,
                                 meta_dim=15, img_size=cfg.img_size).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=False)["model_state"])
    model.eval()

    preds, probs, labs = [], [], []
    with torch.no_grad():
        for imgs, metas, labels in loader:
            imgs, metas = imgs.to(DEVICE), metas.to(DEVICE)
            p = torch.softmax(model(imgs, metas), dim=1)
            valid = labels >= 0
            probs.extend(p[valid].cpu().numpy())
            labs.extend(labels[valid].numpy())
            preds.extend(p[valid].argmax(1).cpu().numpy())
    preds, probs, labs = np.array(preds), np.array(probs), np.array(labs)

    print("=" * 70)
    print(f"Diagnosa champion GEN | img_mode={IMG_MODE} | n_test={len(labs)}")
    print(f"Distribusi test: {np.bincount(labs, minlength=7).tolist()}")
    print("=" * 70)
    print(classification_report(labs, preds, target_names=LAB_NAMES, digits=4,
                                zero_division=0))

    cm = confusion_matrix(labs, preds, labels=range(7))
    print("\nConfusion (baris=sebenarnya, kolom=prediksi):")
    print("      " + "   ".join(f"{n:>4s}" for n in LAB_NAMES))
    for i, row in enumerate(cm):
        print(f"{LAB_NAMES[i]:>5s} " + "   ".join(f"{v:4d}" for v in row))

    conf = probs.max(1)
    print("\nConfidence (max prob) — benar vs salah:")
    for ok, tag in [(True, "BENAR"), (False, "SALAH")]:
        m = preds == labs if ok else preds != labs
        if m.sum() == 0:
            print(f"  {tag}: n=0")
            continue
        c = conf[m]
        print(f"  {tag}: n={m.sum():4d}  {c.mean():.3f} ± {c.std():.3f}  "
              f"[<0.5:{np.mean(c<0.5)*100:.1f}% | 0.5-0.7:{np.mean((c>=0.5)&(c<0.7))*100:.1f}% "
              f"| >=0.7:{np.mean(c>=0.7)*100:.1f}%]")

    conf_hit = {}
    for i in range(7):
        m = labs == i
        if m.sum() == 0:
            continue
        ok = (preds == labs) & m
        conf_hit[i] = (float(conf[m].mean()), int(ok.sum()), int(m.sum()))
    print("\nConfidence rata-rata TRUTH (benar/berlabel):")
    for i in range(7):
        cmv, hit, total = conf_hit.get(i, (0, 0, 0))
        print(f"  {LAB_NAMES[i]:5s} conf={cmv:.3f}  hit={hit}/{total} "
              f"({100*hit/max(total,1):.0f}%)")

    macro = f1_score(labs, preds, average="macro", zero_division=0)
    print(f"\n>>> F1 macro = {macro:.4f}")
    json.dump({"img_mode": IMG_MODE, "n": int(len(labs)),
               "confusion": cm.tolist(), "preds": preds.tolist(), "labs": labs.tolist(),
               "probs": probs.tolist()},
              open("birnet_diag_bcn_gen_test.json", "w"), indent=1)

if __name__ == "__main__":
    main()