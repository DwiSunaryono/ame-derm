"""Evaluasi ablasi ABCD 4-grup dari checkpoint (fold-0 test, konsisten tanpa TTA).
Tiap grup memakai mask ABCD sesuai training-nya masing-masing. Bandingkan
macro-F1 + per-kelas.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from ame_derm.common import birnet_final as bf
from ame_derm.abcd_ablation import train_multimodal_abcd as tabcd  # level modul: patch RAW + helper
from ame_derm.abcd_ablation.train_multimodal_ablation import MultimodalClassifier
from ame_derm.experts.train_classifier_phase2_final import CROP_DIR
from ame_derm.data_splits.representative_val_split import stratified_ham_kfold_fully_representative
from ame_derm.data_splits.diagnose_metadata_imbalance import detect_metadata_fields

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BB = "vit_base_patch14_reg4_dinov2"
LAB = {0: "mel", 1: "nv", 2: "bcc", 3: "akiec", 4: "bkl", 5: "df", 6: "vasc"}


def main():
    tabcd.tma.set_active_meta_fields("localization")
    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    ham = bf.build_ham10000_dataframe(paths)
    ham = ham[ham["image_id"].apply(lambda i: os.path.exists(os.path.join(CROP_DIR, f"{i}.jpg")))].reset_index(drop=True)
    bucketed, _ = detect_metadata_fields(ham, 9)
    ov = [c for c in bucketed.columns if c in ham.columns]
    hm = __import__("pandas").concat([ham.drop(columns=ov), bucketed], axis=1)
    for _r, fi, _tr, _va, te in stratified_ham_kfold_fully_representative(hm, cfg, ["dataset"], n_splits=5, n_repeats=1):
        if fi == 0:
            test_df = ham.iloc[te].reset_index(drop=True); break
    _, val_aug = bf.build_augmentations(cfg.img_size, seed=42)
    abcd_map = tabcd.load_abcd_map(tabcd.ABCD_CSV)

    print(f"{'grup':10s} {'macro':>7s} {'mel':>6s} {'nv':>6s} {'akiec':>6s} {'bkl':>6s}")
    rows = {}
    for group in ["none", "geometry", "color", "all"]:
        ckpt = f"{tabcd.CKPT_DIR_BASE}/cls_{BB}_abcd_{group}_fold0_best.pth"
        if not os.path.exists(ckpt):
            print(f"{group}: checkpoint tak ada"); continue
        mask = tabcd.abcd_group_mask(group)
        ds = tabcd.SkinLesionDatasetABCD(test_df, cfg.img_size, abcd_map=abcd_map,
                                         transform=val_aug, abcd_mask=mask)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
        model = MultimodalClassifier(BB, num_classes=7, meta_dim=23, img_size=cfg.img_size).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False)["model_state"])
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for img, meta, lab in loader:
                out = model(img.to(DEVICE), meta.to(DEVICE))
                preds.extend(out.argmax(1).cpu().numpy()); ys.extend(lab.numpy())
        ys, preds = np.array(ys), np.array(preds)
        macro = f1_score(ys, preds, average="macro", zero_division=0)
        per = f1_score(ys, preds, average=None, labels=list(range(7)), zero_division=0)
        rows[group] = macro
        print(f"{group:10s} {macro:7.4f} {per[0]:6.3f} {per[1]:6.3f} {per[3]:6.3f} {per[4]:6.3f}")

    if "none" in rows:
        b = rows["none"]
        print("\nΔ vs placebo (none):")
        for g in ["geometry", "color", "all"]:
            if g in rows:
                print(f"  {g:10s}: {rows[g]-b:+.4f}")
        print(f"\nRINGKAS abcd_ablation: none={rows.get('none',0):.4f} geometry={rows.get('geometry',0):.4f} "
              f"color={rows.get('color',0):.4f} all={rows.get('all',0):.4f}")


if __name__ == "__main__":
    main()
