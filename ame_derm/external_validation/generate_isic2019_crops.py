"""Buat crop Phase1 untuk 14885 citra eksternal ISIC2019 (BCN20000/MSK,
bukan HAM10000) -- pola IDENTIK dgn birnet_generate_phase1_crops.py
(mit_b3+ConvNeXt+hr_resnet50 ensemble, dice=0.9099), disimpan LANGSUNG ke
CROP_DIR yang SAMA dgn gambar HAM10000 (BIRNet_Research/outputs/
cropped_phase1_final/) -- image_id ISIC2019 dijamin tidak bertabrakan dgn
image_id HAM10000 (telah diverifikasi disjoint di
birnet_prepare_isic2019_external.py), sehingga SkinLesionDatasetPrecomputedCrop
dapat langsung memakainya tanpa perlu mengubah kode apa pun.

Prasyarat: birnet_prepare_isic2019_external.py sudah dijalankan (manifest +
raw images sudah diekstrak ke BIRNet_Research/external_data/isic2019/raw_new/).

GPU-heavy (Phase1 inference) -- JANGAN dijalankan berbarengan dgn training Swin
mana pun.

Usage:
    python birnet_generate_isic2019_crops.py
"""
import os
import time
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from ame_derm.common import birnet_final as bf
from ame_derm.segmentation.seg_phase1_final import Phase1SegmentationPipeline
from ame_derm.segmentation.generate_phase1_crops import crop_to_bbox
from ame_derm.experts.train_classifier_phase2_final import CROP_DIR

EXT_DIR = "BIRNet_Research/external_data/isic2019"
RAW_DIR = os.path.join(EXT_DIR, "raw_new")
MANIFEST_CSV = os.path.join(EXT_DIR, "external_pool_manifest.csv")
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RawImageDataset(Dataset):
    def __init__(self, image_ids, img_paths):
        self.entries = list(zip(image_ids, img_paths))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        image_id, img_path = self.entries[idx]
        img = np.array(bf.Image.open(img_path).convert("RGB")).astype(np.uint8)
        mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
        img_224 = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        img_224_norm = (img_224.astype(np.float32) / 255.0 - mean) / std
        img_384 = cv2.resize(img, (384, 384), interpolation=cv2.INTER_LINEAR)
        img_384_norm = (img_384.astype(np.float32) / 255.0 - mean) / std
        return (torch.from_numpy(img_224_norm.transpose(2, 0, 1)).float(),
                torch.from_numpy(img_384_norm.transpose(2, 0, 1)).float(),
                image_id, img_path)


def main():
    manifest = pd.read_csv(MANIFEST_CSV)
    manifest["img_path"] = manifest["image_id"].apply(lambda iid: os.path.join(RAW_DIR, f"{iid}.jpg"))
    has_raw = manifest["img_path"].apply(os.path.exists)
    print(f"Manifest: {len(manifest)} total | raw file tersedia: {has_raw.sum()}")
    manifest = manifest[has_raw].reset_index(drop=True)

    already_done = set(f.replace(".jpg", "") for f in os.listdir(CROP_DIR)) if os.path.isdir(CROP_DIR) else set()
    remaining = manifest[~manifest["image_id"].isin(already_done)].reset_index(drop=True)
    print(f"Sudah ada crop-nya (HAM10000 + eksternal sebelumnya): {len(already_done)} | Sisa diproses: {len(remaining)}")
    if len(remaining) == 0:
        print("Semua sudah selesai.")
        return

    ds = RawImageDataset(remaining["image_id"].tolist(), remaining["img_path"].tolist())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=6,
                         collate_fn=lambda batch: (
                             torch.stack([b[0] for b in batch]),
                             torch.stack([b[1] for b in batch]),
                             [b[2] for b in batch],
                             [b[3] for b in batch],
                         ))

    pipeline = Phase1SegmentationPipeline()
    os.makedirs(CROP_DIR, exist_ok=True)

    t_start = time.time()
    n_done = 0
    for imgs_224, imgs_384, image_ids, img_paths in loader:
        masks = pipeline.predict(imgs_224, imgs_384)
        for i in range(len(image_ids)):
            img_native = np.array(bf.Image.open(img_paths[i]).convert("RGB")).astype(np.uint8)
            cropped = crop_to_bbox(img_native, masks[i, 0])
            out_path = os.path.join(CROP_DIR, f"{image_ids[i]}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        n_done += len(image_ids)
        elapsed = time.time() - t_start
        rate = n_done / max(elapsed, 1e-6)
        eta = (len(remaining) - n_done) / max(rate, 1e-6)
        print(f"  {n_done}/{len(remaining)} selesai | {rate:.1f} img/s | ETA {eta/60:.1f} menit")

    print(f"\nSELESAI. Crop eksternal ISIC2019 baru: {n_done}")
    print(f"RESULT_SUMMARY model=generate_isic2019_crops total_new={n_done}")


if __name__ == "__main__":
    main()
