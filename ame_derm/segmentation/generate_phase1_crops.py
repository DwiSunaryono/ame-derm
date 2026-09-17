"""Generates lesion crops ahead of time for every HAM10000 image by running
the FINALIZED Phase 1 segmentation pipeline (birnet_seg_phase1_final.py:
mit_b3+ConvNeXt+hr_resnet50 ensemble, TTA, guided filter, threshold=0.46,
LCC filter -- dice=0.9099). Earlier crop-experiment scripts instead inferred
masks per image on the fly while the classifier trained.

Reason for pre-computing rather than doing it live: the legacy scripts'
mask_source=unetpp* choices performed just a SINGLE lightweight B4 forward
pass for each image each epoch -- cheap enough to run during training. The
Phase 1 pipeline, by contrast, combines 3 models with 4-way TTA plus a CPU
guided filter and LCC; running that for every image every epoch over 5
GroupKFold folds and many epochs would cost far too much time. Executing it
ONCE across all 10015 images and writing the crops to disk is therefore the
only workable option.

Output: RGB images saved as .jpg, cropped but not yet resized/CLAHE'd (those
steps stay in the classifier's own preprocessing, keeping augmentation on a
consistent, simple format), one file per HAM10000 image_id, following the
mask bounding box + pad_frac=0.15 convention shared by all prior crop
experiments in this project.

Usage:
    python birnet_generate_phase1_crops.py
"""
import os
import time
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from ame_derm.common import birnet_final as bf
from ame_derm.segmentation.seg_phase1_final import Phase1SegmentationPipeline

OUT_DIR = "BIRNet_Research/outputs/cropped_phase1_final"
PAD_FRAC = 0.15
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RawImageDataset(Dataset):
    """Reads raw HAM10000 images and yields normalized 224 and 384 tensors
    (covering the Phase1 pipeline's two input resolutions), together with
    the untouched full-resolution image (used for cropping) and its native
    dimensions."""

    def __init__(self, df):
        self.entries = list(zip(df["image_id"], df["img_path"]))

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


def crop_to_bbox(img, mask_224, pad_frac=PAD_FRAC):
    """mask_224: (224,224) binary float32. The mask is upsampled to img's
    native size to derive the bounding box, after which the ORIGINAL image
    gets cropped."""
    h, w = img.shape[:2]
    mask_native = cv2.resize((mask_224 * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mask_native > 127)
    if len(ys) == 0:
        return img
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    bh, bw = y1 - y0, x1 - x0
    py, px = int(bh * pad_frac), int(bw * pad_frac)
    y0 = max(0, y0 - py); y1 = min(h, y1 + py)
    x0 = max(0, x0 - px); x1 = min(w, x1 + px)
    if y1 <= y0 or x1 <= x0:
        return img
    return img[y0:y1, x0:x1]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    ham_df = bf.build_ham10000_dataframe(paths)
    print(f"Total gambar HAM10000: {len(ham_df)}")

    already_done = set(f.replace(".jpg", "") for f in os.listdir(OUT_DIR)) if os.path.isdir(OUT_DIR) else set()
    remaining_df = ham_df[~ham_df["image_id"].isin(already_done)].reset_index(drop=True)
    print(f"Sudah selesai sebelumnya: {len(already_done)} | Sisa diproses: {len(remaining_df)}")
    if len(remaining_df) == 0:
        print("Semua sudah selesai, tidak ada yang perlu diproses.")
        return

    ds = RawImageDataset(remaining_df)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=6,
                         collate_fn=lambda batch: (
                             torch.stack([b[0] for b in batch]),
                             torch.stack([b[1] for b in batch]),
                             [b[2] for b in batch],
                             [b[3] for b in batch],
                         ))

    pipeline = Phase1SegmentationPipeline()

    t_start = time.time()
    n_done = 0
    for imgs_224, imgs_384, image_ids, img_paths in loader:
        masks = pipeline.predict(imgs_224, imgs_384)  # (B,1,224,224)
        for i in range(len(image_ids)):
            img_native = np.array(bf.Image.open(img_paths[i]).convert("RGB")).astype(np.uint8)
            cropped = crop_to_bbox(img_native, masks[i, 0])
            out_path = os.path.join(OUT_DIR, f"{image_ids[i]}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        n_done += len(image_ids)
        elapsed = time.time() - t_start
        rate = n_done / max(elapsed, 1e-6)
        eta = (len(remaining_df) - n_done) / max(rate, 1e-6)
        print(f"  {n_done}/{len(remaining_df)} selesai | {rate:.1f} img/s | ETA {eta/60:.1f} menit")

    print(f"\nSELESAI. Total crop tersimpan di {OUT_DIR}: {len(os.listdir(OUT_DIR))}")
    print(f"RESULT_SUMMARY model=generate_phase1_crops total={len(os.listdir(OUT_DIR))}")


if __name__ == "__main__":
    main()
