"""Ekstrak fitur ABCD (Asymmetry, Border, Color, Diameter) dari mask segmentasi
Phase 1 untuk SELURUH gambar yang sudah di-crop.

Alur:
  1. Baca gambar crop RAW dari CROP_DIR
  2. Jalankan segmentasi Phase 1 (3-model ensemble, GPU) → binary mask
  3. Hitung fitur ABCD dari mask + gambar asli (CPU)
  4. Simpan ke CSV (image_id → 8 fitur ABCD)

Fitur yang diekstrak (8 dimensi):
  - asymmetry_h : 1 - IoU(mask, flip-horizontal(mask))   [0=simetris, 1=asimetris]
  - asymmetry_v : 1 - IoU(mask, flip-vertical(mask))
  - border_irreg: perimeter² / (4π × area)  [=1 untuk lingkaran, >1 = tidak teratur]
  - border_jagged: #contour_points / sqrt(area)  [kasar vs halus]
  - color_std_r : std deviasi channel R di dalam mask
  - color_std_g : std deviasi channel G di dalam mask
  - color_std_b : std deviasi channel B di dalam mask
  - compactness : area / (bounding-box area)  [=π/4 untuk lingkaran]

Fitur-fitur ini menargetkan PERSIS bottleneck MEL vs NV:
  - MEL cenderung asimetris, border tidak teratur, multi-warna
  - NV cenderung simetris, border halus, warna seragam

Output: BIRNet_Research/outputs/abcd_features.csv
Kolom: image_id, asymmetry_h, asymmetry_v, border_irreg, border_jagged,
        color_std_r, color_std_g, color_std_b, compactness

CPU-heavy (setelah segmentasi GPU). Bisa jalan paralel dengan training di GPU
jika segmentasi pipeline cukup ringan dalam batch kecil.

Usage:
    .venv/bin/python3 birnet_extract_abcd_features.py
    .venv/bin/python3 birnet_extract_abcd_features.py --cpu_only  # skip segmentasi, pakai mask sederhana dari Otsu
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import cv2
from PIL import Image

CROP_DIR = "BIRNet_Research/outputs/cropped_phase1_final"
OUT_CSV = "BIRNet_Research/outputs/abcd_features.csv"


# ── ABCD Feature Extraction (CPU) ──────────────────────────────────────

def _iou(mask_a, mask_b):
    inter = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return inter / max(union, 1)


def _asymmetry(mask_bin):
    """Asimetri horizontal dan vertikal: 1 - IoU(mask, flip(mask))."""
    h_flip = np.fliplr(mask_bin)
    v_flip = np.flipud(mask_bin)
    asym_h = 1.0 - _iou(mask_bin, h_flip)
    asym_v = 1.0 - _iou(mask_bin, v_flip)
    return asym_h, asym_v


def _border_features(mask_bin):
    """Border irregularity (circularity) dan jaggedness."""
    contours, _ = cv2.findContours(mask_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 1.0, 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, closed=True)
    if area < 1:
        return 1.0, 0.0
    # circularity: perimeter^2 / (4*pi*area), =1 untuk lingkaran sempurna
    circularity = (perim ** 2) / (4 * np.pi * area)
    # jaggedness: jumlah titik kontur / sqrt(area)
    jagged = len(cnt) / np.sqrt(area)
    return circularity, jagged


def _color_variability(img_rgb, mask_bin):
    """Standar deviasi warna R/G/B di dalam mask (color variegation)."""
    pixels = img_rgb[mask_bin > 0]  # (N, 3)
    if len(pixels) < 10:
        return 0.0, 0.0, 0.0
    std_r = float(np.std(pixels[:, 0]))
    std_g = float(np.std(pixels[:, 1]))
    std_b = float(np.std(pixels[:, 2]))
    return std_r, std_g, std_b


def _compactness(mask_bin):
    """Area / bounding-box-area. =π/4 untuk lingkaran, <1 untuk bentuk tidak teratur."""
    ys, xs = np.where(mask_bin > 0)
    if len(ys) < 2:
        return 0.0
    bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    return float(mask_bin.sum()) / max(bbox_area, 1)


def extract_abcd(img_rgb, mask_bin):
    """Ekstrak 8 fitur ABCD dari satu gambar + mask biner."""
    asym_h, asym_v = _asymmetry(mask_bin)
    circularity, jagged = _border_features(mask_bin)
    std_r, std_g, std_b = _color_variability(img_rgb, mask_bin)
    compact = _compactness(mask_bin)
    return {
        "asymmetry_h": round(asym_h, 5),
        "asymmetry_v": round(asym_v, 5),
        "border_irreg": round(circularity, 5),
        "border_jagged": round(jagged, 5),
        "color_std_r": round(std_r / 255.0, 5),   # normalize ke 0-1
        "color_std_g": round(std_g / 255.0, 5),
        "color_std_b": round(std_b / 255.0, 5),
        "compactness": round(compact, 5),
    }


# ── Mask generation ────────────────────────────────────────────────────

def mask_from_otsu(img_rgb):
    """Fallback CPU-only: Otsu thresholding pada channel L (LAB)."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_chan = lab[:, :, 0]
    _, mask = cv2.threshold(l_chan, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Ambil komponen terbesar
    from scipy import ndimage
    labeled, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        mask = (labeled == (np.argmax(sizes) + 1)).astype(np.uint8)
    return mask.astype(bool)


def mask_from_phase1(pipeline, img_rgb, img_size=224):
    """Mask dari Phase 1 segmentasi pipeline (GPU)."""
    import torch
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_224 = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    img_224_norm = (img_224.astype(np.float32) / 255.0 - mean) / std
    img_384 = cv2.resize(img_rgb, (384, 384), interpolation=cv2.INTER_LINEAR)
    img_384_norm = (img_384.astype(np.float32) / 255.0 - mean) / std
    t224 = torch.from_numpy(img_224_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    t384 = torch.from_numpy(img_384_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    masks = pipeline.predict(t224, t384)  # (1, 1, 224, 224)
    # Upsample mask ke ukuran gambar asli
    h, w = img_rgb.shape[:2]
    mask_native = cv2.resize((masks[0, 0] * 255).astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST)
    return (mask_native > 127)


# ── Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu_only", action="store_true",
                    help="Gunakan Otsu thresholding (CPU) sebagai pengganti Phase1 segmentasi (GPU)")
    args = ap.parse_args()

    # Kumpulkan semua image_id yang punya crop
    all_ids = sorted(f.replace(".jpg", "") for f in os.listdir(CROP_DIR) if f.endswith(".jpg"))
    print(f"Total gambar crop ditemukan: {len(all_ids)}")

    # Cek apakah sudah ada CSV parsial (resume-safe)
    done_ids = set()
    if os.path.exists(OUT_CSV):
        existing = pd.read_csv(OUT_CSV)
        done_ids = set(existing["image_id"].tolist())
        print(f"Sudah diekstrak sebelumnya: {len(done_ids)} — melanjutkan sisanya")

    remaining = [iid for iid in all_ids if iid not in done_ids]
    print(f"Sisa diproses: {len(remaining)}")
    if not remaining:
        print("Semua sudah selesai.")
        return

    # Load pipeline jika pakai GPU
    pipeline = None
    if not args.cpu_only:
        try:
            from birnet_seg_phase1_final import Phase1SegmentationPipeline
            pipeline = Phase1SegmentationPipeline()
            print("Phase1 segmentation pipeline loaded (GPU)")
        except Exception as e:
            print(f"Gagal load Phase1 pipeline: {e}")
            print("Fallback ke Otsu (CPU-only)")
            args.cpu_only = True

    if args.cpu_only:
        print("Mode: Otsu thresholding (CPU-only, tanpa GPU)")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    results = []
    t_start = time.time()
    write_header = not os.path.exists(OUT_CSV) or len(done_ids) == 0

    for i, iid in enumerate(remaining):
        img_path = os.path.join(CROP_DIR, f"{iid}.jpg")
        img_rgb = np.array(Image.open(img_path).convert("RGB"))

        # Generate mask
        if args.cpu_only:
            mask_bin = mask_from_otsu(img_rgb)
        else:
            mask_bin = mask_from_phase1(pipeline, img_rgb)

        # Ekstrak fitur ABCD
        feats = extract_abcd(img_rgb, mask_bin)
        feats["image_id"] = iid
        results.append(feats)

        # Flush setiap 500 gambar (resume-safe)
        if len(results) >= 500 or i == len(remaining) - 1:
            df_batch = pd.DataFrame(results)
            cols = ["image_id", "asymmetry_h", "asymmetry_v", "border_irreg", "border_jagged",
                    "color_std_r", "color_std_g", "color_std_b", "compactness"]
            df_batch = df_batch[cols]
            df_batch.to_csv(OUT_CSV, mode="a", header=write_header, index=False)
            write_header = False
            done_ids.update(r["image_id"] for r in results)
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(remaining) - i - 1) / max(rate, 1e-6)
            print(f"  {i+1}/{len(remaining)} selesai | {rate:.1f} img/s | ETA {eta/60:.1f} menit | batch flushed")
            results = []

    # Verifikasi output
    final = pd.read_csv(OUT_CSV)
    print(f"\nSELESAI. Total fitur ABCD diekstrak: {len(final)}")
    print(f"Output: {OUT_CSV}")
    print(f"\nStatistik fitur (seluruh dataset):")
    print(final.describe().round(4).to_string())

    # Quick sanity: distribusi per kelas (jika metadata HAM tersedia)
    try:
        import birnet_final as bf
        cfg = bf.Config()
        paths = bf.DatasetPaths(cfg)
        ham = bf.build_ham10000_dataframe(paths)[["image_id", "label"]]
        lab_name = {0: "mel", 1: "nv", 2: "bcc", 3: "akiec", 4: "bkl", 5: "df", 6: "vasc"}
        merged = final.merge(ham, on="image_id", how="inner")
        merged["dx"] = merged["label"].map(lab_name)
        print(f"\nRata-rata fitur per kelas (HAM10000, n={len(merged)}):")
        print(merged.groupby("dx")[["asymmetry_h", "asymmetry_v", "border_irreg",
                                      "color_std_r", "compactness"]].mean().round(4).to_string())
    except Exception:
        pass


if __name__ == "__main__":
    main()
