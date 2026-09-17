#!/usr/bin/env python3
"""De-vignette + tight lesion crop untuk BCN20000.

Latar (percobaanReal2.md, temuan Aug-14): alur "wp>crop" TIDAK ikut membuang
vignette FOV hitam. Citra dengan vignette berat (>15% hitam, ~26% test) mencatat
error-rate 42.5% vs 28.8% (+13.8pp). Skrip ini membenahinya secara klasik
(tanpa model; Phase-1 HAM under-crop di BCN):

  Tahap 1 — DE-VIGNETTE: deteksi lingkaran FOV terang, pangkas ke bounding-box FOV,
            (opsional) putihkan piksel di luar FOV supaya tidak menjadi fitur spurious.
  Tahap 2 — TIGHT LESION CROP: di dalam FOV, temukan lesi lewat deviasi warna LAB
            terhadap kulit (cincin tepi FOV = acuan kulit), ambil komponen
            lesi terbesar yang dekat pusat, potong + margin, jadikan kotak, resize.

Fallback aman: bila FOV ~ seluruh gambar (tanpa vignette) -> lewati tahap 1;
bila lesi tidak terdeteksi -> center-crop FOV. Warna RAW tetap dipertahankan (tanpa CLAHE).

Output: cache mode 'devig_crop' supaya bisa langsung dipakai retrain via pipe-system.

Usage:
  # cek visual dulu (SEBELUM proses penuh) — 12 contoh before/after:
  .venv/bin/python3 birnet_devignette_crop_bcn.py --preview
  # proses seluruh BCN -> cache:
  .venv/bin/python3 birnet_devignette_crop_bcn.py --out BIRNet_Research/outputs/bcn_pipe/devig_crop
"""
import argparse, os, glob
import numpy as np
import cv2

import config

BCN_IMG = config.BCN_IMG
DEFAULT_OUT = "BIRNet_Research/outputs/bcn_pipe/devig_crop"


def detect_fov(gray, dark_thr=40):
    """Hasilkan mask boolean area TERANG (FOV) = komponen terang terbesar, lubang ditambal."""
    lit = (gray > dark_thr).astype(np.uint8)
    lit = cv2.morphologyEx(lit, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(lit, connectivity=8)
    if n <= 1:
        return np.ones_like(gray, dtype=bool)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (lbl == big).astype(np.uint8)
    # tutup lubang: floodfill dari luar lalu balik
    h, w = mask.shape
    ff = mask.copy()
    m2 = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, m2, (0, 0), 1)
    holes = (ff == 0)
    mask = (mask.astype(bool) | holes)
    return mask


def skin_reference(lab, fov):
    """Acuan kulit = median LAB di CINCIN tepi bagian dalam FOV (umumnya kulit normal)."""
    fov_u = fov.astype(np.uint8)
    er = cv2.erode(fov_u, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    ring = fov & (~er.astype(bool))
    src = lab[ring] if ring.sum() > 100 else lab[fov]
    return np.median(src, axis=0)


def detect_lesion(lab, fov, ref, thr=22):
    """Mask lesi = piksel FOV yang berjarak jauh dari kulit (LAB), dibersihkan
    secara morfologi, komponen terbesar yang cenderung ke pusat."""
    dist = np.sqrt(((lab - ref) ** 2).sum(axis=2))
    les = ((dist > thr) & fov).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    les = cv2.morphologyEx(les, cv2.MORPH_CLOSE, k)
    les = cv2.morphologyEx(les, cv2.MORPH_OPEN, k)
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(les, connectivity=8)
    if n <= 1:
        return None
    h, w = fov.shape
    cx, cy = w / 2, h / 2
    best, best_score = None, -1
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.005 * h * w:  # singkirkan bintik
            continue
        dxy = np.hypot(cent[i][0] - cx, cent[i][1] - cy)
        score = area - 2.0 * dxy * dxy  # luas & dekat pusat
        if score > best_score:
            best_score, best = score, i
    return (lbl == best) if best is not None else None


def square_bbox(mask, shape, margin=0.18):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0, y1 - y0
    side = int(max(bw, bh) * (1 + 2 * margin))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    H, W = shape
    x0 = max(0, cx - side // 2); y0 = max(0, cy - side // 2)
    x1 = min(W, x0 + side); y1 = min(H, y0 + side)
    x0 = max(0, x1 - side); y0 = max(0, y1 - side)
    return x0, y0, x1, y1


def process(img_bgr, img_size=224, whiten_outside=True, margin=0.18):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    fov = detect_fov(gray)
    # Tahap 1 — de-vignette: bila vignette cukup besar, pangkas ke bbox FOV
    if fov.mean() < 0.92:
        ys, xs = np.where(fov)
        img_bgr = img_bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        fov = fov[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        if whiten_outside:  # netralkan area luar-FOV supaya bukan fitur (gunakan warna kulit)
            lab0 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            ref0 = skin_reference(lab0, fov)
            fill_bgr = cv2.cvtColor(np.uint8([[ref0]]), cv2.COLOR_LAB2BGR)[0, 0]
            img_bgr[~fov] = fill_bgr
    # Tahap 2 — pangkas lesi rapat
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = skin_reference(lab, fov)
    les = detect_lesion(lab, fov, ref)
    bb = square_bbox(les, img_bgr.shape[:2], margin) if les is not None else None
    if bb is None:  # fallback: center-crop 78% dari FOV
        h, w = img_bgr.shape[:2]; s = int(min(h, w) * 0.78)
        cx, cy = w // 2, h // 2
        bb = (max(0, cx - s // 2), max(0, cy - s // 2))
        bb = (bb[0], bb[1], bb[0] + s, bb[1] + s)
    x0, y0, x1, y1 = bb
    crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        crop = img_bgr
    return cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_LINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=BCN_IMG)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--margin", type=float, default=0.18)
    ap.add_argument("--no_whiten", action="store_true", help="jangan netralkan luar-FOV")
    ap.add_argument("--limit", type=int, default=0, help="proses N gambar pertama (0=semua)")
    ap.add_argument("--preview", action="store_true", help="simpan montase before/after 12 contoh, tak nulis cache")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*.jpg")) + glob.glob(os.path.join(args.src, "*.JPG")))
    if args.limit:
        files = files[:args.limit]

    if args.preview:
        from PIL import Image
        sel = files[::max(1, len(files) // 12)][:12]
        TS = 220
        canvas = Image.new("RGB", (2 * (TS + 8) + 8, len(sel) * (TS + 8) + 8), (20, 20, 20))
        for r, f in enumerate(sel):
            im = cv2.imread(f)
            if im is None:
                continue
            out = process(im, args.img_size, not args.no_whiten, args.margin)
            before = Image.fromarray(cv2.cvtColor(cv2.resize(im, (TS, TS)), cv2.COLOR_BGR2RGB))
            after = Image.fromarray(cv2.cvtColor(cv2.resize(out, (TS, TS)), cv2.COLOR_BGR2RGB))
            y = 8 + r * (TS + 8)
            canvas.paste(before, (8, y)); canvas.paste(after, (8 + TS + 8, y))
        outp = "hard_cases_vlm/_DEVIG_PREVIEW.png"
        os.makedirs("hard_cases_vlm", exist_ok=True)
        canvas.save(outp)
        print(f"PREVIEW (kiri=asli, kanan=de-vignette+crop): {outp}  ({len(sel)} contoh)")
        return

    os.makedirs(args.out, exist_ok=True)
    ok = 0
    for i, f in enumerate(files):
        im = cv2.imread(f)
        if im is None:
            continue
        iid = os.path.splitext(os.path.basename(f))[0]
        out = process(im, args.img_size, not args.no_whiten, args.margin)
        cv2.imwrite(os.path.join(args.out, f"{iid}.JPG"), out)
        ok += 1
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(files)} ...")
    print(f"Selesai: {ok}/{len(files)} -> {args.out}")
    print("Lalu retrain pakai cache ini (mis. env BCN_IMG mode / --img_mode devig_crop, sesuaikan pipeline).")


if __name__ == "__main__":
    main()
