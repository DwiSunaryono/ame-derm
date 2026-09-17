"""Persiapan data eksternal ISIC 2019 (BCN20000 + MSK, di luar HAM10000)
-- lever pertama yang GENUINELY berbeda dari semua eksperimen boundary-
injection sebelumnya: menambah gambar REAL baru (bukan mengedit ulang
10015 gambar HAM10000 yang sama), khusus utk memperkaya kelas minoritas
(DF, VASC, AKIEC) dgn keragaman piksel yang genuinely baru.

Verifikasi (dicek langsung dari GroundTruth CSV): seluruh 10015 gambar
HAM10000 adalah SUBSET dari 25331 gambar ISIC 2019 (overlap 10015/10015),
menyisakan 15316 gambar BENAR-BENAR BARU dari BCN20000/MSK. Distribusi
kelas pada 15316 gambar baru itu: MEL=3409 NV=6170 BCC=2809 AK=737
BKL=1525 DF=124 VASC=111 SCC=431 -- SCC (squamous cell carcinoma) TIDAK
ADA di taksonomi 7-kelas HAM10000 kita, jadi didrop. UNK juga didrop.
AK dipetakan ke AKIEC (kelas yg sama, notasi beda).

Leakage safety: gambar eksternal ini akan DITAMBAHKAN ke TRAIN saja (utk
SEMUA fold sekaligus, sama persis) -- karena tidak overlap dgn HAM10000
sama sekali (lesion_id benar2x beda), tidak ada risiko kebocoran ke VAL/
TEST manapun, walau lesion_id sebagian externalnya kosong (1 gambar per
lesion di BCN20000). VAL & TEST tetap 100% HAM10000 real, tidak berubah
sama sekali -- test_f1 dari eksperimen ini tetap bisa dibandingkan
langsung dgn baseline 0.7859 dan semua eksperimen boundary-injection.

Langkah script ini (CPU-only, aman dijalankan kapan saja termasuk saat
GPU dipakai training lain):
  1. Baca GroundTruth+Metadata CSV (harus sudah didownload).
  2. Filter ke 15316 gambar baru, mapping kelas, buang SCC/UNK.
  3. Simpan manifest CSV (image_id, label).
  4. Ekstrak HANYA file jpg yg dibutuhkan (bukan semua 25331) dari
     ISIC_2019_Training_Input.zip ke folder raw sementara.

Usage:
    python birnet_prepare_isic2019_external.py
"""
import argparse
import os
import zipfile
import pandas as pd

import birnet_final as bf

EXT_DIR = "BIRNet_Research/external_data/isic2019"
GT_CSV = os.path.join(EXT_DIR, "ISIC_2019_Training_GroundTruth.csv")
ZIP_PATH = os.path.join(EXT_DIR, "ISIC_2019_Training_Input.zip")
RAW_OUT_DIR = os.path.join(EXT_DIR, "raw_new")
MANIFEST_CSV = os.path.join(EXT_DIR, "external_pool_manifest.csv")

CLASS_NAMES = ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC")
ISIC_TO_OURS = {"MEL": "MEL", "NV": "NV", "BCC": "BCC", "AK": "AKIEC", "BKL": "BKL", "DF": "DF", "VASC": "VASC"}


def build_manifest():
    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    ham_df = bf.build_ham10000_dataframe(paths)
    ham_ids = set(ham_df["image_id"])
    print(f"HAM10000 total: {len(ham_ids)}")

    gt = pd.read_csv(GT_CSV)
    print(f"ISIC2019 total: {len(gt)}")
    new_df = gt[~gt["image"].isin(ham_ids)].reset_index(drop=True)
    print(f"Gambar BARU (bukan HAM10000): {len(new_df)}")

    rows = []
    for _, r in new_df.iterrows():
        # buang SCC (kelas tak ada di taksonomi kita) & UNK
        if r.get("SCC", 0.0) == 1.0 or r.get("UNK", 0.0) == 1.0:
            continue
        for isic_col, our_name in ISIC_TO_OURS.items():
            if r[isic_col] == 1.0:
                rows.append(dict(image_id=r["image"], label=CLASS_NAMES.index(our_name), true_name=our_name))
                break
    manifest = pd.DataFrame(rows)
    os.makedirs(EXT_DIR, exist_ok=True)
    manifest.to_csv(MANIFEST_CSV, index=False)
    print(f"\nManifest eksternal tersimpan: {MANIFEST_CSV}")
    print(f"Total eligible (setelah buang SCC/UNK): {len(manifest)}")
    print(manifest.groupby("true_name").size().reindex(CLASS_NAMES))
    return manifest


def extract_needed_images(manifest):
    if not os.path.exists(ZIP_PATH):
        print(f"PERINGATAN: {ZIP_PATH} belum ada (download belum selesai) -- lewati ekstraksi utk sekarang.")
        return
    os.makedirs(RAW_OUT_DIR, exist_ok=True)
    already = set(f.replace(".jpg", "") for f in os.listdir(RAW_OUT_DIR))
    needed_ids = set(manifest["image_id"]) - already
    print(f"Sudah diekstrak sebelumnya: {len(already)} | Perlu diekstrak: {len(needed_ids)}")
    if not needed_ids:
        print("Semua sudah diekstrak.")
        return

    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()
        # arsip resmi ISIC2019: folder "ISIC_2019_Training_Input/<id>.jpg"
        name_map = {}
        for n in names:
            base = os.path.basename(n)
            if base.endswith(".jpg"):
                name_map[base[:-4]] = n
        n_extracted = 0
        for i, image_id in enumerate(sorted(needed_ids)):
            if image_id not in name_map:
                continue
            with zf.open(name_map[image_id]) as src, open(os.path.join(RAW_OUT_DIR, f"{image_id}.jpg"), "wb") as dst:
                dst.write(src.read())
            n_extracted += 1
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(needed_ids)} diekstrak...")
    print(f"Ekstraksi selesai: {n_extracted} gambar baru -> {RAW_OUT_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_extract", action="store_true",
                         help="Hanya buat manifest, jangan ekstrak zip (mis. kalau zip belum selesai download).")
    args = parser.parse_args()
    manifest = build_manifest()
    if not args.skip_extract:
        extract_needed_images(manifest)
    print(f"\nRESULT_SUMMARY model=prepare_isic2019_external total_eligible={len(manifest)}")


if __name__ == "__main__":
    main()
