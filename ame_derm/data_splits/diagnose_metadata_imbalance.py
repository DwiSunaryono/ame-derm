"""
Diagnosis OTOMATIS ketidakseimbangan representasi METADATA (bukan cuma
kelas) di dalam TRAIN-VAL -- generalisasi dari temuan manual (MEL x usia
korelasi n-vs-error = -0.814, NV x lokasi = -0.580), TAPI field metadata
mana yg diuji TIDAK di-hardcode ke usia/sex/lokasi saja -- field APAPUN
yg ada di metadata dataset (dideteksi otomatis dari kolom yg tersedia,
minus kolom identitas/path/label) ikut diuji, begitu pula SEMUA kelas.
Kalau di fold lain ternyata kombinasi (kelas, field) yg berbeda yg jadi
sumbu ketidakseimbangan -- script ini akan menemukannya sendiri, bukan
diarahkan ke MEL/NV/usia/lokasi secara khusus.

PRINSIP (ide user, analogi regresi suhu tubuh manusia): dalam SATU kelas
yg sama, subgrup (mis. kelas X pada nilai field Y tertentu) yg JARANG
direpresentasikan di training akan "tertarik" ke pola mayoritas kelasnya
sendiri saat prediksi -- diukur via korelasi NEGATIF antara n
(representasi) dan error_rate (dari audit out-of-fold) di seluruh nilai
suatu field metadata, DI DALAM satu kelas. Kalau korelasinya kuat negatif
utk (kelas, field) tertentu, field itu genuinely jadi sumbu ketidak-
seimbangan tersembunyi utk kelas itu -- bukan cuma noise.

DETEKSI FIELD OTOMATIS: field kandidat = semua kolom pada dataframe
metadata dataset SELAIN identitas/path/label (mis utk HAM10000: age, sex,
localization, dataset -- tapi kalau kolom metadata lain ditambahkan di
masa depan, otomatis ikut diuji tanpa ubah kode). Kolom numerik (spt age)
di-bucket otomatis (quantile-based, jumlah bin dari --n_bins_numeric);
kolom kategorikal dipakai apa adanya.

LEAKAGE SAFETY: HANYA memakai audit out-of-fold TRAIN-VAL (pool_fold yg
dipilih) -- TIDAK PERNAH menyentuh TEST, sesuai disiplin project ini.

PRASYARAT: audit out-of-fold utk --pool_fold yg dipilih HARUS SUDAH ADA
(dari birnet_audit_trainval_confident_learning.py --pool_fold N) -- audit
ini mahal (5 sub-model dilatih), jadi TIDAK dipicu otomatis di sini, cuma
dicek keberadaannya.

Output: BIRNet_Research/outputs/metadata_imbalance/pool{N}/weak_subgroups.csv
-- resep OTOMATIS (class_name, field, value, n, error_rate, target_n) utk
birnet_augment_metadata_targeted.py.

Usage:
    python birnet_diagnose_metadata_imbalance.py --pool_fold 0
"""
import argparse
import os
import pandas as pd

from ame_derm.common import birnet_final as bf

CLASS_NAMES = ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC")

# Kolom yg BUKAN metadata (identitas/path/label) -- selain ini, SEMUA
# kolom pada dataframe HAM10000 diuji otomatis sbg kandidat field.
NON_METADATA_COLS = {"lesion_id", "image_id", "dx", "dx_type", "label", "img_path", "mask_path"}


def detect_metadata_fields(ham_df, n_bins_numeric):
    """Deteksi kolom metadata otomatis dari dataframe -- TIDAK hardcode
    nama field spesifik. Kolom numerik di-bucket (quantile-based) supaya
    bisa diperlakukan sama dgn kolom kategorikal; kolom kategorikal
    dipakai apa adanya (NaN -> 'unknown')."""
    candidate_cols = [c for c in ham_df.columns if c not in NON_METADATA_COLS]
    bucketed = pd.DataFrame(index=ham_df.index)
    field_names = []
    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(ham_df[col]):
            field_name = f"{col}_bucket"
            try:
                bucketed[field_name] = pd.qcut(ham_df[col], q=n_bins_numeric, duplicates="drop").astype(str)
            except ValueError:
                continue  # kolom numerik tapi tidak cukup variasi utk di-bucket
            bucketed[field_name] = bucketed[field_name].where(ham_df[col].notna(), "unknown")
        else:
            field_name = col
            bucketed[field_name] = ham_df[col].fillna("unknown").astype(str)
        field_names.append(field_name)
    return bucketed, field_names


def find_weak_subgroups(predictions_df, field_names, class_names=CLASS_NAMES,
                         min_n_per_cell=8, min_cells_for_corr=3, corr_threshold=-0.4,
                         max_oversample_ratio=10.0):
    """Inti algoritma diagnosis -- DIPISAH dari main() supaya bisa dipakai
    ULANG oleh training adaptif (train_classifier_phase2_metadata_adaptive.py),
    bukan cuma dari audit BEKU sekali jalan. `predictions_df` butuh kolom
    'true_name' (nama kelas asli) + 'is_wrong' (bool) + semua kolom di
    field_names -- bisa berasal dari audit out-of-fold statis ATAU dari
    prediksi model yg SEDANG berjalan (evaluasi live tiap stage), sehingga
    diagnosis subgrup lemah bisa berubah/adaptif seiring training,
    persis ide user: keputusan brp banyak sintetis dibutuhkan didasarkan
    pada kondisi error/benar TERKINI, bukan angka statis di awal."""
    rows = []
    for class_name in class_names:
        sub = predictions_df[predictions_df["true_name"] == class_name]
        if len(sub) < min_n_per_cell:
            continue
        overall_class_error = sub["is_wrong"].mean()

        for field in field_names:
            grouped = sub.groupby(field)["is_wrong"].agg(["mean", "size"]).rename(
                columns={"mean": "error_rate", "size": "n"})
            grouped = grouped[grouped.index.astype(str) != "unknown"]
            grouped = grouped[grouped["n"] >= min_n_per_cell]
            if len(grouped) < min_cells_for_corr:
                continue
            corr = grouped[["n", "error_rate"]].corr().iloc[0, 1]
            if pd.isna(corr) or corr > corr_threshold:
                continue

            majority_n = grouped["n"].max()
            median_n = grouped["n"].median()
            for value, row in grouped.iterrows():
                is_sparse = row["n"] < median_n
                is_high_error = row["error_rate"] > overall_class_error
                if is_sparse and is_high_error:
                    target_n = min(majority_n, int(row["n"] * (1 + max_oversample_ratio)))
                    rows.append(dict(
                        class_name=class_name, label=class_names.index(class_name),
                        field=field, value=value, n=int(row["n"]), error_rate=round(row["error_rate"], 4),
                        class_overall_error=round(overall_class_error, 4),
                        field_correlation=round(corr, 4), majority_n=int(majority_n), target_n=int(target_n),
                    ))
    weak_df = pd.DataFrame(rows)
    if len(weak_df):
        weak_df = weak_df.sort_values(["field_correlation", "error_rate"])
    return weak_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool_fold", type=int, default=0)
    parser.add_argument("--min_n_per_cell", type=int, default=8,
                         help="Sel (kelas,field,value) diabaikan kalau n di bawah ini -- terlalu sedikit utk dipercaya.")
    parser.add_argument("--min_cells_for_corr", type=int, default=3,
                         help="Minimal berapa nilai berbeda dlm satu field spy korelasi n-vs-error dianggap valid.")
    parser.add_argument("--corr_threshold", type=float, default=-0.4,
                         help="Ambang korelasi (n vs error_rate) utk anggap field ini sumbu ketidakseimbangan nyata.")
    parser.add_argument("--n_bins_numeric", type=int, default=9,
                         help="Jumlah bucket utk kolom metadata numerik (mis. usia) -- otomatis, bukan hardcode rentang.")
    parser.add_argument("--max_oversample_ratio", type=float, default=10.0,
                         help="Target balance ke subgrup MAYORITAS (bukan median) dlm field yg sama, TAPI needed "
                              "dibatasi maks sekian x dari n asli -- cegah oversampling ekstrem (puluhan x) dari "
                              "basis sumber yg sangat kecil.")
    parser.add_argument("--audit_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    audit_dir = args.audit_dir or f"BIRNet_Research/outputs/audit_trainval/pool{args.pool_fold}"
    audit_csv = os.path.join(audit_dir, "audit_master_predictions.csv")
    if not os.path.exists(audit_csv):
        print(f"ERROR: {audit_csv} belum ada.")
        print(f"Jalankan dulu: python birnet_audit_trainval_confident_learning.py --pool_fold {args.pool_fold}")
        return

    out_dir = args.out_dir or f"BIRNet_Research/outputs/metadata_imbalance/pool{args.pool_fold}"
    os.makedirs(out_dir, exist_ok=True)

    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    ham_df = bf.build_ham10000_dataframe(paths)

    bucketed, field_names = detect_metadata_fields(ham_df, args.n_bins_numeric)
    ham_meta = pd.concat([ham_df[["image_id"]], bucketed], axis=1)
    print(f"Field metadata terdeteksi otomatis ({len(field_names)}): {field_names}")

    audit = pd.read_csv(audit_csv)
    audit = audit.merge(ham_meta, on="image_id", how="left")
    print(f"Audit out-of-fold pool_fold={args.pool_fold}: {len(audit)} sampel, "
          f"{audit['is_wrong'].sum()} salah ({audit['is_wrong'].mean():.4f})")

    weak_df = find_weak_subgroups(audit, field_names, CLASS_NAMES,
                                   args.min_n_per_cell, args.min_cells_for_corr,
                                   args.corr_threshold, args.max_oversample_ratio)
    out_csv = os.path.join(out_dir, "weak_subgroups.csv")
    weak_df.to_csv(out_csv, index=False)
    print(f"\n{'='*70}\nSubgrup LEMAH ditemukan (representasi sedikit DAN error tinggi, "
          f"pada field dgn korelasi <= {args.corr_threshold}): {len(weak_df)}\n{'='*70}")
    if len(weak_df):
        print(weak_df.to_string(index=False))
    print(f"\nDisimpan ke: {out_csv}")
    print(f"RESULT_SUMMARY model=diagnose_metadata_imbalance pool_fold={args.pool_fold} n_weak_subgroups={len(weak_df)}")


if __name__ == "__main__":
    main()
