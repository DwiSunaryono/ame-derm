"""
Split TRAIN-VAL yang VAL-nya representatif thd metadata (bukan cuma
kelas) -- ide user: kalau VAL dibuat statistically dekat dgn TEST (across
kelas DAN metadata: usia/sex/lokasi/sumber-dataset), lalu DIKUNCI (tidak
pernah diubah selama training adaptif), maka menaikkan VAL akan lebih
genuinely menaikkan TEST -- krn VAL bukan sekadar sampel acak yg
kebetulan mewakili, tapi sengaja dikonstruksi mewakili.

BEDA dari bf.stratified_ham_kfold: split OUTER (mana yg jadi TEST) TETAP
IDENTIK (stratifikasi cuma pakai label, SAMA PERSIS spt sebelumnya) --
supaya TEST fold manapun TIDAK BERUBAH sama sekali (semua 20+ angka
historis di percobaan.md tetap bisa dibandingkan apple-to-apple). Yang
BERUBAH cuma split INNER (train vs val dari 80% pool): stratifikasi
sekarang pakai KOMBINASI (label + field metadata terpilih, default
'dataset' -- field dgn korelasi ketidakseimbangan terkuat & terbanyak dari
diagnosis Fase 8), bukan cuma label sendirian.

Konsistensi per-lesion: usia/sex/lokasi/dataset sudah diverifikasi HAMPIR
SELALU konsisten dalam 1 lesion_id yg sama (0-9 dari 7470 lesion py variasi
kecil, diabaikan) -- jadi aman dipakai sbg kunci stratifikasi per-lesion.

Usage (dipanggil dari script lain, bukan CLI):
    from ame_derm.data_splits.representative_val_split import stratified_ham_kfold_representative_val
    for fold_i, train_idx, val_idx, test_idx in stratified_ham_kfold_representative_val(
            ham_df, cfg, val_stratify_fields=["dataset"]):
        ...
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def stratified_ham_kfold_representative_val(df, cfg, val_stratify_fields=("dataset",), n_splits=5):
    idx = np.arange(len(df))
    labels = df["label"].values
    groups = df["lesion_id"].values

    combined_key = df["label"].astype(str)
    for f in val_stratify_fields:
        combined_key = combined_key + "_" + df[f].astype(str)
    combined_codes = pd.factorize(combined_key)[0]

    # Outer split (penentu TEST) TETAP pakai label SAJA -- identik dgn
    # bf.stratified_ham_kfold, supaya TEST fold manapun tidak berubah.
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    for fold_i, (train_val_idx, test_idx) in enumerate(sgkf.split(idx, labels, groups)):
        sub_groups = groups[train_val_idx]
        sub_combined = combined_codes[train_val_idx]
        sgkf_val = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=cfg.seed)
        train_sub_idx, val_sub_idx = next(
            sgkf_val.split(np.arange(len(sub_combined)), sub_combined, sub_groups)
        )
        train_idx = train_val_idx[train_sub_idx]
        val_idx = train_val_idx[val_sub_idx]
        assert set(groups[train_idx]).isdisjoint(set(groups[val_idx])), "leakage lesion train/val!"
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), "leakage lesion train/test!"
        assert set(groups[val_idx]).isdisjoint(set(groups[test_idx])), "leakage lesion val/test!"
        yield fold_i, train_idx, val_idx, test_idx


def stratified_ham_kfold_fully_representative(df, cfg, stratify_fields=("dataset",), n_splits=5, n_repeats=1, seed_offset=0):
    """FONDASI EVALUASI BARU (beda dari stratified_ham_kfold_representative_val
    di atas): outer split (penentu TEST) DAN inner split (penentu VAL)
    SAMA-SAMA distratifikasi label+stratify_fields -- TEST & VAL jadi
    genuinely representatif & mirip satu sama lain, bukan cuma VAL
    "dipaksa" representatif sementara TEST asal.

    INI MENGUBAH IDENTITAS TEST (beda gambar dari bf.stratified_ham_kfold
    ataupun stratified_ham_kfold_representative_val) -- SENGAJA, sesuai
    keputusan eksplisit user stlh temuan VAL bisa berayun 0.78->0.95 murni
    dari sampling-variance lesi individual (bukan ketidakseimbangan
    metadata) -- percobaan.md (TEST lama) jadi arsip Fase 1-8, evaluasi
    baru & seterusnya pakai fondasi split ini, dicatat terpisah.

    n_repeats>1: gaya RepeatedStratifiedGroupKFold -- ulangi SELURUH proses
    split dgn random_state berbeda tiap repeat (seed=cfg.seed+seed_offset+
    repeat_i), utk meredam variance dari KEBETULAN pengelompokan lesi (baru
    terbukti jadi sumber varians dominan, lebih besar dari ketidak-
    seimbangan metadata semata). Default n_repeats=1 (hemat, spt sblmnya)
    -- naikkan HANYA kalau butuh estimasi lebih stabil (biaya training
    berkali lipat, n_repeats x n_splits total run).

    Yields: (repeat_i, fold_i, train_idx, val_idx, test_idx)
    """
    idx = np.arange(len(df))
    groups = df["lesion_id"].values

    combined_key = df["label"].astype(str)
    for f in stratify_fields:
        combined_key = combined_key + "_" + df[f].astype(str)
    combined_codes = pd.factorize(combined_key)[0]

    for repeat_i in range(n_repeats):
        seed = cfg.seed + seed_offset + repeat_i
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_i, (train_val_idx, test_idx) in enumerate(sgkf.split(idx, combined_codes, groups)):
            sub_groups = groups[train_val_idx]
            sub_combined = combined_codes[train_val_idx]
            sgkf_val = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
            train_sub_idx, val_sub_idx = next(
                sgkf_val.split(np.arange(len(sub_combined)), sub_combined, sub_groups)
            )
            train_idx = train_val_idx[train_sub_idx]
            val_idx = train_val_idx[val_sub_idx]
            assert set(groups[train_idx]).isdisjoint(set(groups[val_idx])), "leakage lesion train/val!"
            assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), "leakage lesion train/test!"
            assert set(groups[val_idx]).isdisjoint(set(groups[test_idx])), "leakage lesion val/test!"
            yield repeat_i, fold_i, train_idx, val_idx, test_idx


def compare_val_to_reference(val_df, reference_df, fields, label_col="label"):
    """Cek statistik: seberapa dekat distribusi VAL ke distribusi acuan
    (biasanya TEST atau pool) di kelas + tiap field metadata. Return dict
    berisi jarak (mean abs diff proporsi) per field -- makin kecil makin
    representatif."""
    results = {}
    for f in [label_col] + list(fields):
        val_dist = val_df[f].value_counts(normalize=True)
        ref_dist = reference_df[f].value_counts(normalize=True)
        all_vals = set(val_dist.index) | set(ref_dist.index)
        diffs = [abs(val_dist.get(v, 0) - ref_dist.get(v, 0)) for v in all_vals]
        results[f] = {"mean_abs_diff": float(np.mean(diffs)), "max_abs_diff": float(np.max(diffs))}
    return results


if __name__ == "__main__":
    from ame_derm.common import birnet_final as bf
    from ame_derm.data_splits.diagnose_metadata_imbalance import detect_metadata_fields

    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    ham_df = bf.build_ham10000_dataframe(paths)

    bucketed, field_names = detect_metadata_fields(ham_df, n_bins_numeric=9)
    overlap = [c for c in bucketed.columns if c in ham_df.columns]
    ham_meta = pd.concat([ham_df.drop(columns=overlap), bucketed], axis=1)

    old_test_idx, old_val_idx = None, None
    print("=== SPLIT LAMA (bf.stratified_ham_kfold, VAL stratify=label saja) ===")
    for fold_i, train_idx, val_idx, test_idx in bf.stratified_ham_kfold(ham_meta, cfg, n_splits=5):
        if fold_i != 0:
            continue
        old_test_idx, old_val_idx = set(test_idx), set(val_idx)
        val_df_old = ham_meta.iloc[val_idx]
        test_df = ham_meta.iloc[test_idx]
        print(f"  n_val={len(val_df_old)} n_test={len(test_df)}")
        cmp_old = compare_val_to_reference(val_df_old, test_df, field_names)
        for f, v in cmp_old.items():
            print(f"  {f}: mean_abs_diff={v['mean_abs_diff']:.4f} max_abs_diff={v['max_abs_diff']:.4f}")
        break

    print("\n=== SPLIT BARU (representative, VAL stratify=label+dataset) ===")
    for fold_i, train_idx, val_idx, test_idx in stratified_ham_kfold_representative_val(
            ham_meta, cfg, val_stratify_fields=["dataset"], n_splits=5):
        if fold_i != 0:
            continue
        print(f"  TEST identik dgn split lama? {set(test_idx) == old_test_idx}")
        print(f"  VAL identik dgn split lama? {set(val_idx) == old_val_idx} (HARUS beda -- itu tujuannya)")
        val_df_new = ham_meta.iloc[val_idx]
        test_df = ham_meta.iloc[test_idx]
        print(f"  n_val={len(val_df_new)} n_test={len(test_df)}")
        cmp_new = compare_val_to_reference(val_df_new, test_df, field_names)
        for f, v in cmp_new.items():
            print(f"  {f}: mean_abs_diff={v['mean_abs_diff']:.4f} max_abs_diff={v['max_abs_diff']:.4f}")
        break
