"""
Automatic diagnosis of METADATA representation imbalance (not just class)
inside TRAIN-VAL -- it generalizes the earlier manual finding (MEL x age
n-vs-error correlation = -0.814, NV x location = -0.580), YET which metadata
field gets tested is NOT hardcoded to age/sex/location -- ANY field present
in the dataset metadata (auto-detected from the available columns, minus the
identity/path/label columns) is tested too, and so is EVERY class. Should a
later fold turn out to have a different (class, field) combination serving
as the imbalance axis, this script finds it on its own rather than being
steered specifically toward MEL/NV/age/location.

PRINCIPLE (user idea, human body-temperature regression analogy): within a
SINGLE class, subgroups (e.g. class X at a particular value of field Y) that
are RARELY represented in training will be "pulled" toward their own class
majority pattern when predicting -- gauged through the NEGATIVE correlation
between n (representation) and error_rate (from the out-of-fold audit) across
all values of a metadata field, WITHIN one class. If the correlation is
strongly negative for a given (class, field), that field genuinely acts as a
hidden imbalance axis for that class -- not merely noise.

AUTOMATIC FIELD DETECTION: candidate fields = every column in the dataset
metadata dataframe APART FROM identity/path/label (e.g. for HAM10000: age,
sex, localization, dataset -- but any further metadata column added later is
tested automatically without a code change). Numeric columns (like age) are
bucketed automatically (quantile-based, bin count from --n_bins_numeric);
categorical columns are taken as they are.

LEAKAGE SAFETY: uses ONLY the TRAIN-VAL out-of-fold audit (the chosen
pool_fold) -- NEVER touches TEST, in keeping with this project's discipline.

PREREQUISITE: the out-of-fold audit for the chosen --pool_fold MUST ALREADY
EXIST (from birnet_audit_trainval_confident_learning.py --pool_fold N) --
that audit is costly (5 sub-models trained), so it is NOT triggered
automatically here, only checked for existence.

Output: BIRNet_Research/outputs/metadata_imbalance/pool{N}/weak_subgroups.csv
-- an AUTOMATIC recipe (class_name, field, value, n, error_rate, target_n) for
birnet_augment_metadata_targeted.py.

Usage:
    python birnet_diagnose_metadata_imbalance.py --pool_fold 0
"""
import argparse
import os
import pandas as pd

from ame_derm.common import birnet_final as bf

CLASS_NAMES = ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC")

# Columns that are NOT metadata (identity/path/label) -- beyond these, ALL
# columns in the HAM10000 dataframe are auto-tested as candidate fields.
NON_METADATA_COLS = {"lesion_id", "image_id", "dx", "dx_type", "label", "img_path", "mask_path"}


def detect_metadata_fields(ham_df, n_bins_numeric):
    """Auto-detect metadata columns from the dataframe -- specific field
    names are NOT hardcoded. Numeric columns get bucketed (quantile-based)
    so they can be handled like categorical ones; categorical columns are
    taken as they are (NaN -> 'unknown')."""
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
    """Core of the diagnosis algorithm -- kept SEPARATE from main() so that
    adaptive training (train_classifier_phase2_metadata_adaptive.py) can
    REUSE it, instead of it being tied to a one-shot FROZEN audit.
    `predictions_df` requires a 'true_name' column (the original class name)
    plus 'is_wrong' (bool) and every column listed in field_names -- these
    may come from a static out-of-fold audit OR from a model that is STILL
    training (live per-stage evaluation), so weak-subgroup diagnosis can
    shift/adapt as training proceeds, exactly the user's idea: how many
    synthetics are needed is decided from the CURRENT error/correct state,
    not from numbers fixed at the start."""
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
