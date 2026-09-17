#!/usr/bin/env python3
"""Pasok TERTARGET kelas langka untuk BCN train dari sumber dermoskopi eksternal.

Latar (percobaanReal2.md, Aug-15): plafon BCN tersandung kelangkaan data kelas
langka (akiec 147, df 25, vasc 22). Bukan pasrah — PASOK datanya. Sumber
dermoskopi (domain cocok BCN): MSKCC, HIBA, HAM10000. Cuma kelas target yang
di-pool, kelas umum dibiarkan (hindari domain-shift kelas mudah). PAD-UFES
SENGAJA dikecualikan (klinis/HP, beda domain).

Label canonical (EXT_TO_LABEL): 0=mel 1=nv 2=bcc 3=akiec 4=bkl 5=df 6=vasc.
Schema output IDENTIK build_bcn_df: {image_id,img_path,label,localization,age,sex,dataset,lesion_id}.
lesion_id diberi prefix sumber -> tak mungkin bentrok/leak dg lesion_id BCN.

Ketersediaan (diagnosis->label, dermoskopi):
  akiec(3): MSKCC 321 + HIBA 63 + HAM 327
  df(5)   : MSKCC 55  + HIBA 61 + HAM 115
  vasc(6) : MSKCC 63(angioma) + HIBA 51 + HAM 142
"""
import os
import numpy as np
import pandas as pd
from ame_derm.external_validation.external_zero_shot import EXT_TO_LABEL, EXT_LOC_MAP

import config

SRC = {
    "mskcc": ("panderm_datasets/MSKCC/MSKCC/mskcc.csv", "panderm_datasets/MSKCC/MSKCC/images"),
    "hiba":  ("panderm_datasets/HIBA/HIBA/hiba.csv",    "panderm_datasets/HIBA/HIBA/images"),
}
CROP_DIR = config.CROP_DIR


def _resolve(iid, raw_p, use_crop):
    """Gunakan crop Phase-1 (menyamakan framing BCN 'crop') kalau ada; jika tidak RAW."""
    if use_crop:
        cp = os.path.join(CROP_DIR, f"{iid}.jpg")
        if os.path.exists(cp):
            return cp, True
    return (raw_p, False) if os.path.exists(raw_p) else (None, False)


def _from_csv(csv, imgdir, src, label_ids, use_crop=True):
    df = pd.read_csv(csv)
    df = df[df["diagnosis"].isin(EXT_TO_LABEL)].copy()
    df["label"] = df["diagnosis"].map(EXT_TO_LABEL)
    df = df[df["label"].isin(label_ids)]
    df["sex"] = df["sex"].fillna("unknown")
    df["sex"] = df["sex"].where(df["sex"].isin(["male", "female"]), "unknown")
    rows = []
    n_crop = 0
    for _, r in df.iterrows():
        p, is_crop = _resolve(str(r["isic_id"]), os.path.join(imgdir, str(r["image"])), use_crop)
        if p is None:
            continue
        n_crop += int(is_crop)
        rows.append({
            "image_id": str(r["isic_id"]), "img_path": p, "label": int(r["label"]),
            "localization": EXT_LOC_MAP.get(r.get("anatom_site_general"), "unknown"),
            "age": r.get("age_approx", np.nan), "sex": r["sex"],
            "dataset": src, "lesion_id": f"{src}_{r['lesion_id']}",
        })
    print(f"  [{src}] {len(rows)} sampel, {n_crop} pakai crop Phase-1 (sisa RAW)")
    return rows


def _from_ham(label_ids, cfg, use_crop=True):
    """HAM kelas langka, label 0-6 canonical. Gunakan crop Phase-1 bila tersedia (match BCN 'crop')."""
    from ame_derm.common import birnet_final as bf
    paths = bf.DatasetPaths(cfg)
    ham = bf.build_ham10000_dataframe(paths)
    ham = ham[ham["label"].isin(label_ids)].copy()
    n_crop = 0
    paths_out = []
    for _, r in ham.iterrows():
        p, is_crop = _resolve(str(r["image_id"]), r["img_path"], use_crop)
        n_crop += int(is_crop)
        paths_out.append(p if p else r["img_path"])
    ham["img_path"] = paths_out
    ham["dataset"] = "ham"
    ham["lesion_id"] = "ham_" + ham["lesion_id"].astype(str)
    print(f"  [ham] {len(ham)} sampel, {n_crop} pakai crop Phase-1 (sisa RAW)")
    keep = ["image_id", "img_path", "label", "localization", "age", "sex", "dataset", "lesion_id"]
    return ham[keep]


def build_external_rare_df(label_ids, sources=("mskcc", "hiba", "ham"), cfg=None, use_crop=True):
    parts = []
    for s in sources:
        if s == "ham":
            parts.append(_from_ham(label_ids, cfg, use_crop))
        elif s in SRC:
            csv, imgdir = SRC[s]
            parts.append(pd.DataFrame(_from_csv(csv, imgdir, s, label_ids, use_crop)))
        else:
            raise ValueError(f"sumber tak dikenal: {s}")
    out = pd.concat([p for p in parts if len(p)], ignore_index=True)
    return out


if __name__ == "__main__":
    from ame_derm.common import birnet_final as bf
    df = build_external_rare_df([3, 5, 6], ("mskcc", "hiba", "ham"), cfg=bf.Config())
    print(f"total suplai eksternal (akiec/df/vasc) = {len(df)}")
    print(df.groupby(["dataset", "label"]).size())
