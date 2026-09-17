"""Heterogeneous MoE — evaluasi SWIN-TORSO swap (fold-0, RAW).

Menguji apakah mengganti Pakar Torso DINOv2 -> Pakar Torso SWIN memecah
"echo chamber" konsili dan menaikkan F1. Melaporkan, di fold-0:

  A. ENSEMBLE (Statis 80:20, TTA):
     - Baseline council (Torso DINOv2)      -> harus reproduksi 0.8235
     - Hetero  council (Torso SWIN)         -> pemenang?
  B. DIVERSITY di SUBSET TORSO (menjawab pertanyaan yg selama ini kosong: Q di RAW):
     - F1 zona: DINOv2-Torso vs SWIN-Torso vs Generalist
     - Q-statistic & disagreement: SWIN-Torso vs Generalist, DINOv2-Torso vs Generalist,
       SWIN-Torso vs DINOv2-Torso
  Kriteria lolos (pre-registered): swap membantu HANYA jika
     F1(Swin-Torso) >= F1(DINOv2-Torso) di zona torso  DAN  Q(Swin,gen) < Q(DINOv2,gen).

Semua model dievaluasi di crop RAW yg sama (cropped_phase1_final), TTA 6-view.
Fungsi extract_probs/eval_logic diimpor dari birnet_eval_moe_pure_champion (identik).

Usage:  .venv/bin/python3 birnet_eval_hetero_torso_fold0.py
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from ame_derm.common import birnet_final as bf
from ame_derm.experts.train_specialist_expert import MultimodalClassifier, set_active_meta_fields, make_loader, detect_metadata_fields
from ame_derm.experts.train_classifier_phase2_final import CROP_DIR
from ame_derm.data_splits.representative_val_split import stratified_ham_kfold_fully_representative
from ame_derm.moe_fusion.eval_moe_pure_champion import extract_probs, get_expert_name

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINO = "vit_base_patch14_reg4_dinov2.lvd142m"
SWIN = "swin_base_patch4_window7_224"
GEN_DIR = "BIRNet_Research/outputs/checkpoints/phase2_multimodal_ablation"
EXP_DIR = "BIRNet_Research/outputs/checkpoints/phase2_multimodal"
FOLD = 0

# key -> (backbone, checkpoint_path)
SPECS = {
    "gen_pure":   (DINO, f"{GEN_DIR}/cls_{DINO}_multimodal_fold{FOLD}_best.pth"),
    "dino_torso": (DINO, f"{EXP_DIR}/cls_{DINO}_expert_torso_fold{FOLD}_best.pth"),
    "limb":       (DINO, f"{EXP_DIR}/cls_{DINO}_expert_limb_fold{FOLD}_best.pth"),
    "head_v2":    (DINO, f"{EXP_DIR}/cls_{DINO}_expert_head_fold{FOLD}_active_noise_best.pth"),
    "swin_torso": (SWIN, f"{EXP_DIR}/cls_{SWIN}_expert_torso_fold{FOLD}_best.pth"),
}


def meta_dim_of(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model_state"]
    k = [x for x in sd if "meta_embed.0.weight" in x][0]
    return int(sd[k].shape[1])


def q_statistic(pred_a, pred_b, y):
    a, b = (pred_a == y), (pred_b == y)
    n11 = int((a & b).sum()); n00 = int((~a & ~b).sum())
    n01 = int((~a & b).sum()); n10 = int((a & ~b).sum())
    denom = n11 * n00 + n01 * n10
    q = (n11 * n00 - n01 * n10) / denom if denom > 0 else float("nan")
    disagree = (n01 + n10) / max(len(y), 1)
    return q, disagree


def council_f1(probs, test_df, labels, torso_key):
    """Statis 80:20: expert relevan per-zona + 20% generalist. torso_key pilih sumber Torso."""
    gen = probs["gen_pure"]
    preds = []
    for i, row in test_df.iterrows():
        zone = get_expert_name(row["localization"])
        if zone == "torso":
            exp = probs[torso_key][i]
        elif zone in ("limb",):
            exp = probs["limb"][i]
        elif zone == "head":
            exp = probs["head_v2"][i]
        else:
            exp = gen[i]  # zona tanpa pakar -> generalist saja
        merged = 0.8 * np.asarray(exp) + 0.2 * np.asarray(gen[i])
        preds.append(int(np.argmax(merged)))
    valid = labels >= 0
    return f1_score(labels[valid], np.array(preds)[valid], average="macro")


def main():
    # cek keberadaan checkpoint
    missing = [k for k, (_, p) in SPECS.items() if not os.path.exists(p)]
    if missing:
        print(f"Checkpoint belum ada: {missing}\nJalankan run_hetero_swin_torso_fold0.sh dulu.")
        return

    # cek meta_dim semua model harus 15 (council memberi meta 15-dim)
    dims = {k: meta_dim_of(p) for k, (_, p) in SPECS.items()}
    bad = {k: d for k, d in dims.items() if d != 15}
    if bad:
        print(f"MISMATCH meta_dim (harus 15): {bad}")
        print("-> Latih ulang Swin generalist dgn pipeline sekarang (meta_dim=15) sebelum eval.")
        return

    set_active_meta_fields("localization")
    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    ham_df = bf.build_ham10000_dataframe(paths)
    has = ham_df["image_id"].apply(lambda i: os.path.exists(os.path.join(CROP_DIR, f"{i}.jpg")))
    ham_df = ham_df[has].reset_index(drop=True)
    bucketed, _ = detect_metadata_fields(ham_df, n_bins_numeric=9)
    overlap = [c for c in bucketed.columns if c in ham_df.columns]
    ham_meta = pd.concat([ham_df.drop(columns=overlap), bucketed], axis=1)

    test_df = None
    for _, fi, _, _, tst in stratified_ham_kfold_fully_representative(ham_meta, cfg, ["dataset"], n_splits=5, n_repeats=1):
        if fi == FOLD:
            test_df = ham_df.iloc[tst].reset_index(drop=True)
            break

    # load semua model (backbone campur)
    models = {}
    for key, (bb, path) in SPECS.items():
        m = MultimodalClassifier(bb, num_classes=cfg.num_classes, meta_dim=15, img_size=cfg.img_size).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False)["model_state"], strict=True)
        m.eval()
        models[key] = m

    _, val_aug = bf.build_augmentations(cfg.img_size, seed=42)
    loader = make_loader(test_df, cfg, val_aug, shuffle=False)
    print("Ekstraksi probabilitas TTA (5 model, backbone campur DINOv2+Swin)...")
    probs, labels, _ = extract_probs(models, loader, cfg)
    labels = np.array(labels)

    # ---- A. Ensemble ----
    f1_base = council_f1(probs, test_df, labels, torso_key="dino_torso")
    f1_hetero = council_f1(probs, test_df, labels, torso_key="swin_torso")

    # ---- B. Diversity di subset TORSO ----
    torso_mask = test_df["localization"].apply(lambda l: get_expert_name(l) == "torso").values
    y_t = labels[torso_mask]
    def preds_of(key):
        return np.argmax(np.asarray(probs[key]), axis=1)[torso_mask]
    p_gen, p_dino, p_swin = preds_of("gen_pure"), preds_of("dino_torso"), preds_of("swin_torso")
    f1_gen_t = f1_score(y_t, p_gen, average="macro", zero_division=0)
    f1_dino_t = f1_score(y_t, p_dino, average="macro", zero_division=0)
    f1_swin_t = f1_score(y_t, p_swin, average="macro", zero_division=0)
    q_sg, d_sg = q_statistic(p_swin, p_gen, y_t)
    q_dg, d_dg = q_statistic(p_dino, p_gen, y_t)
    q_sd, d_sd = q_statistic(p_swin, p_dino, y_t)

    W = 64
    print("\n" + "=" * W)
    print("A. ENSEMBLE COUNCIL (Statis 80:20, TTA) — fold-0")
    print("=" * W)
    print(f"  Baseline (Torso DINOv2)  F1 = {f1_base:.4f}   (target reproduksi 0.8235)")
    print(f"  Hetero   (Torso SWIN)    F1 = {f1_hetero:.4f}")
    print(f"  Delta                       = {f1_hetero - f1_base:+.4f}")

    print("\n" + "=" * W)
    print(f"B. DIVERSITY @ SUBSET TORSO (n={int(torso_mask.sum())})  [Q rendah = lebih beragam]")
    print("=" * W)
    print(f"  F1 zona: Generalist={f1_gen_t:.4f} | DINOv2-Torso={f1_dino_t:.4f} | SWIN-Torso={f1_swin_t:.4f}")
    print(f"  Q(Swin , Gen)  = {q_sg:+.4f}  | disagreement = {d_sg:.4f}")
    print(f"  Q(DINOv2,Gen)  = {q_dg:+.4f}  | disagreement = {d_dg:.4f}")
    print(f"  Q(Swin , DINO) = {q_sd:+.4f}  | disagreement = {d_sd:.4f}")

    print("\n" + "=" * W)
    print("VERDICT (kriteria pre-registered)")
    print("=" * W)
    cond_f1 = f1_swin_t >= f1_dino_t
    cond_div = q_sg < q_dg
    print(f"  [{'OK ' if cond_f1 else 'GAGAL'}] F1(Swin-Torso) >= F1(DINOv2-Torso) : {f1_swin_t:.4f} vs {f1_dino_t:.4f}")
    print(f"  [{'OK ' if cond_div else 'GAGAL'}] Q(Swin,Gen) < Q(DINOv2,Gen)       : {q_sg:+.4f} vs {q_dg:+.4f}")
    if cond_f1 and cond_div:
        print("  => Kedua syarat lolos: swap Swin-Torso LAYAK dinaikkan ke 5-fold.")
    elif f1_hetero > f1_base:
        print("  => Syarat tak penuh TAPI ensemble tetap naik — periksa: mungkin gain kebetulan/noise 1 fold.")
    else:
        print("  => Tidak lolos: pakar Swin lebih lemah dan/atau tidak lebih beragam. Pertahankan DINOv2-Torso.")


if __name__ == "__main__":
    main()
