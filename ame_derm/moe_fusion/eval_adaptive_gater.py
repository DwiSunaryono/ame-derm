"""Adaptive Dual-Generalist Gating untuk BirNET Asymmetric MoE.

Masalah yang diselesaikan:
  - Static alpha (0.65-0.80) berlaku sama untuk SEMUA gambar.
  - Di Fold 4, DINOv2 Large lemah (f1=0.8096 vs Base 0.8317), menarik ensemble turun.

Solusi: AdaptiveDualGater — MLP kecil yang belajar dari VAL:
  "Kapan percaya Large Gen? Kapan percaya Base Gen? Berapa besar Specialist?"

Arsitektur (per-gambar):
  Input:  [prob_spec(7) + prob_large(7) + prob_base(7) + meta(15) + abcd(8)] = 44 dim
  Output: [α_spec, α_large, α_base] via Softmax — secara alami sum-to-1
  Merge:  final = α_spec*P_spec + α_large*P_large + α_base*P_base

Training:
  - Dilatih pada VAL set saja (no leakage ke TEST)
  - Loss: CrossEntropy terhadap final merged prob
  - Ensemble 5 gater (seed berbeda) untuk stabilitas

Usage:
  .venv/bin/python3 birnet_eval_adaptive_gater.py --fold 0
  .venv/bin/python3 birnet_eval_adaptive_gater.py --fold 4
  .venv/bin/python3 birnet_eval_adaptive_gater.py --fold 0 --no_abcd
  .venv/bin/python3 birnet_eval_adaptive_gater.py --all_folds
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report

from ame_derm.common import birnet_final as bf
from ame_derm.experts.train_specialist_expert import (
    MultimodalClassifier, EXPERT_MAPPING,
    set_active_meta_fields, make_loader, detect_metadata_fields,
)
from ame_derm.experts.train_classifier_phase2_final import CROP_DIR
from ame_derm.data_splits.representative_val_split import stratified_ham_kfold_fully_representative
from ame_derm.moe_fusion.eval_moe_pure_champion import extract_probs, get_expert_name, tta_views

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE  = "vit_base_patch14_reg4_dinov2.lvd142m"
LARGE = "vit_large_patch14_reg4_dinov2"
ABCD_CSV = "BIRNet_Research/outputs/abcd_features.csv"
LAB_NAMES = {0:"mel", 1:"nv", 2:"bcc", 3:"akiec", 4:"bkl", 5:"df", 6:"vasc"}
NUM_CLASSES = 7
META_DIM    = 15
ABCD_DIM    = 8


# ─────────────────────────────────────────────────────────────────────────────
# 1.  AdaptiveDualGater
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveDualGater(nn.Module):
    """MLP yang memprediksi (α_spec, α_large, α_base) per gambar.

    Input concatenation:
      prob_spec  (7) + prob_large (7) + prob_base (7) + meta (15) + abcd (8) = 44
    Output:
      3 weight logits → Softmax → [α_spec, α_large, α_base]
    """

    def __init__(self, use_abcd: bool = True):
        super().__init__()
        abcd_in = ABCD_DIM if use_abcd else 0
        in_dim  = NUM_CLASSES * 3 + META_DIM + abcd_in  # 21+15+8=44 or 36

        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3),   # 3 komponen: spec, large, base
        )

    def forward(self, p_spec, p_large, p_base, meta, abcd=None):
        parts = [p_spec, p_large, p_base, meta]
        if abcd is not None:
            parts.append(abcd)
        x = torch.cat(parts, dim=1)
        logits = self.net(x)
        return F.softmax(logits, dim=1)   # (B, 3) — sum=1


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_ckpt_paths(backbone, fold):
    ablation = "BIRNet_Research/outputs/checkpoints/phase2_multimodal_ablation"
    expert   = "BIRNet_Research/outputs/checkpoints/phase2_multimodal"
    return {
        "gen":   f"{ablation}/cls_{backbone}_multimodal_fold{fold}_best.pth",
        "torso": f"{expert}/cls_{backbone}_expert_torso_fold{fold}_best.pth",
        "limb":  f"{expert}/cls_{backbone}_expert_limb_fold{fold}_best.pth",
        "head":  f"{expert}/cls_{backbone}_expert_head_fold{fold}_active_noise_best.pth",
    }


def build_large_gen_ckpt(fold):
    ablation = "BIRNet_Research/outputs/checkpoints/phase2_multimodal_ablation"
    candidates = [
        f"{ablation}/cls_{LARGE}_multimodal_fold{fold}_large_best.pth",
        f"{ablation}/cls_{LARGE}_multimodal_fold{fold}_best.pth",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def load_model(backbone, ckpt_path, cfg, meta_dim=META_DIM):
    m = MultimodalClassifier(backbone, num_classes=cfg.num_classes,
                             meta_dim=meta_dim, img_size=cfg.img_size).to(DEVICE)
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ck["model_state"], strict=True)
    m.eval()
    return m


def build_specialist_probs(probs_dict, df):
    """Pilih prob spesialis yang sesuai lokalisasi per gambar."""
    exp_arr = []
    for i, (_, row) in enumerate(df.iterrows()):
        zone = get_expert_name(row["localization"])
        if zone in ("torso", "limb"):
            exp_arr.append(probs_dict[zone][i])
        elif zone == "head":
            exp_arr.append(probs_dict["head"][i])
        else:
            exp_arr.append(probs_dict["gen_base"][i])
    return np.array(exp_arr)


def load_abcd(df):
    """Muat fitur ABCD untuk gambar di df; kembalikan array (N, 8) atau None."""
    if not os.path.exists(ABCD_CSV):
        return None
    abcd_df = pd.read_csv(ABCD_CSV, index_col="image_id")
    feat_cols = ["asymmetry_h","asymmetry_v","border_irreg","border_jagged",
                 "color_std_r","color_std_g","color_std_b","compactness"]
    out = []
    for img_id in df["image_id"]:
        if img_id in abcd_df.index:
            out.append(abcd_df.loc[img_id, feat_cols].values.astype(np.float32))
        else:
            out.append(np.zeros(ABCD_DIM, dtype=np.float32))
    arr = np.array(out)
    # Normalise kolom
    mu  = arr.mean(axis=0) + 1e-8
    std = arr.std(axis=0) + 1e-8
    return (arr - mu) / std


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Training & Evaluasi Gater
# ─────────────────────────────────────────────────────────────────────────────

def train_gater_ensemble(
    p_spec, p_large, p_base, metas, labels,
    abcd=None, n_gaters=5, epochs=200, patience=30
):
    """Train ensemble gater pada VAL set. Kembalikan list of trained gaters."""
    use_abcd = abcd is not None
    gaters   = []

    # Konversi ke tensor
    T = lambda x: torch.tensor(x, dtype=torch.float32)
    Tp_spec  = T(p_spec);   Tp_large = T(p_large)
    Tp_base  = T(p_base);   Tmeta    = T(metas)
    Tlabel   = torch.tensor(labels, dtype=torch.long)
    Tabcd    = T(abcd) if use_abcd else None

    valid  = (Tlabel >= 0)
    Tp_spec  = Tp_spec[valid];  Tp_large = Tp_large[valid]
    Tp_base  = Tp_base[valid];  Tmeta    = Tmeta[valid]
    Tlabel   = Tlabel[valid]
    if Tabcd is not None:
        Tabcd = Tabcd[valid]

    parts = [Tp_spec, Tp_large, Tp_base, Tmeta]
    if Tabcd is not None:
        parts.append(Tabcd)
    dataset = torch.utils.data.TensorDataset(*parts, Tlabel)

    for seed in range(n_gaters):
        torch.manual_seed(seed * 17 + 42)
        gater = AdaptiveDualGater(use_abcd=use_abcd).to(DEVICE)
        opt   = torch.optim.Adam(gater.parameters(), lr=5e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                             shuffle=True, drop_last=False)

        best_loss = float("inf")
        no_improve = 0

        for ep in range(epochs):
            gater.train()
            epoch_loss = 0.0
            for batch in loader:
                *feat_parts, lbl = [b.to(DEVICE) for b in batch]
                ps, pl, pb, mt = feat_parts[0], feat_parts[1], feat_parts[2], feat_parts[3]
                ab = feat_parts[4] if use_abcd else None

                opt.zero_grad()
                weights = gater(ps, pl, pb, mt, ab)  # (B, 3)
                # Merged probability
                merged = (weights[:,0:1]*ps + weights[:,1:2]*pl + weights[:,2:3]*pb)
                loss   = F.cross_entropy(torch.log(merged.clamp(1e-8)), lbl)
                loss.backward()
                nn.utils.clip_grad_norm_(gater.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item()

            sched.step()
            if epoch_loss < best_loss:
                best_loss  = epoch_loss
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

        gater.eval()
        gaters.append(gater)
        print(f"  Gater {seed+1}/{n_gaters} selesai (best_loss={best_loss:.4f}, ep={ep+1})")

    return gaters


def eval_with_gater(p_spec, p_large, p_base, metas, labels,
                    gaters, abcd=None):
    """Inferensi adaptif menggunakan ensemble gater. Kembalikan Macro-F1."""
    use_abcd = (abcd is not None) and (gaters[0].net[0].in_features > 36)
    all_merged = []

    with torch.no_grad():
        for i in range(len(labels)):
            ps = torch.tensor(p_spec[i],  dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pl = torch.tensor(p_large[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pb = torch.tensor(p_base[i],  dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mt = torch.tensor(metas[i],   dtype=torch.float32).unsqueeze(0).to(DEVICE)
            ab = torch.tensor(abcd[i], dtype=torch.float32).unsqueeze(0).to(DEVICE) \
                 if (use_abcd and abcd is not None) else None

            # Rata-rata alpha dari semua gater
            alpha_sum = torch.zeros(1, 3, device=DEVICE)
            for g in gaters:
                alpha_sum += g(ps, pl, pb, mt, ab)
            alpha_avg = alpha_sum / len(gaters)

            merged = (alpha_avg[:,0]*ps + alpha_avg[:,1]*pl + alpha_avg[:,2]*pb)
            all_merged.append(merged.squeeze(0).cpu().numpy())

    all_merged = np.array(all_merged)
    preds      = np.argmax(all_merged, axis=1)
    valid      = labels >= 0
    macro      = f1_score(labels[valid], preds[valid], average="macro", zero_division=0)
    return macro, preds, all_merged


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main
# ─────────────────────────────────────────────────────────────────────────────

def run_fold(fold, cfg, ham_df, use_abcd=True):
    print(f"\n{'='*70}")
    print(f"[ADAPTIVE DUAL GATER] fold={fold}")
    print(f"{'='*70}")

    from ame_derm.experts.train_specialist_expert import detect_metadata_fields
    bucketed, _ = detect_metadata_fields(ham_df, 9)
    ov = [c for c in bucketed.columns if c in ham_df.columns]
    hm = pd.concat([ham_df.drop(columns=ov), bucketed], axis=1)

    val_df = test_df = None
    for _, fi, _, va, te in stratified_ham_kfold_fully_representative(
            hm, cfg, ["dataset"], n_splits=5, n_repeats=1):
        if fi == fold:
            val_df  = ham_df.iloc[va].reset_index(drop=True)
            test_df = ham_df.iloc[te].reset_index(drop=True)
            break

    _, val_aug = bf.build_augmentations(cfg.img_size, seed=42)
    val_ld  = make_loader(val_df,  cfg, val_aug, shuffle=False)
    test_ld = make_loader(test_df, cfg, val_aug, shuffle=False)

    # ── Load model ──────────────────────────────────────────────────────────
    ckpts_base = build_ckpt_paths(BASE, fold)
    large_ckpt = build_large_gen_ckpt(fold)

    if not os.path.exists(large_ckpt):
        print(f"  ❌ Large Generalist tidak ditemukan: {large_ckpt}")
        return None

    print("Memuat model...")
    models = {}
    for k, pth in ckpts_base.items():
        if os.path.exists(pth):
            key_name = "gen_base" if k == "gen" else k
            models[key_name] = load_model(BASE, pth, cfg)
            print(f"  ✅ base/{k}")
        else:
            print(f"  ❌ base/{k} tidak ditemukan: {pth}")
            return None

    models["large_gen"] = load_model(LARGE, large_ckpt, cfg)
    print(f"  ✅ large/gen ({os.path.basename(large_ckpt)})")

    # ── Ekstraksi probabilitas ───────────────────────────────────────────────
    print("\nEkstraksi VAL...")
    val_probs, y_val, val_metas = extract_probs(models, val_ld, cfg)
    print("Ekstraksi TEST...")
    test_probs, y_test, test_metas = extract_probs(models, test_ld, cfg)

    # Build arrays numpy
    vp_base  = np.array(val_probs["gen_base"])
    vp_large = np.array(val_probs["large_gen"])
    vp_spec  = build_specialist_probs(val_probs, val_df)

    tp_base  = np.array(test_probs["gen_base"])
    tp_large = np.array(test_probs["large_gen"])
    tp_spec  = build_specialist_probs(test_probs, test_df)

    # ── ABCD features ────────────────────────────────────────────────────────
    val_abcd  = None
    test_abcd = None
    if use_abcd:
        val_abcd  = load_abcd(val_df)
        test_abcd = load_abcd(test_df)
        if val_abcd is None:
            print("  ⚠️  ABCD CSV tidak ditemukan — training tanpa ABCD")
            use_abcd = False
        else:
            print(f"  ✅ ABCD features dimuat ({val_abcd.shape[1]} dim)")

    # ── Baseline & Static Hybrid (referensi) ────────────────────────────────
    def static_eval(gen_probs, spec_probs, labels, alpha=0.80):
        merged = alpha * spec_probs + (1 - alpha) * gen_probs
        preds  = np.argmax(merged, axis=1)
        valid  = labels >= 0
        return f1_score(labels[valid], preds[valid], average="macro", zero_division=0)

    # Cari α terbaik untuk Large dari VAL
    best_alpha, best_val = 0.65, -1.0
    for alpha in np.arange(0.60, 0.96, 0.05):
        m = static_eval(vp_large, vp_spec, y_val, alpha)
        if m > best_val:
            best_val, best_alpha = m, alpha

    f1_base   = static_eval(tp_base,  tp_spec, y_test, alpha=0.80)
    f1_hybrid = static_eval(tp_large, tp_spec, y_test, alpha=best_alpha)

    # ── Training Gater ───────────────────────────────────────────────────────
    print(f"\nTraining {5} Adaptive Dual Gaters (pada VAL set)...")
    gaters = train_gater_ensemble(
        vp_spec, vp_large, vp_base, val_metas, y_val,
        abcd=val_abcd, n_gaters=5, epochs=200, patience=30
    )

    # ── Evaluasi Adaptif pada TEST ───────────────────────────────────────────
    f1_gater, preds_gater, merged_gater = eval_with_gater(
        tp_spec, tp_large, tp_base, test_metas, y_test,
        gaters, abcd=test_abcd
    )

    # ── Ringkasan ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"HASIL FINAL — TEST (fold={fold})")
    print(f"{'='*70}")
    print(f"{'Konfigurasi':<45} {'macro':>7} {'mel':>7} {'akiec':>7}")
    print(f"{'-'*70}")

    def per_class(labels, preds):
        valid = labels >= 0
        pc    = f1_score(labels[valid], preds[valid], average=None,
                          labels=list(range(7)), zero_division=0)
        return pc

    # Baseline preds
    preds_base   = np.argmax(0.80*tp_spec + 0.20*tp_base,  axis=1)
    preds_hybrid = np.argmax(best_alpha*tp_spec + (1-best_alpha)*tp_large, axis=1)

    pc_base   = per_class(y_test, preds_base)
    pc_hybrid = per_class(y_test, preds_hybrid)
    pc_gater  = per_class(y_test, preds_gater)

    print(f"  Base spec + Base gen  α=0.80 [Champion]     "
          f"{f1_base:.4f}  {pc_base[0]:.4f}  {pc_base[3]:.4f}")
    print(f"  Base spec + Large gen α={best_alpha:.2f} [Static Hybrid] "
          f"{f1_hybrid:.4f}  {pc_hybrid[0]:.4f}  {pc_hybrid[3]:.4f}")
    print(f"  Adaptive Dual Gater  [PROPOSED]             "
          f"{f1_gater:.4f}  {pc_gater[0]:.4f}  {pc_gater[3]:.4f}")
    print(f"{'-'*70}")
    delta_vs_champ  = f1_gater - f1_base
    delta_vs_hybrid = f1_gater - f1_hybrid
    print(f"  Δ Gater vs Champion:      {delta_vs_champ:+.4f}")
    print(f"  Δ Gater vs Static Hybrid: {delta_vs_hybrid:+.4f}")

    # Classification report lengkap
    valid = y_test >= 0
    print(f"\nClassification Report — Adaptive Gater (fold={fold}):")
    print(classification_report(y_test[valid], preds_gater[valid],
                                target_names=list(LAB_NAMES.values()),
                                digits=4, zero_division=0))

    tag_abcd = "with_abcd" if use_abcd else "no_abcd"
    print(f"\nRESULT_SUMMARY variant=adaptive_dual_gater fold={fold} "
          f"f1_champion={f1_base:.4f} f1_hybrid={f1_hybrid:.4f} "
          f"f1_gater={f1_gater:.4f} "
          f"delta_champ={delta_vs_champ:+.4f} delta_hybrid={delta_vs_hybrid:+.4f} "
          f"mel_gater={pc_gater[0]:.4f} akiec_gater={pc_gater[3]:.4f} {tag_abcd}")

    return {
        "fold": fold,
        "f1_champion": f1_base,
        "f1_hybrid":   f1_hybrid,
        "f1_gater":    f1_gater,
        "delta_champ":  delta_vs_champ,
        "delta_hybrid": delta_vs_hybrid,
        "mel_champion": pc_base[0],
        "mel_hybrid":   pc_hybrid[0],
        "mel_gater":    pc_gater[0],
        "akiec_champion": pc_base[3],
        "akiec_hybrid":   pc_hybrid[3],
        "akiec_gater":    pc_gater[3],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--all_folds", action="store_true",
                    help="Jalankan semua fold 0-4 dan cetak ringkasan")
    ap.add_argument("--no_abcd", action="store_true",
                    help="Nonaktifkan fitur ABCD (fallback jika CSV belum ada)")
    args = ap.parse_args()

    set_active_meta_fields("localization")
    cfg   = bf.Config()
    paths = bf.DatasetPaths(cfg)
    ham_df = bf.build_ham10000_dataframe(paths)
    has   = ham_df["image_id"].apply(
                lambda i: os.path.exists(os.path.join(CROP_DIR, f"{i}.jpg")))
    ham_df = ham_df[has].reset_index(drop=True)

    use_abcd = not args.no_abcd

    if args.all_folds:
        results = []
        for fold in range(5):
            res = run_fold(fold, cfg, ham_df, use_abcd=use_abcd)
            if res:
                results.append(res)

        # ── Tabel ringkasan 5-fold ────────────────────────────────────────
        if results:
            print(f"\n{'='*80}")
            print("RINGKASAN 5-FOLD — ADAPTIVE DUAL GATER")
            print(f"{'='*80}")
            header = f"{'Fold':>4} | {'Champion':>9} | {'Static Hybrid':>13} | {'Adaptive Gater':>14} | {'Δ vs Champ':>10} | {'Δ vs Hybrid':>11}"
            print(header)
            print("-"*len(header))
            f1c_list, f1h_list, f1g_list = [], [], []
            for r in results:
                print(f"  {r['fold']:>2}  | {r['f1_champion']:>9.4f} | "
                      f"{r['f1_hybrid']:>13.4f} | {r['f1_gater']:>14.4f} | "
                      f"{r['delta_champ']:>+10.4f} | {r['delta_hybrid']:>+11.4f}")
                f1c_list.append(r["f1_champion"])
                f1h_list.append(r["f1_hybrid"])
                f1g_list.append(r["f1_gater"])

            print("-"*len(header))
            print(f"  Mean | {np.mean(f1c_list):>9.4f} | "
                  f"{np.mean(f1h_list):>13.4f} | {np.mean(f1g_list):>14.4f} | "
                  f"{np.mean(f1g_list)-np.mean(f1c_list):>+10.4f} | "
                  f"{np.mean(f1g_list)-np.mean(f1h_list):>+11.4f}")
            print(f"  Std  | {np.std(f1c_list):>9.4f} | "
                  f"{np.std(f1h_list):>13.4f} | {np.std(f1g_list):>14.4f} |")
    else:
        run_fold(args.fold, cfg, ham_df, use_abcd=use_abcd)


if __name__ == "__main__":
    main()
