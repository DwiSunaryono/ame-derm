#!/usr/bin/env python3
"""
FASE C — Anatomical MoE (proposed) under PanDerm's OFFICIAL PROTOCOL.
Combines the Large generalist + 3 base experts per-zone (image-only, identical
architecture), routing by localization, with blend alpha tuned on VAL -> applied to TEST.

Clean ablation: single DINOv2-Large (gen only) vs MoE (gen + expert routing).
Question: does expert routing add on top of the single backbone (0.8253) and
beat PanDerm (0.8234) under the IDENTICAL protocol?

Requires checkpoints (from birnet_dinov2_official_protocol.py):
  - Large gen : results_dinov2_official/dinov2_{images}_seed{seed}_best.pth
  - Experts   : results_dinov2_official/model_{zone}_base_{images}_seed{seed}_best.pth

Example:
  .venv/bin/python birnet_moe_official_eval.py --images full --seed 123
"""
import argparse, os, json
import numpy as np, pandas as pd, torch
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from sklearn.metrics import f1_score, classification_report

import config
ROOT = config.AME_DERM_ROOT
IMG_DIRS = {
    "full": f"{ROOT}/PanDerm/data/HAM10000_clean/ISIC2018",
    "crop": f"{ROOT}/BIRNet_Research/outputs/cropped_phase1_final",
}
ZONE_MAP = {
    "torso": ["back", "trunk", "abdomen", "chest"],
    "limb": ["lower extremity", "upper extremity", "foot", "hand", "acral"],
    "head": ["face", "neck", "scalp", "ear"],
}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
LAB_NAMES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
BASE = "vit_base_patch14_reg4_dinov2.lvd142m"
LARGE = "vit_large_patch14_reg4_dinov2.lvd142m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_zone(loc):
    for z, locs in ZONE_MAP.items():
        if loc in locs:
            return z
    return "gen"


class DS(Dataset):
    def __init__(self, df, img_dir, tf):
        self.df = df.reset_index(drop=True); self.img_dir = img_dir; self.tf = tf
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(Image.open(os.path.join(self.img_dir, str(r["image"]))).convert("RGB")), int(r["label"])


@torch.no_grad()
def extract(model, loader, tta=True):
    model.eval(); out = []
    for x, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        views = [x, torch.flip(x, [3]), torch.flip(x, [2])] if tta else [x]
        p = sum(torch.softmax(model(v), 1) for v in views) / len(views)
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def load_model(backbone, ckpt, img_size):
    m = timm.create_model(backbone, pretrained=False, num_classes=7, img_size=img_size).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    m.eval(); return m


def route(exp_probs, gen_probs, df):
    """Per sample: use its zone's expert; when no zone applies -> gen (fallback)."""
    out = []
    for i, loc in enumerate(df["localization"].values):
        z = get_zone(loc)
        out.append(exp_probs[z][i] if z in exp_probs else gen_probs[i])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", choices=["full", "crop"], default="full")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--csv", default=f"{ROOT}/PanDerm/data/HAM10000_clean/ISIC2018_splits/HAM_clean_fixed.csv")
    ap.add_argument("--ckpt_dir", default=f"{ROOT}/results_dinov2_official")
    ap.add_argument("--alpha_start", type=float, default=0.50)
    ap.add_argument("--alpha_end", type=float, default=0.95)
    ap.add_argument("--alpha_step", type=float, default=0.05)
    args = ap.parse_args()

    img_dir = IMG_DIRS[args.images]
    df = pd.read_csv(args.csv)
    va = df[df.split == "val"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    tf = T.Compose([T.Resize((args.img_size, args.img_size), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize(MEAN, STD)])
    va_ld = DataLoader(DS(va, img_dir, tf), batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    te_ld = DataLoader(DS(te, img_dir, tf), batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # file paths
    gen_ckpt = os.path.join(args.ckpt_dir, f"dinov2_{args.images}_seed{args.seed}_best.pth")
    exp_ckpts = {z: os.path.join(args.ckpt_dir, f"model_{z}_base_{args.images}_seed{args.seed}_best.pth")
                 for z in ZONE_MAP}
    for name, p in [("gen", gen_ckpt)] + list(exp_ckpts.items()):
        if not os.path.exists(p):
            print(f"❌ checkpoint '{name}' TIDAK ADA: {p}\n   Latih dulu (lihat header script)."); return

    print(f"[MoE official] images={args.images} seed={args.seed} | val={len(va)} test={len(te)}")
    print("Loading models...");
    gen = load_model(LARGE, gen_ckpt, args.img_size)
    experts = {z: load_model(BASE, exp_ckpts[z], args.img_size) for z in ZONE_MAP}

    print("Extract probs (TTA)...")
    gen_va, gen_te = extract(gen, va_ld), extract(gen, te_ld)
    exp_va = {z: extract(experts[z], va_ld) for z in ZONE_MAP}
    exp_te = {z: extract(experts[z], te_ld) for z in ZONE_MAP}

    routed_va, routed_te = route(exp_va, gen_va, va), route(exp_te, gen_te, te)
    y_va, y_te = va["label"].values, te["label"].values

    # sweep alpha on VAL
    best_a, best_m = None, -1
    print("\nSWEEP alpha (VAL):")
    a = args.alpha_start
    while a <= args.alpha_end + 1e-9:
        m = f1_score(y_va, (a * routed_va + (1 - a) * gen_va).argmax(1), average="macro", zero_division=0)
        flag = ""
        if m > best_m: best_m, best_a, flag = m, a, " <-best"
        print(f"  a={a:.2f}  val_macro={m:.4f}{flag}")
        a += args.alpha_step

    # apply on TEST
    merged_te = best_a * routed_te + (1 - best_a) * gen_te
    preds = merged_te.argmax(1)
    moe_macro = f1_score(y_te, preds, average="macro", zero_division=0)
    gen_macro = f1_score(y_te, gen_te.argmax(1), average="macro", zero_division=0)
    from sklearn.metrics import balanced_accuracy_score, accuracy_score
    print("\n" + "=" * 60)
    print(f"HASIL TEST (alpha={best_a:.2f} dari VAL, seed={args.seed}, images={args.images})")
    print("=" * 60)
    print(f"  Single Large-gen (baseline)  macro-F1 = {gen_macro:.4f}")
    print(f"  Anatomical MoE   (proposed)  macro-F1 = {moe_macro:.4f}   Δ={moe_macro-gen_macro:+.4f}")
    print(f"  MoE: bacc={balanced_accuracy_score(y_te,preds):.4f} "
          f"wf1={f1_score(y_te,preds,average='weighted',zero_division=0):.4f} "
          f"acc={accuracy_score(y_te,preds):.4f}")
    print("-" * 60)
    print(classification_report(y_te, preds, target_names=LAB_NAMES, digits=4, zero_division=0))

    res = {"seed": args.seed, "images": args.images, "alpha": best_a,
           "gen_macro_f1": gen_macro, "moe_macro_f1": moe_macro,
           "delta": moe_macro - gen_macro,
           "moe_bacc": balanced_accuracy_score(y_te, preds),
           "moe_wf1": f1_score(y_te, preds, average="weighted", zero_division=0),
           "moe_acc": accuracy_score(y_te, preds)}
    with open(os.path.join(args.ckpt_dir, f"moe_result_{args.images}_seed{args.seed}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nRESULT_SUMMARY moe seed={args.seed} alpha={best_a:.2f} "
          f"gen={gen_macro:.4f} moe={moe_macro:.4f} delta={moe_macro-gen_macro:+.4f}")


if __name__ == "__main__":
    main()
