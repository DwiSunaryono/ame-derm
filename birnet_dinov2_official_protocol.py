#!/usr/bin/env python3
"""
FASE C — DINOv2 di PROTOKOL RESMI PanDerm (perbandingan backbone yang adil).

Data IDENTIK dengan PanDerm: split resmi tunggal (kolom 'split' train/val/test),
full-image ISIC2018, test-set 1232 yang sama, metrik macro-F1.
Resep meniru PanDerm sedekat mungkin: layer-decay 0.65, mixup 0.8/cutmix 1.0,
weighted sampler, label smoothing 0.1, warmup+cosine, 50 epoch, TTA saat test.

Backbone default = DINOv2-Large reg4 (sama dgn generalist champion).
--images full  -> PanDerm/data/HAM10000_clean/ISIC2018/  (jawab: efek full-image pd DINOv2)
--images crop  -> BIRNet_Research/outputs/cropped_phase1_final/

Pakai .venv (timm 1.0.28). Contoh:
  .venv/bin/python birnet_dinov2_official_protocol.py --images full --epochs 50
"""
import argparse, os, math, json
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from torchvision import transforms as T
from sklearn.metrics import (f1_score, balanced_accuracy_score, accuracy_score,
                             roc_auc_score)

ROOT = os.environ.get("AME_DERM_ROOT", os.path.dirname(os.path.abspath(__file__)))
IMG_DIRS = {
    "full": f"{ROOT}/PanDerm/data/HAM10000_clean/ISIC2018",
    "crop": f"{ROOT}/BIRNet_Research/outputs/cropped_phase1_final",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Mapping zona anatomis (dari EXPERT_MAPPING champion) untuk expert per-zona
ZONE_MAP = {
    "torso": ["back", "trunk", "abdomen", "chest"],
    "limb": ["lower extremity", "upper extremity", "foot", "hand", "acral"],
    "head": ["face", "neck", "scalp", "ear"],
}


class HAMDataset(Dataset):
    def __init__(self, df, img_dir, tf):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        p = os.path.join(self.img_dir, str(r["image"]))
        img = Image.open(p).convert("RGB")
        return self.tf(img), int(r["label"])


def build_transforms(img_size):
    train_tf = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def layer_decay_param_groups(model, base_lr, weight_decay, decay):
    """LR layer-wise decay ala PanDerm (backbone awal LR kecil)."""
    blocks = getattr(model, "blocks", [])
    n = len(blocks) + 1
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("blocks."):
            lid = int(name.split(".")[1]) + 1
        elif any(name.startswith(k) for k in ("patch_embed", "cls_token", "reg_token",
                                              "pos_embed", "mask_token")):
            lid = 0
        else:  # norm/head
            lid = n
        scale = decay ** (n - lid)
        wd = 0.0 if (p.ndim == 1 or name.endswith(".bias")) else weight_decay
        key = (lid, wd)
        if key not in groups:
            groups[key] = {"params": [], "lr": base_lr * scale, "weight_decay": wd}
        groups[key]["params"].append(p)
    return list(groups.values())


@torch.no_grad()
def evaluate(model, loader, device, tta=False):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        views = [x]
        if tta:
            views += [torch.flip(x, [3]), torch.flip(x, [2])]  # hflip, vflip
        p = torch.zeros(x.size(0), model.num_classes, device=device)
        for v in views:
            p += torch.softmax(model(v), 1)
        p /= len(views)
        probs.append(p.cpu().numpy()); ys.append(y.numpy())
    probs = np.concatenate(probs); ys = np.concatenate(ys)
    preds = probs.argmax(1)
    try:
        auroc = roc_auc_score(ys, probs, multi_class="ovr", average="macro")
    except Exception:
        auroc = float("nan")
    return {
        "macro_f1": f1_score(ys, preds, average="macro"),
        "weighted_f1": f1_score(ys, preds, average="weighted"),
        "bacc": balanced_accuracy_score(ys, preds),
        "acc": accuracy_score(ys, preds),
        "auroc": auroc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=f"{ROOT}/PanDerm/data/HAM10000_clean/ISIC2018_splits/HAM_clean_fixed.csv")
    ap.add_argument("--images", choices=["full", "crop"], default="full")
    ap.add_argument("--backbone", default="vit_large_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--layer_decay", type=float, default=0.65)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--drop_path", type=float, default=0.2)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--out", default=f"{ROOT}/results_dinov2_official")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zone", choices=["torso", "limb", "head"], default=None,
                    help="latih expert per-zona (filter train/val/test ke lokalisasi zona itu)")
    ap.add_argument("--smoke", action="store_true", help="1 batch cek pipeline lalu keluar")
    args = ap.parse_args()

    import random
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    img_dir = IMG_DIRS[args.images]
    df = pd.read_csv(args.csv)
    df = df[df["split"].isin(["train", "val", "test"])].copy()
    n_cls = int(df.label.max()) + 1  # 7 kelas HAM (dihitung dari FULL df sebelum filter zona)
    tr, va, te = (df[df.split == s].copy() for s in ["train", "val", "test"])
    if args.zone:
        locs = ZONE_MAP[args.zone]
        tr = tr[tr["localization"].isin(locs)]
        va = va[va["localization"].isin(locs)]
        te = te[te["localization"].isin(locs)]
        print(f"[EXPERT zona={args.zone}] localization in {locs}")
    print(f"images={args.images} zona={args.zone or 'GEN'} | train={len(tr)} val={len(va)} test={len(te)} | n_cls={n_cls}")

    train_tf, eval_tf = build_transforms(args.img_size)
    ds_tr = HAMDataset(tr, img_dir, train_tf)
    ds_va = HAMDataset(va, img_dir, eval_tf)
    ds_te = HAMDataset(te, img_dir, eval_tf)

    # weighted sampler (inverse class freq) — ala PanDerm --weights
    # robust utk subset zona (kelas bisa hilang): map label->1/count via value_counts
    vc = tr["label"].value_counts()
    sample_w = tr["label"].map(lambda l: 1.0 / vc[l]).values
    sampler = WeightedRandomSampler(torch.DoubleTensor(sample_w), len(sample_w), replacement=True)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler,
                       num_workers=8, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = timm.create_model(args.backbone, pretrained=True, num_classes=n_cls,
                              img_size=args.img_size, drop_path_rate=args.drop_path).to(device)
    model.num_classes = n_cls

    mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,
                  mode="batch", label_smoothing=args.smoothing, num_classes=n_cls)
    crit_soft = SoftTargetCrossEntropy()
    crit_hard = LabelSmoothingCrossEntropy(smoothing=args.smoothing)

    groups = layer_decay_param_groups(model, args.lr, args.weight_decay, args.layer_decay)
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    steps_per_epoch = max(1, len(dl_tr) // args.accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup
    base_lrs = [g["lr"] for g in opt.param_groups]

    def set_lr(step):
        if step < warmup_steps:
            f = step / max(1, warmup_steps)
        else:
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            f = 0.5 * (1 + math.cos(math.pi * prog))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f

    if args.smoke:
        xb, yb = next(iter(dl_tr))
        with torch.cuda.amp.autocast():
            out = model(xb.to(device))
        print(f"SMOKE OK: batch={tuple(xb.shape)} logits={tuple(out.shape)} "
              f"n_param_groups={len(opt.param_groups)}")
        ev = evaluate(model, dl_va, device, tta=False)
        print("SMOKE val (untrained):", {k: round(v, 4) for k, v in ev.items()})
        return

    best_f1, best_state, gstep = -1, None, 0
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        for it, (x, y) in enumerate(dl_tr):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            xm, ym = mixup(x, y)
            with torch.cuda.amp.autocast():
                loss = crit_soft(model(xm), ym) / args.accum
            scaler.scale(loss).backward()
            if (it + 1) % args.accum == 0:
                set_lr(gstep); gstep += 1
                scaler.step(opt); scaler.update(); opt.zero_grad()
        ev = evaluate(model, dl_va, device, tta=False)
        print(f"[ep {ep+1:02d}/{args.epochs}] val macro-F1={ev['macro_f1']:.4f} "
              f"bacc={ev['bacc']:.4f} wf1={ev['weighted_f1']:.4f}")
        if ev["macro_f1"] > best_f1:
            best_f1 = ev["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    res = evaluate(model, dl_te, device, tta=not args.no_tta)
    res["best_val_macro_f1"] = best_f1
    res["images"] = args.images
    res["backbone"] = args.backbone
    res["seed"] = args.seed
    print("\n" + "=" * 60)
    print(f"TEST (images={args.images}, TTA={not args.no_tta}):")
    for k in ["macro_f1", "weighted_f1", "bacc", "acc", "auroc"]:
        print(f"  {k:12s}: {res[k]:.4f}")
    print("=" * 60)
    tag = args.zone if args.zone else "gen"
    bb = "base" if "base" in args.backbone else ("large" if "large" in args.backbone else "bb")
    res["zone"] = args.zone or "GEN"
    stem = f"{tag}_{bb}_{args.images}_seed{args.seed}"
    with open(os.path.join(args.out, f"result_{stem}.json"), "w") as f:
        json.dump(res, f, indent=2)
    torch.save(best_state, os.path.join(args.out, f"model_{stem}_best.pth"))
    print(f"saved: model_{stem}_best.pth")


if __name__ == "__main__":
    main()
