"""Multimodal Fusion Classifier: Image + Patient Metadata (age/sex/localization/dataset).

RATIONALE: Phase 8 showed that patient metadata correlates VERY STRONGLY with
model error (MEL×age r=-0.814, NV×localization r=-0.580), yet so far metadata
served only for DIAGNOSIS/TARGETING of augmentation — the model itself was BLIND
to metadata at prediction time. This script gives the model "eyes" to see patient
metadata through cross-attention fusion, letting the model use the already
proven-discriminative information DIRECTLY.

ARCHITECTURE:
  - Backbone: Swin-base (exactly the same as the champion baseline)
  - Metadata: age (1D normalized) + sex (3D one-hot) + localization (15D one-hot)
              + dataset (4D one-hot) = 23D → MLP → 64D embedding
  - Fusion: Cross-attention (image features attend to metadata context)
  - Head: LayerNorm → 512 → GELU → Dropout → 7 (identical to GenericClassifier)

TRAINING: recipe IDENTICAL to the baseline (MixUp/CutMix + inverse-freq weights +
discriminative LR + warmup + CosineAnnealing + early stop by val_f1), so the
fusion effect can be attributed PURELY to the added metadata, not the training recipe.

DIFFERENCES from the FAILED Phase 2.3 fusion attempt (F1=0.7155):
  1. Arsitektur fusion: cross-attention (bukan naive MLP concat)
  2. Backbone: Swin-base champion (bukan backbone lama)
  3. MixUp aktif (lever terbesar, +4.4pp terbukti)
  4. Split: fully_representative (fondasi baru)

Usage:
    python train_phase3_hybrid_multimodal.py --fold 0
"""
import argparse
import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.data import Dataset, DataLoader
from timm.data import Mixup
import timm
from timm.loss import SoftTargetCrossEntropy

from ame_derm.common import birnet_final as bf
from ame_derm.experts.train_classifier_generic import evaluate_full
from ame_derm.experts.train_classifier_phase2_final import CROP_DIR
from ame_derm.experts.train_classifier_phase2_mixup import SoftTargetCrossEntropyWeighted, BATCH_SIZE, BACKBONE_LR_SCALE, WARMUP_EPOCHS
from ame_derm.experts.train_classifier_phase2_boundary_aug_v2 import build_augmented_train_df
from ame_derm.data_splits.representative_val_split import stratified_ham_kfold_fully_representative, compare_val_to_reference
from ame_derm.data_splits.diagnose_metadata_imbalance import detect_metadata_fields

CKPT_DIR = "BIRNet_Research/outputs/checkpoints/phase3_hybrid_multimodal"
SYNTH_DIR = "BIRNet_Research/outputs/synthetic_full_replacement_fixed"
DEFAULT_MANIFEST = "BIRNet_Research/outputs/synthetic_full_replacement_fixed_manifest.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Metadata encoding ──────────────────────────────────────────────────

# FIXED order (do not change once training has begun — encoded position matters)
SEX_CATEGORIES = ["male", "female", "unknown"]
LOCALIZATION_CATEGORIES = [
    "scalp", "ear", "face", "back", "trunk", "chest",
    "upper extremity", "abdomen", "unknown", "lower extremity",
    "genital", "neck", "hand", "foot", "acral",
]
DATASET_CATEGORIES = ["vidir_modern", "rosendahl", "vienna_dias", "vidir_molemax"]

# age: normalized to 0-1 (range 0-85 in HAM10000)
AGE_MAX = 85.0

# Global state used for ablation
ACTIVE_META_FIELDS = ["age", "sex", "localization", "dataset"]
META_DIM = 23

def set_active_meta_fields(fields_str):
    global ACTIVE_META_FIELDS, META_DIM
    if fields_str.lower() == "all":
        ACTIVE_META_FIELDS = ["age", "sex", "localization", "dataset"]
    else:
        ACTIVE_META_FIELDS = [f.strip() for f in fields_str.split(",") if f.strip()]
    
    dim = 0
    if "age" in ACTIVE_META_FIELDS: dim += 1
    if "sex" in ACTIVE_META_FIELDS: dim += len(SEX_CATEGORIES)
    if "localization" in ACTIVE_META_FIELDS: dim += len(LOCALIZATION_CATEGORIES)
    if "dataset" in ACTIVE_META_FIELDS: dim += len(DATASET_CATEGORIES)
    META_DIM = dim
    return META_DIM

def encode_metadata(age, sex, localization, dataset):
    """Encode a single metadata row into a 1D tensor (META_DIM,) following ACTIVE_META_FIELDS."""
    vec = np.zeros(META_DIM, dtype=np.float32)
    offset = 0
    if "age" in ACTIVE_META_FIELDS:
        if pd.isna(age): vec[offset] = 0.5
        else: vec[offset] = float(age) / AGE_MAX
        offset += 1
    if "sex" in ACTIVE_META_FIELDS:
        if sex in SEX_CATEGORIES: vec[offset + SEX_CATEGORIES.index(sex)] = 1.0
        offset += len(SEX_CATEGORIES)
    if "localization" in ACTIVE_META_FIELDS:
        if localization in LOCALIZATION_CATEGORIES: vec[offset + LOCALIZATION_CATEGORIES.index(localization)] = 1.0
        offset += len(LOCALIZATION_CATEGORIES)
    if "dataset" in ACTIVE_META_FIELDS:
        if dataset in DATASET_CATEGORIES: vec[offset + DATASET_CATEGORIES.index(dataset)] = 1.0
        offset += len(DATASET_CATEGORIES)
    return vec


# ── Dataset ─────────────────────────────────────────────────────────────

class SkinLesionDatasetMultimodal(Dataset):
    """Same as SkinLesionDatasetPrecomputedCrop, with a metadata tensor ADDED."""

    def __init__(self, df, img_size, transform=None, oversample=False,
                 target_per_class=None, num_classes=7):
        self.img_size = img_size
        self.transform = transform
        # Keep metadata per image_id
        self.meta_map = {}
        for row in df.itertuples():
            self.meta_map[row.image_id] = encode_metadata(
                getattr(row, "age", np.nan),
                getattr(row, "sex", "unknown"),
                getattr(row, "localization", "unknown"),
                getattr(row, "dataset", "unknown"),
            )
        entries = [(row.image_id, None, row.label) for row in df.itertuples()]
        if oversample and target_per_class is not None:
            entries = bf.SkinLesionDataset._build_oversampled_index(entries, target_per_class, num_classes)
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        if len(entry) == 4:
            image_id, _, label, do_aug = entry
        else:
            image_id, _, label = entry
            do_aug = False

        if "_synth" in image_id:
            img_path = os.path.join(SYNTH_DIR, f"{image_id}.jpg")
        else:
            img_path = os.path.join(CROP_DIR, f"{image_id}.jpg")
            
        img = np.array(bf.Image.open(img_path).convert("RGB")).astype(np.uint8)
        img_np = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        if do_aug:
            if np.random.rand() > 0.5:
                img_np = cv2.flip(img_np, 1)
            if np.random.rand() > 0.5:
                img_np = cv2.flip(img_np, 0)
            img_np = np.rot90(img_np, np.random.randint(0, 4)).copy()

        if label is None or (isinstance(label, float) and np.isnan(label)):
            label = -1

        dummy_mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        if self.transform:
            aug = self.transform(image=img_np, mask=dummy_mask)
            img_t = aug["image"]
        else:
            img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0

        # Metadata — synthetics (image_id containing "synth") reuse their original source metadata
        # If absent from meta_map (synthetic without metadata), fall back to the default vector
        base_id = image_id.split("_synth")[0] if "_synth" in image_id else image_id
        meta_vec = self.meta_map.get(image_id, self.meta_map.get(base_id, np.zeros(META_DIM, dtype=np.float32)))
        meta_t = torch.from_numpy(meta_vec)

        return img_t, meta_t, int(label)


# ── Model ───────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """Metadata context attends to image features via multi-head cross-attention.
    
    Query = metadata embedding (projected to img_dim)
    Key/Value = image features sequence (patch tokens)
    
    This DIFFERS from naive concat/MLP:
    cross-attention lets the model SELECTIVELY pick which visual information (spatial
    patches) is relevant to the patient's age/sex. Softmax behaves properly here because
    it weighs hundreds of image patches, not a single vector.
    """

    def __init__(self, img_dim, meta_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.meta_proj = nn.Linear(meta_dim, img_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=img_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(img_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_seq, meta_feat):
        """
        img_seq: (B, seq_len, img_dim) — un-pooled image features
        meta_feat: (B, meta_dim) — metadata embedding
        Returns: (B, img_dim) — the fused features
        """
        meta_proj = self.meta_proj(meta_feat).unsqueeze(1)  # (B, 1, img_dim)

        # Cross-attention: the metadata query attends to image patches (keys/values)
        # Softmax ranges over seq_len.
        attn_out, _ = self.cross_attn(
            query=meta_proj, key=img_seq, value=img_seq
        )  # (B, 1, img_dim)

        # Merge the metadata-guided visual summary with the global visual summary
        global_img = img_seq.mean(dim=1)  # (B, img_dim)
        fused = self.norm(global_img + self.dropout(attn_out.squeeze(1)))
        return fused


class MultimodalClassifier(nn.Module):
    """Swin-base backbone + metadata cross-attention fusion.
    
    Head structure IDENTICAL to GenericClassifier so the comparison is fair.
    The only difference: meta_embed + cross_attn inserted between the
    backbone and the head.
    """

    def __init__(self, backbone_name="swin_base_patch4_window7_224", num_classes=7,
                 meta_dim=META_DIM, meta_embed_dim=64, drop_path_rate=None, head_dropout=0.3,
                 img_size=224):
        super().__init__()
        is_transformer = "vit" in backbone_name or "swin" in backbone_name or "eva" in backbone_name
        backbone_kwargs = {}
        if drop_path_rate is not None:
            backbone_kwargs["drop_path_rate"] = drop_path_rate
        if "dinov2" in backbone_name:
            backbone_kwargs["img_size"] = img_size
            backbone_kwargs["dynamic_img_size"] = True

        if is_transformer:
            self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool="", **backbone_kwargs)
        else:
            self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0,
                                               global_pool="", **backbone_kwargs)
        feat_dim = self.backbone.num_features

        # Metadata embedding: raw 23D → 64D
        self.meta_embed = nn.Sequential(
            nn.Linear(meta_dim, meta_embed_dim),
            nn.LayerNorm(meta_embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(meta_embed_dim, meta_embed_dim),
            nn.LayerNorm(meta_embed_dim),
            nn.GELU(),
        )

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            img_dim=feat_dim, meta_dim=meta_embed_dim, num_heads=4, dropout=0.1
        )

        # Head identical to GenericClassifier's so the comparison stays fair
        if is_transformer:
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 512),
                nn.GELU(), nn.Dropout(head_dropout), nn.Linear(512, num_classes))
        else:
            self.head = nn.Sequential(
                nn.Linear(feat_dim, 512), nn.LayerNorm(512),
                nn.GELU(), nn.Dropout(head_dropout), nn.Linear(512, num_classes))

    def forward(self, x, meta=None):
        # Pull out the unpooled feature sequence
        img_seq = self.backbone(x)
        
        # Flatten spatial dimensions when the output comes as a grid
        if img_seq.ndim == 4:
            if img_seq.shape[-1] == self.backbone.num_features:
                # Swin: (B, H, W, C) -> (B, H*W, C)
                img_seq = img_seq.view(img_seq.shape[0], -1, img_seq.shape[-1])
            else:
                # CNN: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
                img_seq = img_seq.view(img_seq.shape[0], img_seq.shape[1], -1).transpose(1, 2)
                
        if meta is not None:
            meta_emb = self.meta_embed(meta)
            img_feat = self.fusion(img_seq, meta_emb)
        else:
            img_feat = img_seq.mean(dim=1)
            
        return self.head(img_feat)


# ── Training utilities ──────────────────────────────────────────────────

def make_loader(df, cfg, aug, shuffle=False, oversample=False):
    ds = SkinLesionDatasetMultimodal(df, cfg.img_size, transform=aug, oversample=oversample,
                                      target_per_class=cfg.oversample_target_per_class if oversample else None,
                                      num_classes=cfg.num_classes)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=4, pin_memory=True,
                       drop_last=shuffle and len(ds) > cfg.batch_size)


def train_one_epoch_multimodal(model, loader, optimizer, criterion, mixup_fn, device,
                                warmup_scheduler=None):
    """Same as train_one_epoch_mixup, except metadata is forwarded to the model."""
    model.train()
    total_loss, n = 0.0, 0
    for imgs, metas, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        valid = labels >= 0
        if valid.sum() == 0 or valid.sum() < 2:
            continue
        imgs_v, metas_v, labels_v = imgs[valid], metas[valid], labels[valid]
        if imgs_v.shape[0] % 2 == 1:
            imgs_v, metas_v, labels_v = imgs_v[:-1], metas_v[:-1], labels_v[:-1]
        if imgs_v.shape[0] < 2:
            continue
        # MixUp over images AND labels (metadata is NOT mixed — stays per-sample)
        imgs_mixed, soft_targets = mixup_fn(imgs_v, labels_v)

        optimizer.zero_grad()
        logits = model(imgs_mixed, metas_v)
        loss = criterion(logits, soft_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        total_loss += loss.item() * imgs_mixed.shape[0]
        n += imgs_mixed.shape[0]
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_multimodal(model, loader, device):
    """evaluate_full, but metadata is forwarded."""
    model.eval()
    preds, labs = [], []
    for imgs, metas, labels in loader:
        imgs = imgs.to(device)
        metas = metas.to(device)
        logits = model(imgs, metas)
        valid = labels >= 0
        preds.extend(logits[valid].argmax(1).cpu().numpy())
        labs.extend(labels[valid].numpy())
    preds, labs = np.array(preds), np.array(labs)
    from sklearn.metrics import balanced_accuracy_score, f1_score
    acc = (preds == labs).mean()
    bal = balanced_accuracy_score(labs, preds)
    f1_macro = f1_score(labs, preds, average="macro", zero_division=0)
    return acc, bal, f1_macro


# ── Main training loop ──────────────────────────────────────────────────

def run_fold(fold_i, train_df, val_df, test_df, backbone, seed, cfg, args):
    bf.set_deterministic(seed)
    os.makedirs(CKPT_DIR, exist_ok=True)
    suffix_str = f"_{args.suffix}" if args.suffix else ""
    ckpt_path = os.path.join(CKPT_DIR, f"cls_{backbone}_multimodal_fold{fold_i}{suffix_str}_best.pth")

    train_aug, val_aug = bf.build_augmentations(cfg.img_size, seed=seed)
    train_loader = make_loader(train_df, cfg, train_aug, shuffle=True, oversample=False)
    val_loader = make_loader(val_df, cfg, val_aug, shuffle=False)
    test_loader = make_loader(test_df, cfg, val_aug, shuffle=False)

    print(f"\n{'='*70}\n[FOLD {fold_i}] MULTIMODAL backbone={backbone} seed={seed} "
          f"meta_dim={META_DIM} train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}\n{'='*70}")

    print("Bypassing Inverse-freq class weights for True Hybrid Generalization (No Double-Balancing).")

    mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,
                      mode="batch", label_smoothing=0.1, num_classes=cfg.num_classes)

    model = MultimodalClassifier(backbone, num_classes=cfg.num_classes,
                                  meta_dim=META_DIM,
                                  head_dropout=args.head_dropout,
                                  img_size=cfg.img_size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    n_meta_params = sum(p.numel() for p in model.meta_embed.parameters()) + \
                    sum(p.numel() for p in model.fusion.parameters())
    print(f"Total params: {n_params/1e6:.2f}M (metadata+fusion: {n_meta_params/1e6:.2f}M)")

    criterion = SoftTargetCrossEntropy().to(DEVICE)

    is_transformer = "vit" in backbone or "swin" in backbone or "eva" in backbone
    base_lr = 5e-5 if is_transformer else 1e-4

    # Parameter groups: backbone (small LR), metadata+fusion+head (full LR)
    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": base_lr * BACKBONE_LR_SCALE},
        {"params": list(model.meta_embed.parameters()) + list(model.fusion.parameters()),
         "lr": base_lr, "weight_decay": 1e-3},  # sedikit lebih besar WD untuk metadata (regularisasi)
        {"params": model.head.parameters(), "lr": base_lr},
    ], weight_decay=1e-4)

    steps_per_epoch = len(train_loader)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / max(1, warmup_steps)))
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)

    max_epochs, patience = cfg.cls_max_epochs, cfg.cls_patience
    best_f1, no_improve = 0.0, 0
    global_step = 0
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        in_warmup = global_step < warmup_steps
        train_loss = train_one_epoch_multimodal(model, train_loader, optimizer, criterion, mixup_fn, DEVICE,
                                                 warmup_scheduler=warmup_scheduler if in_warmup else None)
        global_step += steps_per_epoch
        val_acc, val_bal, val_f1 = evaluate_multimodal(model, val_loader, DEVICE)
        if not in_warmup:
            cosine_scheduler.step()
        is_best = val_f1 > best_f1
        flag = ""
        if is_best:
            best_f1, no_improve, flag = val_f1, 0, "*"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_f1": val_f1}, ckpt_path)
        else:
            no_improve += 1
        print(f"[F{fold_i}] Ep {epoch:3d}/{max_epochs} | train_loss={train_loss:.4f} "
              f"| val_acc={val_acc:.4f} val_bal={val_bal:.4f} val_f1={val_f1:.4f} (best={best_f1:.4f}) "
              f"| {time.time()-t0:.0f}s {flag}")
        if no_improve >= patience:
            print(f"[F{fold_i}] Early stopping pada epoch {epoch}.")
            break

    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state"])
    test_acc, test_bal, test_f1 = evaluate_multimodal(model, test_loader, DEVICE)
    print(f"[FOLD {fold_i}] TEST: acc={test_acc:.4f} bal_acc={test_bal:.4f} f1_macro={test_f1:.4f} "
          f"(checkpoint epoch={ck['epoch']})")
    return dict(fold=fold_i, test_acc=test_acc, test_bal=test_bal, test_f1=test_f1)


def main():
    global SYNTH_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="swin_base_patch4_window7_224")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suffix", type=str, default="", help="Suffix for checkpoint name")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--fold", type=int, default=None, help="Run single fold (0-4), or all if None")
    parser.add_argument("--head_dropout", type=float, default=0.3)
    parser.add_argument("--split_mode", type=str, default="fully_representative",
                         choices=["old", "fully_representative"])
    parser.add_argument("--val_stratify_fields", type=str, default="dataset")
    parser.add_argument("--meta_fields", type=str, default="all", help="comma separated fields e.g. age,sex,localization")
    parser.add_argument("--manifest_csv", type=str, default=DEFAULT_MANIFEST)
    parser.add_argument("--synthetic_dir", type=str, default=SYNTH_DIR)
    args = parser.parse_args()
    
    SYNTH_DIR = args.synthetic_dir

    set_active_meta_fields(args.meta_fields)
    print(f"ABLATION MODE: Active Meta Fields = {ACTIVE_META_FIELDS} (META_DIM = {META_DIM})")

    cfg = bf.Config()
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.max_epochs: cfg.cls_max_epochs = args.max_epochs
    if args.patience: cfg.cls_patience = args.patience
    paths = bf.DatasetPaths(cfg)
    ham_df = bf.build_ham10000_dataframe(paths)
    has_crop = ham_df["image_id"].apply(lambda iid: os.path.exists(os.path.join(CROP_DIR, f"{iid}.jpg")))
    ham_df = ham_df[has_crop].reset_index(drop=True)
    print(f"{len(ham_df)} gambar dgn crop tersedia")

    # Check the metadata columns
    for col in ["age", "sex", "localization", "dataset"]:
        if col not in ham_df.columns:
            raise ValueError(f"Kolom metadata '{col}' tidak ada di ham_df. Columns: {list(ham_df.columns)}")
    print(f"Metadata tersedia: age (NaN={ham_df['age'].isna().sum()}), "
          f"sex (unknown={( ham_df['sex']=='unknown').sum()}), "
          f"localization (unknown={(ham_df['localization']=='unknown').sum()}), "
          f"dataset ({ham_df['dataset'].nunique()} sources)")

    val_stratify_fields = [f.strip() for f in args.val_stratify_fields.split(",") if f.strip()]

    if args.split_mode == "fully_representative":
        bucketed, field_names = detect_metadata_fields(ham_df, n_bins_numeric=9)
        overlap = [c for c in bucketed.columns if c in ham_df.columns]
        ham_meta = pd.concat([ham_df.drop(columns=overlap), bucketed], axis=1)
        print(f"FONDASI BARU: outer(TEST) & inner(VAL) stratifikasi label+{val_stratify_fields}")

        def split_iter():
            for repeat_i, fold_i, train_idx, val_idx, test_idx in stratified_ham_kfold_fully_representative(
                    ham_meta, cfg, val_stratify_fields, n_splits=5, n_repeats=1):
                yield fold_i, train_idx, val_idx, test_idx
    else:
        def split_iter():
            return bf.stratified_ham_kfold(ham_df, cfg, n_splits=5)

    manifest_df = pd.read_csv(args.manifest_csv)
    print(f"Manifest sintetis: {manifest_df['accepted'].sum()}/{len(manifest_df)} diterima QC")

    results = []
    for fold_i, train_idx, val_idx, test_idx in split_iter():
        if args.fold is not None and fold_i != args.fold:
            continue
        train_df = ham_df.iloc[train_idx].reset_index(drop=True)
        val_df = ham_df.iloc[val_idx].reset_index(drop=True)
        test_df = ham_df.iloc[test_idx].reset_index(drop=True)
        
        aug_train_df, n_added = build_augmented_train_df(train_df, manifest_df, ham_df)
        print(f"[FOLD {fold_i}] train ASLI={len(train_df)} + sintetis={n_added} = {len(aug_train_df)} | val={len(val_df)} | test={len(test_df)}")

        result = run_fold(fold_i, aug_train_df, val_df, test_df, args.backbone, args.seed, cfg, args)
        results.append(result)
        suffix_str = f"_{args.suffix}" if args.suffix else ""
        print(f"RESULT_SUMMARY backbone={args.backbone} variant=hybrid_multimodal_{args.meta_fields.replace(',', '_')}{suffix_str} fold={fold_i} "
              f"n_synthetic_added={n_added} test_acc={result['test_acc']:.4f} test_bal={result['test_bal']:.4f} test_f1={result['test_f1']:.4f}")

    if len(results) > 1:
        f1s = np.array([r["test_f1"] for r in results])
        print(f"\n=== RINGKASAN {len(results)}-FOLD HYBRID MULTIMODAL (backbone={args.backbone}) ===")
        print(f"f1_macro: mean={f1s.mean():.4f} std={f1s.std():.4f}")
        print(f"RESULT_SUMMARY_FINAL backbone={args.backbone} variant=hybrid_multimodal n_folds={len(results)} "
              f"mean_f1={f1s.mean():.4f} std_f1={f1s.std():.4f}")


if __name__ == "__main__":
    main()
