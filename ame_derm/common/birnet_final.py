# -*- coding: utf-8 -*-
"""
BIRNet: A Boundary Irregularity-Guided Cross-Attention Network for
Joint Skin Lesion Segmentation and Classification.

Revisi pasca-penolakan Medical Image Analysis (MEDIA-D-26-02004, alasan:
novelty metodologis tidak cukup). Perubahan utama pada revisi ini: modul
BIG diganti dari elementwise spatial gating menjadi genuine multi-head
scaled dot-product cross-attention (lihat BIGCrossAttention), dan
ditambahkan ablasi kontrol (placebo, oracle) untuk membuktikan gain
performa berasal dari sinyal boundary irregularity, bukan sekadar
tambahan kapasitas parameter. Modul gating generasi awal dipertahankan
sebagai BIGGatingLegacy untuk perbandingan head-to-head.

Komponen utama:
    - Shared encoder EfficientNet-B4
    - Decoder segmentasi bergaya UNet++ dengan modul ASPP-lite
    - Head klasifikasi CBAM
    - Modul BIG (Boundary Irregularity-Guided), dua varian:
        * BIGGatingLegacy  : gating elementwise generasi awal (config C5)
              F'_cls = F_cls (dot) (1 + alpha * I_tilde)
        * BIGCrossAttention: cross-attention Q/K/V genuine (config C6),
          Query dari irregularity map (Sobel + local variance w=9),
          Key/Value dari classification feature, residual gamma-scaled:
              F'_cls = F_cls + gamma * Attn(Q=Encode(I_tilde), K=F_cls, V=F_cls)
    - Joint loss:
          L_total = 0.40 * L_seg + 0.50 * L_cls + 0.10 * L_boundary
          L_seg   = Dice + BCE + 0.5 * Tversky(alpha=0.3, beta=0.7)
          L_cls   = Label Smoothing Focal Loss (gamma=2.0, eps=0.1)
          L_bnd   = BCE(Sobel(pred_mask), Sobel(gt_mask))
    - Threshold segmentasi optimal tau* = 0.78 (hasil grid search pada
      validation set ISIC 2018 Task1, dilaporkan di Bagian 3.4 paper)
    - Ablasi C1-C8 (lihat ABLATION_CONFIGS): C1-C5 sesuai draft submission
      MedIA, C6 (cross-attention baru), C7 (placebo, tanpa sinyal
      irregularity), C8 (oracle, BIG diberi GT mask saat training sebagai
      diagnostic upper-bound, TIDAK dipakai untuk angka final). Jalankan
      lewat `python birnet_final.py --config C6`.

Dataset yang digunakan:
    - HAM10000            : klasifikasi utama, split stratified 70/15/15 (seed 42)
    - ISIC 2018 Task1      : segmentasi utama, 2594/100/1000
    - ISIC 2016, ISIC 2017 : data tambahan untuk training segmentasi
    - PH2                  : data tambahan untuk training segmentasi
    - ISIC 2018 Task3      : evaluasi cross-dataset (tidak overlap dengan training)

Catatan penting:
    - Semua random seed dikunci ke 42 untuk reproducibility.
    - Oversampling dilakukan HANYA pada partisi training, dengan augmentasi
      geometris (flip + rotasi), bukan SMOTE/ADASYN (paper menunjukkan
      augmentation-based oversampling mengungguli keduanya). Bisa
      dimatikan lewat toggle use_oversampling pada config ablasi.
    - Arsitektur pencarian sebelumnya (hybrid U-Net-EfficientNet ensemble)
      TIDAK digunakan di sini karena telah ditinggalkan dalam riset ini.
"""

import os
import json
import time
import random
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score


def set_deterministic(seed: int):
    """Enforces full run-to-run determinism -- model init, data shuffling,
    oversampling duplicate choice, augmentation (see build_augmentations'
    `seed` kwarg), AND cuDNN/CUDA kernel selection -- so that any measured
    difference between two runs (e.g. plain vs BGFF mask, HAM-only vs
    pooled-BCN) reflects the variable actually being tested, not incidental
    non-determinism from GPU algorithm autotuning. Without this, cuDNN's
    default `benchmark` mode picks among several convolution algorithms
    heuristically and can return slightly different numerical results run
    to run even with identical seeds elsewhere.

    Trade-off: disables cuDNN's fastest autotuned kernels, so training is
    measurably slower. Accepted deliberately here so that ablation
    comparisons are not confounded by "cherry-picking" from uncontrolled
    randomness -- correctness over speed for verification experiments."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ======================================================================
# 1. KONFIGURASI
# ======================================================================

@dataclass
class Config:
    # Path dasar (sesuaikan dengan lokasi dataset)
    base_dir: str = "./BIRNet_Research"
    dataset_dir: str = field(init=False)
    output_dir: str = field(init=False)
    ckpt_dir: str = field(init=False)
    log_dir: str = field(init=False)

    # Reproducibility
    seed: int = 42

    # Data
    img_size: int = 224
    num_classes: int = 7
    class_names: Tuple[str, ...] = (
        "MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"
    )
    oversample_target_per_class: int = 1500

    # Split HAM10000
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15

    # Training - klasifikasi
    cls_batch_size: int = 32
    cls_lr: float = 1e-4
    cls_weight_decay: float = 1e-4
    cls_max_epochs: int = 80
    cls_patience: int = 12
    dropout: float = 0.4

    # Training - segmentasi
    seg_batch_size: int = 16
    seg_lr: float = 1e-4
    seg_weight_decay: float = 1e-4
    seg_max_epochs: int = 80
    seg_patience: int = 15
    seg_threshold: float = 0.78  # tau*, hasil grid search validasi ISIC 2018 Task1

    # Scheduler
    scheduler_t0: int = 20
    scheduler_eta_min: float = 1e-6

    # Joint loss weights
    lambda_seg: float = 0.40
    lambda_cls: float = 0.50
    lambda_boundary: float = 0.10

    # BIG module
    big_alpha: float = 1.0          # dipakai hanya oleh varian "gating" (legacy)
    big_window: int = 9
    big_variant: str = "crossattn"  # none | gating | crossattn | placebo | oracle

    # Label smoothing focal loss
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1

    # Tversky loss (segmentasi)
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7

    # Toggle komponen ablasi (lihat ABLATION_CONFIGS, config C1-C8)
    use_boundary_loss: bool = True
    use_label_smoothing: bool = True
    use_oversampling: bool = True

    # Nama config ablasi aktif, dipakai untuk memisahkan checkpoint/log
    # antar-run supaya tidak saling menimpa saat menjalankan C1..C8
    config_name: str = "C6"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.dataset_dir = os.path.join(self.base_dir, "dataset")
        self.output_dir = os.path.join(self.base_dir, "outputs")
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints", self.config_name)
        self.log_dir = os.path.join(self.output_dir, "logs", self.config_name)
        for d in (self.output_dir, self.ckpt_dir, self.log_dir):
            os.makedirs(d, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================================================
# 1b. KONFIGURASI ABLASI (C1-C8)
# ======================================================================
#
# C1-C5 sesuai Tabel 7 pada draft submission MedIA. C6-C8 adalah ablasi
# tambahan untuk memperkuat klaim novelty pada revisi:
#   C6 - BIGCrossAttention menggantikan BIGGatingLegacy sebagai modul utama
#        (cross-attention Q/K/V genuine, bukan elementwise gating).
#   C7 - "placebo": arsitektur cross-attention identik dengan C6, tapi
#        Query dikosongkan (tanpa sinyal irregularity). Mengisolasi apakah
#        gain performa berasal dari sinyal boundary irregularity itu
#        sendiri, atau hanya dari tambahan kapasitas attention.
#   C8 - "oracle": BIG menerima GT mask (bukan mask prediksi) saat
#        training/diagnostic eval. Mengisolasi seberapa besar performa
#        C6 dibatasi oleh kualitas mask segmentasi yang diprediksi.
#        HANYA untuk diagnostic run, tidak pernah dipakai sebagai angka
#        akhir yang dilaporkan (memakai GT test set = bukan kondisi
#        deployment nyata).
ABLATION_CONFIGS: Dict[str, Dict[str, object]] = {
    "C1": dict(big_variant="none",      use_boundary_loss=False, use_label_smoothing=False, use_oversampling=False),
    "C2": dict(big_variant="none",      use_boundary_loss=False, use_label_smoothing=False, use_oversampling=True),
    "C3": dict(big_variant="none",      use_boundary_loss=False, use_label_smoothing=True,  use_oversampling=True),
    "C4": dict(big_variant="none",      use_boundary_loss=True,  use_label_smoothing=True,  use_oversampling=True),
    "C5": dict(big_variant="gating",    use_boundary_loss=True,  use_label_smoothing=True,  use_oversampling=True),
    "C6": dict(big_variant="crossattn", use_boundary_loss=True,  use_label_smoothing=True,  use_oversampling=True),
    "C7": dict(big_variant="placebo",   use_boundary_loss=True,  use_label_smoothing=True,  use_oversampling=True),
    "C8": dict(big_variant="oracle",    use_boundary_loss=True,  use_label_smoothing=True,  use_oversampling=True),
}


def apply_ablation_config(cfg: Config, config_name: str) -> Config:
    """Menerapkan salah satu preset ABLATION_CONFIGS ke sebuah Config,
    termasuk mengarahkan ulang ckpt_dir/log_dir supaya tiap config
    ablasi punya direktori output sendiri (tidak saling menimpa)."""
    if config_name not in ABLATION_CONFIGS:
        raise ValueError(f"Config ablasi tidak dikenal: {config_name}. Pilihan: {list(ABLATION_CONFIGS)}")
    opts = ABLATION_CONFIGS[config_name]
    cfg.big_variant = opts["big_variant"]
    cfg.use_boundary_loss = opts["use_boundary_loss"]
    cfg.use_label_smoothing = opts["use_label_smoothing"]
    cfg.use_oversampling = opts["use_oversampling"]
    cfg.config_name = config_name
    cfg.__post_init__()  # rebuild ckpt_dir/log_dir dengan config_name baru
    return cfg


# ======================================================================
# 2. PATH DATASET
# ======================================================================

class DatasetPaths:
    """Kumpulan path ke seluruh dataset yang digunakan pada paper."""

    def __init__(self, cfg: Config):
        d = cfg.dataset_dir

        # HAM10000
        self.ham_img1 = os.path.join(d, "ham10000/HAM10000_classification/HAM10000_images_part_1")
        self.ham_img2 = os.path.join(d, "ham10000/HAM10000_classification/HAM10000_images_part_2")
        self.ham_mask = os.path.join(d, "ham10000/HAM10000_segmentations_lesion_tschandl")
        self.ham_csv = os.path.join(d, "ham10000/HAM10000_classification/HAM10000_metadata.csv")

        # ISIC 2018 Task 1 (segmentasi)
        self.isic_t1_train_img = os.path.join(d, "isic2018/isic_task1/ISIC2018_Task1-2_Training_Input")
        self.isic_t1_train_mask = os.path.join(d, "isic2018/isic_task1/ISIC2018_Task1_Training_GroundTruth")
        self.isic_t1_val_img = os.path.join(d, "isic2018/isic_task1/ISIC2018_Task1-2_Validation_Input")
        self.isic_t1_val_mask = os.path.join(d, "isic2018/isic_task1/ISIC2018_Task1_Validation_GroundTruth")
        self.isic_t1_test_img = os.path.join(d, "isic2018/isic_task1/ISIC2018_Task1-2_Test_Input")
        self.isic_t1_test_mask = os.path.join(d, "isic2018/isic_task1/ISIC2018_Task1_Test_GroundTruth")

        # ISIC 2018 Task 3 (evaluasi cross-dataset, tanpa GT mask)
        self.isic_t3_train_img = os.path.join(d, "isic2018/isic_task3/ISIC2018_Task3_Training_Input")
        self.isic_t3_train_csv = os.path.join(
            d, "isic2018/isic_task3/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"
        )

        # ISIC 2016 / 2017 (data tambahan segmentasi)
        self.isic16_img = os.path.join(d, "isic2016/ISBI2016_ISIC_Part1_Training_Data")
        self.isic16_mask = os.path.join(d, "isic2016/ISBI2016_ISIC_Part1_Training_GroundTruth")
        self.isic17_img = os.path.join(d, "isic2017/ISIC-2017_Training_Data")
        self.isic17_mask = os.path.join(d, "isic2017/ISIC-2017_Training_Part1_GroundTruth")

        # PH2 (data tambahan segmentasi)
        self.ph2_root = os.path.join(d, "ph2/PH2Dataset/PH2 Dataset images")
        self.ph2_txt = os.path.join(d, "ph2/PH2Dataset/PH2_dataset.txt")


# ======================================================================
# 3. PREPROCESSING & AUGMENTASI
# ======================================================================

def build_augmentations(img_size: int, seed: int = None):
    """Augmentasi training & validasi sesuai Bagian 3.2 paper (CLAHE,
    bilateral filter dilakukan terpisah di load_and_preprocess_image;
    di sini hanya augmentasi geometris/warna standar).

    NOTE on reproducibility: albumentations>=2.0 uses its OWN internal RNG
    (np.random.default_rng(seed) + random.Random(seed) inside
    BasicTransform.set_random_seed / A.Compose's `seed` kwarg) -- it is
    COMPLETELY INDEPENDENT of the global torch.manual_seed()/np.random.seed()
    calls scripts make. Verified empirically: two separate processes with
    identical torch/numpy seeding but no `seed=` passed here produced
    DIFFERENT augmented outputs for the same input. Callers that need true
    run-to-run reproducibility (e.g. multi-seed verification experiments)
    MUST pass their --seed argument through here; existing callers that
    don't pass a seed keep the previous (non-reproducible-augmentation)
    behavior unchanged."""

    train_aug = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], is_check_shapes=False, seed=seed)

    val_aug = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], is_check_shapes=False, seed=seed)

    return train_aug, val_aug


def load_and_preprocess_image(path: str, img_size: int) -> np.ndarray:
    """CLAHE (LAB space) + bilateral filtering, sesuai Bagian 3.2."""
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".bmp":
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.array(Image.open(path).convert("RGB"))
    img = img.astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return img


def load_mask(path: Optional[str], img_size: int) -> np.ndarray:
    if not isinstance(path, str) or not os.path.exists(path):
        return np.zeros((img_size, img_size), dtype=np.float32)
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".bmp":
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        mask = np.array(Image.open(path).convert("L"))
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.float32)


def augment_oversample(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Oversampling berbasis augmentasi geometris (flip + rotasi 90 derajat
    kelipatan), sesuai Bagian 3.3. Terbukti mengungguli SMOTE/ADASYN karena
    mempertahankan tekstur spasial dan morfologi batas lesi."""
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 0)
        mask = cv2.flip(mask, 0)
    k = np.random.randint(0, 4)
    img = np.rot90(img, k).copy()
    mask = np.rot90(mask, k).copy()
    return img, mask


# ======================================================================
# 4. DATASET LOADERS
# ======================================================================

HAM_LABEL_MAP = {"mel": 0, "nv": 1, "bcc": 2, "akiec": 3, "bkl": 4, "df": 5, "vasc": 6}
ISIC_T3_COLS = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def build_ham10000_dataframe(paths: DatasetPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.ham_csv)
    df["label"] = df["dx"].map(HAM_LABEL_MAP)

    def img_path(img_id):
        p1 = os.path.join(paths.ham_img1, img_id + ".jpg")
        p2 = os.path.join(paths.ham_img2, img_id + ".jpg")
        return p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)

    def mask_path(img_id):
        return os.path.join(paths.ham_mask, img_id + "_segmentation.png")

    df["img_path"] = df["image_id"].apply(img_path)
    df["mask_path"] = df["image_id"].apply(mask_path)
    df = df[df["img_path"].apply(lambda p: p is not None and os.path.exists(p))]
    df = df[df["mask_path"].apply(os.path.exists)].reset_index(drop=True)
    return df


def build_isic_task3_dataframe(paths: DatasetPaths) -> pd.DataFrame:
    """Digunakan hanya untuk evaluasi cross-dataset (tidak untuk training)."""
    df = pd.read_csv(paths.isic_t3_train_csv)
    df["label"] = df[ISIC_T3_COLS].values.argmax(axis=1)
    df["img_path"] = df["image"].apply(
        lambda x: os.path.join(paths.isic_t3_train_img, x + ".jpg")
    )
    df = df[df["img_path"].apply(os.path.exists)].reset_index(drop=True)
    df["mask_path"] = ""
    return df


def build_seg_dataframe(img_dir: str, mask_dir: str,
                         img_ext: str = ".jpg",
                         mask_suffix: str = "_segmentation.png") -> pd.DataFrame:
    """Dataframe generik untuk dataset segmentasi (ISIC Task1/2016/2017)."""
    rows = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith(img_ext):
            continue
        mask_name = fname.replace(img_ext, mask_suffix)
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            rows.append({
                "img_path": os.path.join(img_dir, fname),
                "mask_path": mask_path,
                "label": -1,
            })
    return pd.DataFrame(rows)


def build_ph2_dataframe(paths: DatasetPaths) -> pd.DataFrame:
    """PH2 hanya dipakai sebagai data tambahan segmentasi, label kelas
    tidak digunakan di jalur training utama."""
    records = []
    for img_id in sorted(os.listdir(paths.ph2_root)):
        if not img_id.startswith("IMD"):
            continue
        img_path = os.path.join(paths.ph2_root, img_id, f"{img_id}_Dermoscopic_Image", f"{img_id}.bmp")
        mask_path = os.path.join(paths.ph2_root, img_id, f"{img_id}_lesion", f"{img_id}_lesion.bmp")
        if os.path.exists(img_path) and os.path.exists(mask_path):
            records.append({"img_path": img_path, "mask_path": mask_path, "label": -1})
    return pd.DataFrame(records)


def stratified_ham_split(df: pd.DataFrame, cfg: Config):
    """Split stratified ~70/15/15 dengan seed tetap, di-grup per lesion_id
    agar tidak ada lesion yang muncul di lebih dari satu split.

    REVISI (post-audit): versi sebelumnya (train_test_split biasa, stratified
    by class tapi TIDAK di-grup per lesion) terbukti bocor -- 34.8% gambar
    test dan 36.1% gambar val berbagi lesion_id dengan gambar training,
    karena 1,956 dari 7,470 lesion unik di HAM10000 punya lebih dari satu
    foto (hingga 6). StratifiedGroupKFold menahan constraint kelompok
    (lesion_id) sekaligus menjaga proporsi kelas sedekat mungkin. Rasio
    70/15/15 menjadi ~71.4/14.3/14.3 akibat constraint kelompok; ini
    diverifikasi menghasilkan nol overlap lesion_id lintas split (lihat
    fix_lesion_grouped_split.py)."""
    idx = np.arange(len(df))
    labels = df["label"].values
    groups = df["lesion_id"].values

    sgkf_test = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=cfg.seed)
    train_val_idx, test_idx = next(sgkf_test.split(idx, labels, groups))

    sub_labels = labels[train_val_idx]
    sub_groups = groups[train_val_idx]
    sgkf_val = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=cfg.seed)
    train_sub_idx, val_sub_idx = next(
        sgkf_val.split(np.arange(len(sub_labels)), sub_labels, sub_groups)
    )
    train_idx = train_val_idx[train_sub_idx]
    val_idx = train_val_idx[val_sub_idx]

    train_lesions = set(groups[train_idx])
    val_lesions = set(groups[val_idx])
    test_lesions = set(groups[test_idx])
    assert not (train_lesions & val_lesions), "lesion leakage: train & val"
    assert not (train_lesions & test_lesions), "lesion leakage: train & test"
    assert not (val_lesions & test_lesions), "lesion leakage: val & test"

    return train_idx, val_idx, test_idx


def stratified_ham_kfold(df: pd.DataFrame, cfg: Config, n_splits: int = 5):
    """Proper lesion-grouped K-fold cross-validation: yields (fold_i, train_idx,
    val_idx, test_idx) for EVERY fold, so a model's performance is measured by
    averaging across n_splits genuinely different data partitions -- not by
    retraining the SAME fixed partition with different random seeds.

    Rationale (raised directly by the user): stratified_ham_split() above
    only ever uses fold 0 of a StratifiedGroupKFold(n_splits=7) split,
    discarding the other 6 folds forever. That one ~14.3% test slice could
    be a "lucky" or "unlucky" partition -- we never found out, because we
    never looked at the other 6. Multi-seed testing (this project's earlier
    approach) retrains the SAME lucky-or-unlucky fixed test set repeatedly
    with different initializations -- it addresses TRAINING stochasticity,
    not DATA-PARTITION variance, which is the more fundamental question when
    the dataset has severely underrepresented classes (df=115, vasc=142
    total images). K-fold directly attacks that: across all n_splits folds,
    every single image is used as test exactly once, so the averaged metric
    reflects the WHOLE dataset's difficulty, not one arbitrary slice of it.

    Each fold still carves its own val set out of the fold's train portion
    (for early stopping), using the same StratifiedGroupKFold(n_splits=6)
    approach as stratified_ham_split, and the same zero-leakage assertions.
    """
    idx = np.arange(len(df))
    labels = df["label"].values
    groups = df["lesion_id"].values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    for fold_i, (train_val_idx, test_idx) in enumerate(sgkf.split(idx, labels, groups)):
        sub_labels = labels[train_val_idx]
        sub_groups = groups[train_val_idx]
        sgkf_val = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=cfg.seed)
        train_sub_idx, val_sub_idx = next(
            sgkf_val.split(np.arange(len(sub_labels)), sub_labels, sub_groups)
        )
        train_idx = train_val_idx[train_sub_idx]
        val_idx = train_val_idx[val_sub_idx]

        train_lesions = set(groups[train_idx])
        val_lesions = set(groups[val_idx])
        test_lesions = set(groups[test_idx])
        assert not (train_lesions & val_lesions), f"fold {fold_i}: lesion leakage train & val"
        assert not (train_lesions & test_lesions), f"fold {fold_i}: lesion leakage train & test"
        assert not (val_lesions & test_lesions), f"fold {fold_i}: lesion leakage val & test"

        yield fold_i, train_idx, val_idx, test_idx


# ======================================================================
# 5. TORCH DATASET
# ======================================================================

class SkinLesionDataset(Dataset):
    """Dataset umum untuk gambar + mask + label, dengan preprocessing
    dan augmentasi dilakukan on-the-fly (tidak di-cache ke RAM/disk)."""

    def __init__(self, df: pd.DataFrame, img_size: int,
                 transform=None, oversample: bool = False,
                 target_per_class: Optional[int] = None,
                 num_classes: int = 7):
        self.img_size = img_size
        self.transform = transform

        img_paths = df["img_path"].tolist()
        mask_paths = df["mask_path"].tolist()
        labels = df["label"].tolist()

        self.entries = list(zip(img_paths, mask_paths, labels))

        if oversample and target_per_class is not None:
            self.entries = self._build_oversampled_index(
                self.entries, target_per_class, num_classes
            )

    @staticmethod
    def _build_oversampled_index(entries, target_per_class, num_classes):
        """Membentuk indeks entri dengan augmentasi flag, bukan menyalin
        gambar penuh ke memori. Setiap entri tambahan ditandai augment=True
        agar __getitem__ menerapkan augmentasi oversampling secara on-the-fly."""
        by_class = {c: [] for c in range(num_classes)}
        for i, (img_p, mask_p, label) in enumerate(entries):
            if label is not None and label >= 0:
                by_class[int(label)].append(i)

        expanded = [(p, m, l, False) for (p, m, l) in entries]
        for cls_idx, indices in by_class.items():
            current_n = len(indices)
            needed = target_per_class - current_n
            if needed <= 0 or current_n == 0:
                continue
            for _ in range(needed):
                src_i = np.random.choice(indices)
                p, m, l = entries[src_i]
                expanded.append((p, m, l, True))
        return expanded

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        if len(entry) == 4:
            img_path, mask_path, label, do_aug = entry
        else:
            img_path, mask_path, label = entry
            do_aug = False

        img_np = load_and_preprocess_image(img_path, self.img_size)
        mask_np = load_mask(mask_path, self.img_size)

        if do_aug:
            img_np, mask_np = augment_oversample(img_np, mask_np)

        if label is None or (isinstance(label, float) and np.isnan(label)):
            label = -1

        if self.transform:
            aug = self.transform(image=img_np, mask=mask_np)
            img_t = aug["image"]
            mask_t = aug["mask"].unsqueeze(0).float()
        else:
            img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).float()

        return img_t, mask_t, int(label)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool,
                 train: bool = False, num_workers: int = 4) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        drop_last=train, persistent_workers=(num_workers > 0),
    )


# ======================================================================
# 6. ARSITEKTUR BIRNet
# ======================================================================

class ASPPLite(nn.Module):
    """Modul ASPP ringan yang digunakan pada bottleneck sebelum decoder
    segmentasi, untuk memperluas receptive field tanpa menambah parameter
    secara berlebihan."""

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv_r6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv_r12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.ReLU(inplace=True))
        self.proj = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1))

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        gap_out = F.interpolate(self.gap(x), size=(h, w), mode="bilinear", align_corners=False)
        out = torch.cat([self.conv1x1(x), self.conv_r6(x), self.conv_r12(x), gap_out], dim=1)
        return self.proj(out)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 16):
        super().__init__()
        mid = max(1, in_channels // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., 2018), digunakan
    sebagai head klasifikasi sesuai Bagian 3.4 paper."""

    def __init__(self, in_channels: int, ratio: int = 16):
        super().__init__()
        self.ca = ChannelAttention(in_channels, ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)


class _SobelIrregularityMixin:
    """Ekstraksi irregularity map dari mask segmentasi (Sobel + local
    variance window w, normalisasi min-max), dipakai bersama oleh
    BIGGatingLegacy dan BIGCrossAttention. Bagian ini adalah geometric
    prior yang tetap (non-learnable) dan menjadi dasar operasionalisasi
    komponen Border pada aturan klinis ABCD; dipertahankan di kedua
    varian modul karena juga dipakai untuk visualisasi interpretability
    (Grad-CAM overlay, Fig. 7 paper)."""

    def _make_sobel_buffers(self):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def extract_irregularity(self, seg_mask: torch.Tensor) -> torch.Tensor:
        """Persamaan 2: E = sqrt((Kx * M)^2 + (Ky * M)^2), lalu local
        variance pada window w dan normalisasi min-max ke [0, 1]."""
        gx = F.conv2d(seg_mask, self.sobel_x, padding=1)
        gy = F.conv2d(seg_mask, self.sobel_y, padding=1)
        boundary = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        avg = F.avg_pool2d(boundary, self.window, stride=1, padding=self.padding)
        avg_sq = F.avg_pool2d(boundary ** 2, self.window, stride=1, padding=self.padding)
        local_var = torch.clamp(avg_sq - avg ** 2, min=0)

        B = local_var.shape[0]
        flat = local_var.view(B, -1)
        min_v = flat.min(1)[0].view(B, 1, 1, 1)
        max_v = flat.max(1)[0].view(B, 1, 1, 1)
        return (local_var - min_v) / (max_v - min_v + 1e-8)


class BIGGatingLegacy(nn.Module, _SobelIrregularityMixin):
    """Modul BIG generasi awal (spatial gating elementwise), dipertahankan
    sebagai baseline ablasi (config C5) untuk pembanding langsung terhadap
    BIGCrossAttention (config C6). BUKAN lagi modul utama BIRNet: ini
    secara arsitektural adalah gating, bukan cross-attention beneran (tidak
    ada query-key-value / softmax antar posisi spasial), yang menjadi salah
    satu titik lemah novelty pada draft submission MedIA sebelumnya.

        F'_cls = F_cls (dot) (1 + alpha * I_tilde)"""

    def __init__(self, feat_channels: int, alpha: float = 1.0, window: int = 9):
        super().__init__()
        self.alpha = alpha
        self.window = window
        self.padding = window // 2
        self._make_sobel_buffers()

        self.irr_proj = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, bias=False), nn.Sigmoid())

    def forward(self, cls_feat: torch.Tensor, seg_mask: torch.Tensor):
        h, w = cls_feat.shape[2], cls_feat.shape[3]
        mask_resized = F.interpolate(seg_mask, size=(h, w), mode="bilinear", align_corners=False)

        irregularity = self.extract_irregularity(mask_resized)
        attn_map = self.irr_proj(irregularity)  # proyeksi ke [0,1] per posisi spasial

        modulated = cls_feat * (1.0 + self.alpha * attn_map)
        return modulated, irregularity


class BIGCrossAttention(nn.Module, _SobelIrregularityMixin):
    """Modul cross-attention BIG generasi baru (Boundary Irregularity-
    Guided), inti novelty revisi paper. Berbeda dari BIGGatingLegacy, modul
    ini adalah genuine scaled dot-product multi-head cross-attention: peta
    irregularity geometris (Sobel + local variance, w=9) di-encode menjadi
    Query, sementara classification bottleneck feature menyediakan Key dan
    Value. Informasi dipertukarkan lintas seluruh posisi spasial (bukan
    hanya gating posisi-per-posisi), dengan residual connection berskala
    learnable gamma (init 0, gaya non-local block / SAGAN) agar training
    stabil sejak awal:

        Q = W_q(Encode(I_tilde)),  K = W_k(F_cls),  V = W_v(F_cls)
        Attn = softmax(Q^T K / sqrt(d_head))
        F'_cls = F_cls + gamma * W_o(Attn V)

    Parameter `variant`:
        - "crossattn": Query berasal dari irregularity map asli (peta
          nyata dari mask yang diprediksi).
        - "placebo": Query berasal dari peta nol (tanpa sinyal
          irregularity), dipakai sebagai ablasi kontrol (config C7) untuk
          mengisolasi apakah gain performa berasal dari sinyal boundary
          irregularity itu sendiri, atau hanya dari kapasitas attention
          tambahan yang mirip secara parameter."""

    def __init__(self, feat_channels: int, window: int = 9,
                 attn_dim: int = 64, num_heads: int = 4, variant: str = "crossattn",
                 attn_dropout: float = 0.1, proj_dropout: float = 0.2):
        super().__init__()
        assert variant in ("crossattn", "placebo")
        assert attn_dim % num_heads == 0, "attn_dim harus habis dibagi num_heads"

        self.window = window
        self.padding = window // 2
        self.variant = variant
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self._make_sobel_buffers()

        self.irr_encoder = nn.Sequential(
            nn.Conv2d(1, attn_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, attn_dim), nn.ReLU(inplace=True))

        self.q_proj = nn.Conv2d(attn_dim, attn_dim, 1)
        self.k_proj = nn.Conv2d(feat_channels, attn_dim, 1)
        self.v_proj = nn.Conv2d(feat_channels, attn_dim, 1)
        self.out_proj = nn.Conv2d(attn_dim, feat_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # Regularisasi tambahan: attention branch (Q/K/V/out_proj) menambah
        # kapasitas parameter cukup besar dibanding BIGGatingLegacy. Tanpa
        # dropout, gradien dari cabang ini (walau gamma kecil) berpotensi
        # membuat fitur bottleneck bersama overfit ke HAM10000 train set
        # (gap train_acc~87% vs val_acc~73% teramati pada C6/C7 sebelum
        # perubahan ini). attn_dropout meniru dropout attention-weight gaya
        # Transformer standar; proj_dropout meregularisasi output sebelum
        # digabung lewat residual.
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, cls_feat: torch.Tensor, seg_mask: torch.Tensor):
        B, C, H, W = cls_feat.shape
        N = H * W

        mask_resized = F.interpolate(seg_mask, size=(H, W), mode="bilinear", align_corners=False)
        irregularity = self.extract_irregularity(mask_resized)
        query_source = torch.zeros_like(irregularity) if self.variant == "placebo" else irregularity

        irr_embed = self.irr_encoder(query_source)  # B, attn_dim, H, W

        q = self.q_proj(irr_embed).view(B, self.num_heads, self.head_dim, N)
        k = self.k_proj(cls_feat).view(B, self.num_heads, self.head_dim, N)
        v = self.v_proj(cls_feat).view(B, self.num_heads, self.head_dim, N)

        attn_logits = torch.einsum("bhdn,bhdm->bhnm", q, k) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_out = torch.einsum("bhnm,bhdm->bhdn", attn_weights, v)
        attn_out = attn_out.reshape(B, self.attn_dim, H, W)
        attn_out = self.proj_dropout(self.out_proj(attn_out))

        modulated = cls_feat + self.gamma * attn_out
        return modulated, irregularity


class SegDecoder(nn.Module):
    """Decoder segmentasi bergaya UNet++ (nested skip connections) dengan
    ASPP-lite pada bottleneck, sesuai Bagian 3.4 paper."""

    def __init__(self, skip1_ch: int, skip2_ch: int, bottleneck_ch: int, aspp_out: int = 256):
        super().__init__()
        self.aspp = ASPPLite(bottleneck_ch, aspp_out)
        self.up1 = nn.Sequential(
            nn.Conv2d(aspp_out + skip2_ch, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(
            nn.Conv2d(128 + skip1_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1))

    def forward(self, bottleneck, skip2, skip1):
        x = self.aspp(bottleneck)
        x = F.interpolate(x, size=skip2.shape[2:], mode="bilinear", align_corners=False)
        x = self.up1(torch.cat([x, skip2], dim=1))
        x = F.interpolate(x, size=skip1.shape[2:], mode="bilinear", align_corners=False)
        x = self.up2(torch.cat([x, skip1], dim=1))
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return self.final(x)


BIG_VARIANTS = ("none", "gating", "crossattn", "placebo", "oracle")


class BIRNet(nn.Module):
    """Model BIRNet lengkap: shared EfficientNet-B4 encoder, decoder
    segmentasi UNet++, head klasifikasi CBAM, dan modul BIG yang
    menghubungkan kedua cabang dalam satu forward pass.

    `big_variant` mengontrol varian modul BIG yang dipakai (dipetakan ke
    config ablasi C1-C8 lewat ABLATION_CONFIGS):
        - "none"      : tanpa modul BIG sama sekali (C1-C4)
        - "gating"     : BIGGatingLegacy, modul gating generasi awal (C5)
        - "crossattn"  : BIGCrossAttention, modul cross-attention baru (C6)
        - "placebo"    : BIGCrossAttention dengan Query dikosongkan (C7)
        - "oracle"     : BIGCrossAttention diberi GT mask alih-alih mask
                          prediksi saat forward (C8, hanya diagnostic run,
                          lihat parameter `gt_mask` pada forward())."""

    def __init__(self, num_classes: int = 7, img_size: int = 224,
                 dropout: float = 0.4, big_alpha: float = 1.0,
                 big_window: int = 9, big_variant: str = "crossattn",
                 pretrained: bool = True):
        super().__init__()
        assert big_variant in BIG_VARIANTS, f"big_variant tidak dikenal: {big_variant}"
        self.big_variant = big_variant

        self.encoder = timm.create_model(
            "efficientnet_b4", pretrained=pretrained,
            features_only=True, out_indices=(1, 2, 4))

        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feats = self.encoder(dummy)
        skip1_ch, skip2_ch, bottleneck_ch = (f.shape[1] for f in feats)

        self.seg_decoder = SegDecoder(skip1_ch, skip2_ch, bottleneck_ch)

        if big_variant == "none":
            self.big_attention = None
        elif big_variant == "gating":
            self.big_attention = BIGGatingLegacy(bottleneck_ch, alpha=big_alpha, window=big_window)
        elif big_variant == "placebo":
            self.big_attention = BIGCrossAttention(bottleneck_ch, window=big_window, variant="placebo")
        else:  # "crossattn" atau "oracle" keduanya pakai modul cross-attention penuh
            self.big_attention = BIGCrossAttention(bottleneck_ch, window=big_window, variant="crossattn")

        self.cbam = CBAM(bottleneck_ch)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_ch, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x, gt_mask: Optional[torch.Tensor] = None):
        """`gt_mask` hanya dipakai ketika big_variant=="oracle" (ablasi C8,
        upper-bound diagnostic): BIG menerima mask ground-truth alih-alih
        mask prediksi. Untuk semua varian lain, argumen ini diabaikan.
        Tidak pernah dipakai untuk angka evaluasi final/deployment."""
        skip1, skip2, bottleneck = self.encoder(x)

        seg_logits = self.seg_decoder(bottleneck, skip2, skip1)
        seg_mask = torch.sigmoid(seg_logits)

        if self.big_attention is None:
            cls_feat_guided, irregularity_map = bottleneck, None
        else:
            mask_for_big = seg_mask
            if self.big_variant == "oracle" and gt_mask is not None:
                mask_for_big = gt_mask
            cls_feat_guided, irregularity_map = self.big_attention(bottleneck, mask_for_big)

        cls_feat_guided = self.cbam(cls_feat_guided)
        cls_logits = self.cls_head(cls_feat_guided)

        return seg_logits, cls_logits, seg_mask, irregularity_map


# ======================================================================
# 7. LOSS FUNCTIONS
# ======================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        B = probs.shape[0]
        p, t = probs.view(B, -1), targets.view(B, -1)
        inter = (p * t).sum(1)
        dice = (2 * inter + self.smooth) / (p.sum(1) + t.sum(1) + self.smooth)
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    """Persamaan 6 paper: alpha_T=0.3, beta_T=0.7, memberi penalti lebih
    besar pada false negative (secara klinis lebih berisiko)."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        B = probs.shape[0]
        p, t = probs.view(B, -1), targets.view(B, -1)
        tp = (p * t).sum(1)
        fp = (p * (1 - t)).sum(1)
        fn = ((1 - p) * t).sum(1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class BoundaryLoss(nn.Module):
    """BCE antara boundary Sobel dari mask prediksi dan mask ground truth."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _boundary(self, mask):
        gx = F.conv2d(mask, self.sobel_x, padding=1)
        gy = F.conv2d(mask, self.sobel_y, padding=1)
        return torch.clamp(torch.sqrt(gx ** 2 + gy ** 2 + 1e-8), 0, 1)

    def forward(self, seg_logits, gt_mask):
        pred_mask = torch.sigmoid(seg_logits)
        return F.binary_cross_entropy(self._boundary(pred_mask), self._boundary(gt_mask))


class SegmentationLoss(nn.Module):
    """L_seg = Dice + BCE + 0.5 * Tversky (Persamaan 5)."""

    def __init__(self, tversky_alpha: float = 0.3, tversky_beta: float = 0.7, tversky_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(tversky_alpha, tversky_beta)
        self.tversky_weight = tversky_weight

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.bce(logits, targets) + \
            self.tversky_weight * self.tversky(logits, targets)


def compute_inverse_freq_weights(labels: np.ndarray, num_classes: int = 7) -> torch.Tensor:
    """Inverse-frequency class weights (normalized to mean=1) from the
    TRAINING portion of a fold, for LabelSmoothingFocalLoss's `weight` arg
    -- previously always None despite being fully wired into the loss.
    Rationale: macro-F1 (unlike balanced accuracy) penalizes precision loss
    from a class being over-predicted at another's expense (diagnosed in
    birnet_diagnose_pooled_bcn.py); down-weighting the loss contribution of
    already-abundant classes (nv, mel) relative to rare ones (df, vasc)
    discourages that over-prediction pattern instead of just measuring it
    after the fact."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1)  # avoid div-by-zero for any missing class in a fold
    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.mean()  # normalize so mean weight = 1 (keeps loss scale comparable)
    return torch.tensor(weights, dtype=torch.float32)


class LabelSmoothingFocalLoss(nn.Module):
    """L_cls (Persamaan 1): focal loss dengan label smoothing,
    gamma=2.0, epsilon=0.1."""

    def __init__(self, gamma: float = 2.0, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits, targets):
        n_cls = logits.size(1)
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.smoothing / (n_cls - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_prob = F.log_softmax(logits, dim=1)
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal_weight = (1 - pt) ** self.gamma

        loss = -(smooth_targets * log_prob).sum(dim=1)
        return (focal_weight * loss).mean()


class BIRNetJointLoss(nn.Module):
    """L_total = lambda_seg * L_seg + lambda_cls * L_cls + lambda_bnd * L_boundary
    (Persamaan 4 paper: 0.40 / 0.50 / 0.10)."""

    def __init__(self, cfg: Config, cls_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.lambda_seg = cfg.lambda_seg
        self.lambda_cls = cfg.lambda_cls
        self.lambda_bnd = cfg.lambda_boundary
        self.use_boundary_loss = cfg.use_boundary_loss

        self.seg_loss = SegmentationLoss(cfg.tversky_alpha, cfg.tversky_beta)
        self.boundary_loss = BoundaryLoss()
        smoothing = cfg.label_smoothing if cfg.use_label_smoothing else 0.0
        self.cls_loss = LabelSmoothingFocalLoss(
            gamma=cfg.focal_gamma, smoothing=smoothing, weight=cls_weight
        )

    def forward(self, seg_logits, cls_logits, gt_mask, gt_label,
                has_mask: bool = True, has_label: bool = True):
        loss_dict = {}
        total = seg_logits.sum() * 0.0  # anchor ke computation graph

        if has_mask:
            l_seg = self.seg_loss(seg_logits, gt_mask)
            loss_dict["seg"] = l_seg.item()
            total = total + self.lambda_seg * l_seg
            if self.use_boundary_loss:
                l_bnd = self.boundary_loss(seg_logits, gt_mask)
                loss_dict["boundary"] = l_bnd.item()
                total = total + self.lambda_bnd * l_bnd

        if has_label:
            valid = gt_label >= 0
            if valid.sum() > 0:
                l_cls = self.cls_loss(cls_logits[valid], gt_label[valid])
                loss_dict["cls"] = l_cls.item()
                total = total + self.lambda_cls * l_cls

        loss_dict["total"] = total.item()
        return total, loss_dict


# ======================================================================
# 8. METRIK EVALUASI
# ======================================================================

def dice_score(logits: torch.Tensor, gt: torch.Tensor, thr: float = 0.5, smooth: float = 1.0) -> float:
    pred = (torch.sigmoid(logits) > thr).float()
    B = pred.shape[0]
    p, g = pred.view(B, -1), gt.view(B, -1)
    inter = (p * g).sum(1)
    return ((2 * inter + smooth) / (p.sum(1) + g.sum(1) + smooth)).mean().item()


def iou_score(logits: torch.Tensor, gt: torch.Tensor, thr: float = 0.5, smooth: float = 1.0) -> float:
    pred = (torch.sigmoid(logits) > thr).float()
    B = pred.shape[0]
    p, g = pred.view(B, -1), gt.view(B, -1)
    inter = (p * g).sum(1)
    union = p.sum(1) + g.sum(1) - inter
    return ((inter + smooth) / (union + smooth)).mean().item()


def cls_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    valid = labels >= 0
    if valid.sum() == 0:
        return 0.0
    return (logits[valid].argmax(1) == labels[valid]).float().mean().item()


@torch.no_grad()
def evaluate_classification(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    preds, labs = [], []
    for imgs, _, labels in loader:
        imgs = imgs.to(device)
        _, cls_logits, _, _ = model(imgs)
        valid = labels >= 0
        preds.extend(cls_logits[valid].argmax(1).cpu().numpy())
        labs.extend(labels[valid].numpy())
    preds, labs = np.array(preds), np.array(labs)
    acc = (preds == labs).mean() if len(labs) else 0.0
    bal_acc = balanced_accuracy_score(labs, preds) if len(labs) else 0.0
    return float(acc), float(bal_acc), preds, labs


@torch.no_grad()
def evaluate_segmentation(model: nn.Module, loader: DataLoader, device: str, thr: float = 0.5):
    model.eval()
    total_dice = total_iou = n = 0
    for imgs, masks, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        seg_logits, _, _, _ = model(imgs)
        total_dice += dice_score(seg_logits, masks, thr)
        total_iou += iou_score(seg_logits, masks, thr)
        n += 1
    n = max(n, 1)
    return {"dice": total_dice / n, "iou": total_iou / n}


def find_best_threshold(model: nn.Module, loader: DataLoader, device: str,
                         thr_range=np.arange(0.30, 0.81, 0.02)) -> Tuple[float, float]:
    """Grid search threshold optimal pada validation set (paper: tau*=0.78)."""
    model.eval()
    all_probs, all_masks = [], []
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs = imgs.to(device)
            _, _, seg_probs, _ = model(imgs)
            all_probs.append(seg_probs.cpu().numpy())
            all_masks.append(masks.numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    best_thr, best_dice = 0.5, 0.0
    for thr in thr_range:
        pred = (all_probs > thr).astype(np.float32)
        B = pred.shape[0]
        p, g = pred.reshape(B, -1), all_masks.reshape(B, -1)
        inter = (p * g).sum(1)
        dice = ((2 * inter + 1) / (p.sum(1) + g.sum(1) + 1)).mean()
        if dice > best_dice:
            best_dice, best_thr = dice, float(thr)
    return best_thr, float(best_dice)


# ======================================================================
# 9. CHECKPOINT MANAGER
# ======================================================================

class CheckpointManager:
    def __init__(self, ckpt_dir: str, model: nn.Module,
                 optimizer: optim.Optimizer, scheduler):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.best_path = os.path.join(ckpt_dir, "birnet_best.pth")
        self.last_path = os.path.join(ckpt_dir, "birnet_last.pth")

    def save(self, epoch: int, history: dict, best_score: float,
             val_dice: float, val_bal_acc: float, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_score": best_score,
            "val_dice": val_dice,
            "val_acc": val_bal_acc,
            "history": history,
        }
        torch.save(state, self.last_path)
        if is_best:
            torch.save(state, self.best_path)

    def load(self, device: str, mode: str = "best"):
        path = self.best_path if mode == "best" else self.last_path
        if not os.path.exists(path):
            print("Checkpoint tidak ditemukan, memulai training dari awal.")
            return None
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        print(f"Resume dari epoch={ckpt['epoch']} score={ckpt['best_score']:.4f}")
        return ckpt


# ======================================================================
# 10. TRAINING LOOP
# ======================================================================

def train_one_epoch(model: nn.Module, loader: DataLoader,
                     optimizer: optim.Optimizer, criterion: BIRNetJointLoss,
                     device: str, big_variant: str = "crossattn") -> Dict[str, float]:
    """`big_variant` dicek untuk kasus "oracle" (config ablasi C8): pada
    varian ini saja, GT mask ikut dilewatkan ke forward() model supaya
    modul BIG memakai boundary GT alih-alih mask prediksi. Untuk varian
    lain tidak berpengaruh (gt_mask diabaikan oleh BIRNet.forward)."""
    model.train()
    total_loss = total_dice = total_iou = total_acc = n = 0
    for imgs, masks, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        if big_variant == "oracle":
            seg_logits, cls_logits, _, _ = model(imgs, gt_mask=masks)
        else:
            seg_logits, cls_logits, _, _ = model(imgs)
        loss, loss_dict = criterion(seg_logits, cls_logits, masks, labels,
                                     has_mask=True, has_label=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss_dict["total"]
        total_dice += dice_score(seg_logits, masks)
        total_iou += iou_score(seg_logits, masks)
        total_acc += cls_accuracy(cls_logits, labels)
        n += 1

    n = max(n, 1)
    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "iou": total_iou / n,
        "acc": total_acc / n,
    }


def compute_class_weights(df: pd.DataFrame, num_classes: int, device: str) -> torch.Tensor:
    """Inverse-frequency class weight (dinormalisasi ke jumlah=num_classes),
    digunakan sebagai bobot tambahan pada focal loss."""
    counts = np.array([int((df["label"] == i).sum()) for i in range(num_classes)], dtype=np.float32)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / (counts / counts.sum())
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_birnet(cfg: Config, model: nn.Module,
                  train_loader: DataLoader, val_loader: DataLoader,
                  seg_val_loader: DataLoader, cls_weight: torch.Tensor):
    device = cfg.device
    criterion = BIRNetJointLoss(cfg, cls_weight=cls_weight).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.cls_lr, weight_decay=cfg.cls_weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.scheduler_t0, eta_min=cfg.scheduler_eta_min)
    ckpt_manager = CheckpointManager(cfg.ckpt_dir, model, optimizer, scheduler)

    history = {
        "train_loss": [], "train_dice": [], "train_acc": [],
        "val_bal_acc": [], "val_acc": [], "val_seg_dice": [],
        "lr": [], "score": [],
    }
    best_score, no_improve = 0.0, 0

    print(f"Mulai training BIRNet | max_epochs={cfg.cls_max_epochs} | patience={cfg.cls_patience}")
    print("=" * 80)

    for epoch in range(1, cfg.cls_max_epochs + 1):
        set_seed(cfg.seed + epoch)
        t0 = time.time()

        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                       big_variant=cfg.big_variant)
        val_acc, val_bal, _, _ = evaluate_classification(model, val_loader, device)
        seg_stats = evaluate_segmentation(model, seg_val_loader, device, thr=0.5)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Skor gabungan: rata-rata balanced accuracy klasifikasi dan Dice segmentasi
        score = 0.5 * val_bal + 0.5 * seg_stats["dice"]

        for key, val in [
            ("train_loss", train_stats["loss"]), ("train_dice", train_stats["dice"]),
            ("train_acc", train_stats["acc"]), ("val_bal_acc", val_bal),
            ("val_acc", val_acc), ("val_seg_dice", seg_stats["dice"]),
            ("lr", lr), ("score", score),
        ]:
            history[key].append(val)

        is_best = score > best_score
        if is_best:
            best_score, no_improve, flag = score, 0, "*"
        else:
            no_improve += 1
            flag = ""

        ckpt_manager.save(epoch, history, best_score, seg_stats["dice"], val_bal, is_best)
        with open(os.path.join(cfg.log_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        print(f"Ep {epoch:3d}/{cfg.cls_max_epochs} "
              f"| train_loss={train_stats['loss']:.4f} train_dice={train_stats['dice']:.4f} "
              f"train_acc={train_stats['acc']:.4f} "
              f"| val_bal={val_bal:.4f} val_acc={val_acc:.4f} val_seg_dice={seg_stats['dice']:.4f} "
              f"| score={score:.4f} | lr={lr:.2e} | {elapsed:.0f}s {flag}")

        if no_improve >= cfg.cls_patience:
            print(f"Early stopping pada epoch {epoch}.")
            break

    return history


# ======================================================================
# 11. FINAL EVALUATION
# ======================================================================

def final_evaluation(cfg: Config, model: nn.Module,
                      test_loader: DataLoader,
                      seg_val_loader: DataLoader, seg_test_loader: DataLoader,
                      cross_dataset_loader: Optional[DataLoader] = None):
    device = cfg.device

    ckpt_manager = CheckpointManager(cfg.ckpt_dir, model,
                                      optim.AdamW(model.parameters()),
                                      CosineAnnealingWarmRestarts(
                                          optim.AdamW(model.parameters()), T_0=cfg.scheduler_t0))
    ckpt = torch.load(ckpt_manager.best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    # Threshold optimal untuk segmentasi (paper: tau*=0.78)
    best_thr, best_val_dice = find_best_threshold(model, seg_val_loader, device)

    t_acc, t_bal, t_preds, t_labs = evaluate_classification(model, test_loader, device)
    seg_test = evaluate_segmentation(model, seg_test_loader, device, thr=best_thr)

    print("\n" + "=" * 60)
    print("HASIL AKHIR BIRNet")
    print("=" * 60)
    print(f"[Klasifikasi - HAM10000 test]")
    print(f"  Akurasi           : {t_acc:.4f} ({t_acc * 100:.2f}%)")
    print(f"  Balanced Accuracy : {t_bal:.4f} ({t_bal * 100:.2f}%)")
    print(f"\n[Segmentasi - ISIC 2018 Task1 test]")
    print(f"  Threshold optimal : {best_thr:.2f} (val Dice={best_val_dice:.4f})")
    print(f"  Dice              : {seg_test['dice']:.4f}")
    print(f"  IoU               : {seg_test['iou']:.4f}")

    if cross_dataset_loader is not None:
        c_acc, c_bal, _, _ = evaluate_classification(model, cross_dataset_loader, device)
        print(f"\n[Cross-dataset - ISIC 2018 Task3]")
        print(f"  Akurasi           : {c_acc:.4f} ({c_acc * 100:.2f}%)")
        print(f"  Balanced Accuracy : {c_bal:.4f} ({c_bal * 100:.2f}%)")

    print(f"\n[Per-class - HAM10000 test]")
    for i, cls_name in enumerate(cfg.class_names):
        mask = t_labs == i
        if mask.sum() == 0:
            continue
        c_acc = (t_preds[mask] == t_labs[mask]).mean()
        print(f"  {cls_name:8s}: {c_acc:.4f} (n={mask.sum()})")


# ======================================================================
# 12. MAIN PIPELINE
# ======================================================================

def build_datasets(cfg: Config, paths: DatasetPaths):
    train_aug, val_aug = build_augmentations(cfg.img_size)

    ham_df = build_ham10000_dataframe(paths)
    train_idx, val_idx, test_idx = stratified_ham_split(ham_df, cfg)
    ham_train_df = ham_df.iloc[train_idx].reset_index(drop=True)
    ham_val_df = ham_df.iloc[val_idx].reset_index(drop=True)
    ham_test_df = ham_df.iloc[test_idx].reset_index(drop=True)

    isic_t1_train_df = build_seg_dataframe(paths.isic_t1_train_img, paths.isic_t1_train_mask)
    isic_t1_val_df = build_seg_dataframe(paths.isic_t1_val_img, paths.isic_t1_val_mask)
    isic_t1_test_df = build_seg_dataframe(paths.isic_t1_test_img, paths.isic_t1_test_mask)

    isic16_df = build_seg_dataframe(paths.isic16_img, paths.isic16_mask, mask_suffix="_Segmentation.png")
    isic17_df = build_seg_dataframe(paths.isic17_img, paths.isic17_mask, mask_suffix="_segmentation.png")
    ph2_df = build_ph2_dataframe(paths)

    isic_t3_df = build_isic_task3_dataframe(paths)  # evaluasi cross-dataset saja

    # Dataset training gabungan: HAM10000 (dengan oversampling) + ISIC Task1
    # Oversampling hanya diterapkan pada partisi HAM train yang punya label
    # kelas (Bagian 3.3), ISIC Task1 tidak memiliki label kelas sehingga
    # tidak dioversample.
    joint_train_df = pd.concat([ham_train_df, isic_t1_train_df], ignore_index=True)

    train_ds = SkinLesionDataset(
        joint_train_df, cfg.img_size, transform=train_aug,
        oversample=cfg.use_oversampling, target_per_class=cfg.oversample_target_per_class,
        num_classes=cfg.num_classes,
    )
    ham_val_ds = SkinLesionDataset(ham_val_df, cfg.img_size, transform=val_aug)
    ham_test_ds = SkinLesionDataset(ham_test_df, cfg.img_size, transform=val_aug)

    seg_val_ds = SkinLesionDataset(isic_t1_val_df, cfg.img_size, transform=val_aug)
    seg_test_ds = SkinLesionDataset(isic_t1_test_df, cfg.img_size, transform=val_aug)

    cross_ds = SkinLesionDataset(isic_t3_df, cfg.img_size, transform=val_aug)

    # Data segmentasi tambahan (ISIC 2016/2017/PH2) digabung ke training
    # penuh apabila diperlukan pelatihan segmentasi lanjutan yang lebih
    # besar; disediakan di sini agar dapat dipakai sesuai kebutuhan.
    extra_seg_df = pd.concat([isic16_df, isic17_df, ph2_df], ignore_index=True)
    extra_seg_ds = SkinLesionDataset(extra_seg_df, cfg.img_size, transform=train_aug) \
        if len(extra_seg_df) > 0 else None

    loaders = {
        "train": make_loader(train_ds, cfg.cls_batch_size, shuffle=True, train=True),
        "ham_val": make_loader(ham_val_ds, cfg.cls_batch_size, shuffle=False),
        "ham_test": make_loader(ham_test_ds, cfg.cls_batch_size, shuffle=False),
        "seg_val": make_loader(seg_val_ds, cfg.seg_batch_size, shuffle=False),
        "seg_test": make_loader(seg_test_ds, cfg.seg_batch_size, shuffle=False),
        "cross": make_loader(cross_ds, cfg.cls_batch_size, shuffle=False),
    }
    if extra_seg_ds is not None:
        loaders["extra_seg"] = make_loader(extra_seg_ds, cfg.seg_batch_size, shuffle=True, train=True)

    cls_weight = compute_class_weights(ham_train_df, cfg.num_classes, cfg.device)

    return loaders, cls_weight, ham_train_df


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Training/ablasi BIRNet (C1-C8).")
    parser.add_argument("--config", default="C6", choices=list(ABLATION_CONFIGS),
                         help="Config ablasi yang dijalankan (default: C6, cross-attention penuh).")
    parser.add_argument("--base_dir", default="./BIRNet_Research",
                         help="Path ke folder BIRNet_Research berisi dataset/.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(base_dir=args.base_dir)
    apply_ablation_config(cfg, args.config)
    set_seed(cfg.seed)

    print(f"Config ablasi  : {cfg.config_name} -> {ABLATION_CONFIGS[cfg.config_name]}")
    print(f"Device : {cfg.device}")
    if cfg.device == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    paths = DatasetPaths(cfg)
    loaders, cls_weight, ham_train_df = build_datasets(cfg, paths)

    print("\nDistribusi kelas HAM10000 (train, sebelum oversampling):")
    counts = Counter(ham_train_df["label"].tolist())
    for i, cls_name in enumerate(cfg.class_names):
        print(f"  {cls_name:8s}: {counts.get(i, 0)}")

    model = BIRNet(
        num_classes=cfg.num_classes, img_size=cfg.img_size,
        dropout=cfg.dropout, big_alpha=cfg.big_alpha, big_window=cfg.big_window,
        big_variant=cfg.big_variant, pretrained=True,
    ).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameter BIRNet: {total_params / 1e6:.2f}M")

    train_birnet(
        cfg, model,
        train_loader=loaders["train"],
        val_loader=loaders["ham_val"],
        seg_val_loader=loaders["seg_val"],
        cls_weight=cls_weight,
    )

    final_evaluation(
        cfg, model,
        test_loader=loaders["ham_test"],
        seg_val_loader=loaders["seg_val"],
        seg_test_loader=loaders["seg_test"],
        cross_dataset_loader=loaders["cross"],
    )


if __name__ == "__main__":
    main()
