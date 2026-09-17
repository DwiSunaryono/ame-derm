# AME-Derm

Code accompanying the manuscript:

> **AME-Derm: An Asymmetric Anatomical Mixture-of-Experts with a Data-Driven
> Rare-Class Verification Algorithm for Dermoscopic Skin-Lesion Classification**
> Dwi Sunaryono, Brama Seta Putra Satria, Misbakhul Munir Irfan Subakti,
> Laras Suciningtyas, Shoffi Izza Sabilla — Institut Teknologi Sepuluh
> Nopember, Surabaya, Indonesia. Submitted to *Biomedical Signal Processing
> and Control*.

This repository contains the scripts used to produce the results reported
in the paper: lesion segmentation, the asymmetric anatomical
mixture-of-experts (MoE) classifier with static fusion and test-time
augmentation, the rare-class verification cascade, external validation on
BCN20000/ISIC2019, and the matched-protocol comparison against the PanDerm
foundation model.

This is a research codebase extracted from a larger iterative-experiment
repository, kept close to its original form so behavior matches what was
run. It is organized and documented here for transparency and
reproducibility, not repackaged as a polished library.

## Pipeline overview

Scripts are flat (no packages) and import each other by module name
(`import birnet_final as bf`, etc.), so run everything from the repo root
with this directory on `PYTHONPATH` (or just `cd` into it).

| Stage | Scripts | Purpose |
|---|---|---|
| **Shared infrastructure** | `birnet_final.py` | `Config`, `DatasetPaths`, dataframe builders, augmentation pipeline, deterministic-seeding helper. Re-used by nearly every script below. Its `BIRNet`/`BIG`/ASPP/CBAM model classes are an earlier single-model architecture from a previous project phase and are **not** the architecture reported in the current paper — kept only because downstream scripts import its data/config utilities. |
| **Data splitting** | `birnet_representative_val_split.py`, `birnet_export_representative_folds.py`, `birnet_audit_split_representativeness.py`, `birnet_diagnose_metadata_imbalance.py`, `fix_lesion_grouped_split.py` | Lesion-grouped, class- and metadata-stratified 5-fold splitting (`StratifiedGroupKFold` on `lesion_id`) and the audit that verifies zero lesion leakage across train/val/test and balanced coverage of label/dataset/sex/localization/age. |
| **Segmentation (Phase 1)** | `birnet_seg_phase1_final.py`, `birnet_seg_mitb3_combo.py`, `birnet_seg_convnext_upernet.py`, `birnet_seg_bgff.py`, `birnet_seg_bgff_v2.py`, `birnet_seg_highres.py`, `birnet_seg_foundation_guided_filter.py`, `birnet_seg_full_ensemble_map.py`, `birnet_verify_seg_foundation_multiseed.py`, `birnet_generate_phase1_crops.py` | Final locked segmentation ensemble (SegFormer/MiT-B3 + UPerNet/ConvNeXt + UNet++/ResNet50, 4-view TTA, guided filter, largest-connected-component) used to crop every lesion image before classification. |
| **Expert / generalist classifiers** | `train_classifier_generic.py`, `train_classifier_phase2_final.py`, `train_classifier_phase2_mixup.py`, `train_specialist_expert.py`, `train_moe_smart_router.py` | DINOv2 backbone training for the generalist and the three anatomical-zone experts (torso/limb/head), including the metadata cross-attention fusion head. |
| **MoE fusion & main ablation ladder** | `birnet_eval_moe_pure_champion.py`, `birnet_eval_moe_champion_5fold.py`, `birnet_eval_adaptive_gater.py` | Static scalar-α fusion + 6-view TTA council evaluation; the adaptive-gater script reproduces the "static routing vs. learnable gater" ablation. |
| **Rare-class verification cascade** | `_dump_bcn_preds.py`, `_vlm_cascade_best.py`, `_vlm_proto_eval.py`, `_vlm_recall_recovery.py`, `_vlm_bcn_vasc_verify.py` | Builds prototype/candidate montages for the VLM verification step. **These scripts do not call any model API** — they prepare images and expect a human (or an interactive chat-based VLM session) to supply the verdicts, exactly as described as a limitation in the paper (a fully automated LoRA-adapted verifier is future work, not part of this release). |
| **External validation (BCN20000 / ISIC2019)** | `birnet_retrain_bcn.py`, `birnet_retrain_bcn_lpft.py`, `birnet_retrain_bcn_clean_crop.py`, `birnet_diagnose_bcn_f1.py`, `birnet_external_zero_shot.py`, `birnet_external_fewshot_router.py`, `birnet_prepare_isic2019_external.py`, `birnet_generate_isic2019_crops.py`, `birnet_pool_external_rare.py`, `birnet_devignette_crop_bcn.py`, `birnet_prep_bcn_clahe.py`, `birnet_prep_bcn_crop_clahe.py`, `birnet_unsharp_crop_bcn.py` | Zero-shot/few-shot/retrain protocols on the independent BCN20000 cohort, external-rare-class pooling from ISIC2019, and the preprocessing-lever ablations (de-vignetting, CLAHE, unsharp masking). |
| **PanDerm comparison** | `birnet_dinov2_official_protocol.py`, `birnet_moe_official_eval.py`, `birnet_panderm_moe_prep.py`, `birnet_panderm_moe_eval.py` | Matched-protocol comparison against the PanDerm dermatology foundation model (same folds, same official fine-tuning recipe). Requires the separate [PanDerm](https://github.com/SiyuanYan1/PanDerm) repository and its pretrained weights. Set `AME_DERM_ROOT` if running these outside this repo's directory. |

## Data

Datasets are not redistributed here. The paper uses:
- **HAM10000** ([ISIC Archive](https://api.isic-archive.com/collections/212/))
- **BCN20000** ([ISIC Archive](https://api.isic-archive.com/collections/249/))
- **ISIC 2019** ([challenge site](https://challenge.isic-archive.com/data/#2019))
- **PH2** dataset

Expected local directory names (`BIRNet_Research/...`, `panderm_datasets/...`)
are visible as constants near the top of each script; point them at your own
copies of the above datasets.

## Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ recommended. GPU with CUDA strongly recommended for training.

## Reproducibility notes

- `birnet_representative_val_split.py` fixes a lesion-leakage issue present
  in an earlier split (plain `train_test_split`, not grouped by lesion):
  1,956 of 7,470 unique HAM10000 lesions have more than one photo, which had
  let the same lesion appear in more than one split. All reported numbers
  in the paper use the lesion-grouped, `StratifiedGroupKFold`-based split.
- `set_deterministic()` in `birnet_final.py` fixes seeds across
  `random`/`numpy`/`torch`/cuDNN and forces deterministic algorithms; pass
  the same `--seed` through `build_augmentations(seed=...)` for fully
  reproducible runs (albumentations keeps its own internal RNG otherwise).
- This is a research codebase distilled from a much larger set of
  intermediate experiment scripts. It reflects our best-effort
  identification of the exact scripts/configuration that produced each
  reported number; a small number of secondary ablation values (noted in
  the manuscript) were reproduced from equivalent re-runs rather than
  recovered byte-for-byte from archived logs.

## License

MIT — see [LICENSE](LICENSE).

## Citation

Citation details will be added once the manuscript is accepted.
