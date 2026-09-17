# System Documentation

This document walks through the pipeline in the order it runs, mapping each
stage to the paper section that describes it and the package that implements
it. For dataset layout, see [DATA_PREPARATION.md](DATA_PREPARATION.md). For a
numbered, runnable version of this same order, see
[`pipeline/`](../pipeline) at the repository root; for the raw outputs and
regenerable tables referenced throughout, see [`results/`](../results).

## 1. Segmentation (`ame_derm/segmentation/`)

A lesion-segmentation ensemble crops every dermoscopic image before
classification (paper §"Lesion Segmentation Stage").

- `seg_phase1_final.py` — the locked-in final ensemble: SegFormer/MiT-B3 +
  UPerNet/ConvNeXt + UNet++/ResNet50 @ 384px, 4-view TTA, guided filter,
  largest-connected-component post-processing. `Phase1SegmentationPipeline`
  is imported by `generate_phase1_crops.py` and
  `external_validation/generate_isic2019_crops.py`.
- `seg_mitb3_combo.py`, `seg_convnext_upernet.py` — training scripts for the
  two strongest individual encoder/decoder combinations in the ensemble.
- `seg_bgff.py` / `seg_bgff_v2.py` / `seg_highres.py` /
  `seg_foundation_guided_filter.py` — supporting architectures (boundary-
  guided feature fusion U-Net, high-resolution variant) and the guided-filter
  post-processing step used by the final ensemble.
- `seg_full_ensemble_map.py` — the encoder x decoder search that produced the
  per-configuration Dice scores reported in the segmentation ablation table.
- `verify_seg_foundation_multiseed.py` — multi-seed (42/43/44) Dice stability
  check.
- `generate_phase1_crops.py` — runs the locked ensemble over the full dataset
  and writes crops to `config.CROP_DIR`.

## 2. Data splitting (`ame_derm/data_splits/`)

All results use a lesion-grouped, class- and metadata-stratified 5-fold split
built on `lesion_id`, not a plain per-image split.

- `representative_val_split.py` — `stratified_ham_kfold_fully_representative`:
  `StratifiedGroupKFold` on `lesion_id`, then a coverage check across
  label/dataset/sex/localization/age buckets.
- `export_representative_folds.py` — exports the fold CSVs consumed by every
  downstream training/eval script (and by PanDerm's own
  `run_class_finetuning --csv_path` for the matched-protocol comparison).
- `audit_split_representativeness.py` — the audit that verifies **zero**
  lesion overlap across train/val/test and reports per-field coverage
  (mean/max absolute deviation from the global class distribution).
- `diagnose_metadata_imbalance.py` — utility used by the splitter/audit to
  detect which metadata fields are present/usable for stratification.
- `fix_lesion_grouped_split.py` — standalone script documenting the leakage
  fix (see "Reproducibility notes" below).

### Why grouping by lesion matters

An earlier version of the split used `sklearn.model_selection.train_test_split`
stratified by class only, **not** grouped by lesion. 1,956 of the 7,470
unique HAM10000 lesions have more than one photograph (up to 6), so the same
lesion could appear in both the training set and the test/val set — a direct
label leak. Auditing that split found **34.8%** of test images and **36.1%**
of val images shared a `lesion_id` with a training image. `ame_derm.common
.birnet_final.stratified_ham_split` and `representative_val_split.py` both
now use `StratifiedGroupKFold` grouped on `lesion_id`, which changes the
70/15/15 target ratio to roughly 71.4/14.3/14.3 (a consequence of the group
constraint) and eliminates the overlap entirely.

## 3. Expert / generalist classifiers (`ame_derm/experts/`)

DINOv2-based classifiers trained per the paper's "Asymmetric Anatomical MoE"
design: one generalist over the whole body, plus one expert each for
torso/limb/head anatomical zones (paper §"Anatomical Partitioning and
Experts").

- `train_classifier_generic.py` — shared `GenericClassifier` base and
  full-loader evaluation helper.
- `train_classifier_phase2_final.py` — precomputed-crop dataset + training
  loop used by the generalist.
- `train_classifier_phase2_mixup.py` — adds MixUp/CutMix and the weighted
  soft-target loss.
- `train_specialist_expert.py` — `MultimodalClassifier`,
  `CrossAttentionFusion` (image features attend over a localization-metadata
  embedding), `EXPERT_MAPPING` (zone assignment), and the per-zone training
  loop.
- `train_moe_smart_router.py` — a learnable router baseline, used as the
  comparison point against static fusion (see below).

## 4. MoE fusion & the main ablation ladder (`ame_derm/moe_fusion/`)

Paper §"Static Fusion and Test-Time Augmentation" and the main results table.

- `eval_moe_pure_champion.py` — shared evaluation primitives (`tta_views`
  [6-view TTA: identity, h-flip, v-flip, 3 rotations], `extract_probs`,
  `build_exp_arrays`, `get_expert_name`) reused by nearly every later
  eval/external-validation script. Implements the static council fusion
  (fixed scalar weighting between the routed expert and the generalist).
- `eval_moe_champion_5fold.py` — the 5-fold evaluation producing the
  headline generalist-only vs. symmetric-MoE numbers.
- `eval_adaptive_gater.py` — three-way comparison: symmetric static MoE vs.
  the asymmetric (large generalist + base experts) static MoE vs. a trained
  `AdaptiveDualGater` MLP router. This is the source of the paper's argument
  that static routing should be preferred over a learnable gater.

## 5. Rare-class verification cascade (`ame_derm/vlm_cascade/`)

Paper §"Rare-Class Verification Cascade" / "Data-Driven Discovery of
VLM-Amenable Rare Classes".

**These scripts do not call any model API.** They build montage images of
candidate crops with hidden ground-truth keys and expect a human (or an
interactive chat-based VLM session) to supply verdicts as a
`--reject_idx`-style argument to an "apply" step. A fully automated,
LoRA-adapted verifier that removes this human-in-the-loop step is discussed
in the paper as future work, not something this release runs end-to-end.

- `dump_bcn_preds.py` — dumps per-sample BCN predictions aligned to the
  lesion-grouped seed-42 split; this is the input the cascade scripts verify.
- `vlm_proto_eval.py`, `vlm_cascade_best.py`, `vlm_recall_recovery.py`,
  `vlm_bcn_vasc_verify.py` — build the candidate montages and apply cascade
  verdicts for different rare-class scenarios explored in the paper.

## 6. External validation (`ame_derm/external_validation/`)

Paper §"External Validation on an Independent Cohort".

- `retrain_bcn.py` — full retrain of the generalist + all three experts on
  BCN20000 under the same lesion-grouped, matched-seed protocol as the
  internal HAM10000 experiments.
- `retrain_bcn_lpft.py`, `retrain_bcn_clean_crop.py` — linear-probe-then-
  fine-tune and clean-crop retrain variants.
- `external_zero_shot.py` / `external_fewshot_router.py` — zero-shot and
  few-shot baselines (model never trained on BCN/MSK) used as the reference
  point in the retrain comparison.
- `diagnose_bcn_f1.py` — per-class confusion/confidence diagnostics on the
  retrained model (source of the "vascular lesions over-predicted" finding
  that motivates the verification cascade).
- `devignette_crop_bcn.py`, `prep_bcn_clahe.py`, `prep_bcn_crop_clahe.py`,
  `unsharp_crop_bcn.py` — preprocessing-lever ablations on the BCN cohort
  (de-vignetting + tight crop, CLAHE color normalization, unsharp masking).
- `prepare_isic2019_external.py`, `generate_isic2019_crops.py`,
  `pool_external_rare.py` — pool additional rare-class examples from
  ISIC2019/MSKCC/HIBA into the BCN retraining set.

## 7. PanDerm comparison (`baselines/panderm_comparison/`)

Paper §"Comparison with a Dermatology Foundation Model Under a Matched
Protocol". Requires the separate [PanDerm](https://github.com/SiyuanYan1/PanDerm)
repository and its pretrained weights; `config.AME_DERM_ROOT` (or the
`AME_DERM_ROOT` environment variable) tells these scripts where to find the
shared fold CSVs and checkpoints.

- `dinov2_official_protocol.py` — replicates PanDerm's exact official
  fine-tuning recipe (layer-decay 0.65, mixup 0.8/cutmix 1.0, label
  smoothing 0.1, 50 epochs) on identical data/splits, isolating the
  backbone comparison from protocol differences.
- `panderm_moe_prep.py` / `panderm_moe_eval.py` — build and evaluate a
  PanDerm-backboned MoE (large generalist + base experts) with the same
  static-router logic as `moe_fusion/`, filling in the PanDerm-MoE cell of
  the 2x2 backbone x architecture comparison table.
- `moe_official_eval.py` — the parallel official-protocol MoE evaluation for
  the DINOv2 backbone.

## 8. Statistical significance (`evaluation/statistical_significance.py`)

Paper §"Statistical Significance". Reproduces the one-sided paired test of
whether AME-Derm improves on the symmetric MoE across the same five folds:
a paired t-test and a Wilcoxon signed-rank test on the per-fold macro-F1
differences. Run with `python evaluation/statistical_significance.py`; see
[`results/README.md`](../results/README.md) for exactly which numbers are
raw vs. taken from the manuscript.

## Reproducibility notes

- `ame_derm.common.birnet_final.set_deterministic(seed)` fixes
  `random`/`numpy`/`torch`/cuDNN seeds and forces deterministic algorithms.
  Pass the same seed through `build_augmentations(seed=...)` — albumentations
  keeps its own internal RNG independent of the global seeds otherwise, so
  omitting it silently reintroduces run-to-run non-determinism in augmented
  outputs.
- A small number of secondary ablation values described in the manuscript
  were reproduced from equivalent re-runs of the scripts in this repository
  rather than recovered byte-for-byte from archived logs of the original
  experiment. The headline results (main ablation ladder, external
  validation, PanDerm comparison, segmentation ensemble) are traceable to
  the scripts listed above and their corresponding JSON output files.
