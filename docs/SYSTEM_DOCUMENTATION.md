# System Documentation

The pipeline is described here in execution order, with every stage tied both
to the paper section that explains it and to the package that implements it.
Dataset layout is covered in [DATA_PREPARATION.md](DATA_PREPARATION.md). A
numbered, runnable rendering of the same order lives in
[`pipeline/`](../pipeline) at the repository root, while the raw outputs and
regenerable tables referred to below are found in [`results/`](../results).

## 1. Segmentation (`ame_derm/segmentation/`)

Before classification, a lesion-segmentation ensemble crops all dermoscopic
images (paper §"Lesion Segmentation Stage").

- `seg_phase1_final.py` — the final locked ensemble: SegFormer/MiT-B3 +
  UPerNet/ConvNeXt + UNet++/ResNet50 @ 384px, with 4-view TTA, guided filter,
  and largest-connected-component post-processing. Both
  `generate_phase1_crops.py` and
  `external_validation/generate_isic2019_crops.py` import
  `Phase1SegmentationPipeline`.
- `seg_mitb3_combo.py`, `seg_convnext_upernet.py` — the training scripts for
  the two strongest single encoder/decoder pairings within the ensemble.
- `seg_bgff.py` / `seg_bgff_v2.py` / `seg_highres.py` /
  `seg_foundation_guided_filter.py` — auxiliary architectures (a
  boundary-guided feature-fusion U-Net and its high-resolution variant) plus
  the guided-filter post-processing step that the final ensemble applies.
- `seg_full_ensemble_map.py` — the encoder × decoder sweep that generated the
  per-configuration Dice scores in the segmentation ablation table.
- `verify_seg_foundation_multiseed.py` — a Dice-stability check across seeds
  42/43/44.
- `generate_phase1_crops.py` — applies the locked ensemble to the whole
  dataset and writes the resulting crops to `config.CROP_DIR`.

## 2. Data splitting (`ame_derm/data_splits/`)

Every result rests on a 5-fold split that is lesion-grouped and stratified by
class and metadata, keyed on `lesion_id` rather than a simple per-image
division.

- `representative_val_split.py` — `stratified_ham_kfold_fully_representative`:
  `StratifiedGroupKFold` keyed on `lesion_id`, followed by a coverage check
  over label/dataset/sex/localization/age buckets.
- `export_representative_folds.py` — writes the fold CSVs that every
  downstream training/evaluation script consumes (including PanDerm's own
  `run_class_finetuning --csv_path` in the matched-protocol comparison).
- `audit_split_representativeness.py` — the audit confirming **zero** lesion
  overlap between train/val/test and reporting per-field coverage (mean/max
  absolute deviation from the global class distribution).
- `diagnose_metadata_imbalance.py` — a helper that the splitter/audit uses to
  determine which metadata fields are present and usable for stratification.
- `fix_lesion_grouped_split.py` — a standalone script that documents the
  leakage fix (see "Reproducibility notes" below).

### Why grouping by lesion matters

One earlier split relied on `sklearn.model_selection.train_test_split` with
stratification by class alone and **no** grouping by lesion. Since 1,956 of
HAM10000's 7,470 distinct lesions have more than one photograph (as many as
6), an individual lesion was free to land in both training and test/val — a
direct label leak. An audit of that split revealed that **34.8%** of test
images and **36.1%** of val images shared a `lesion_id` with some training
image. Both `ame_derm.common.birnet_final.stratified_ham_split` and
`representative_val_split.py` now rely on `StratifiedGroupKFold` grouped by
`lesion_id`; this shifts the intended 70/15/15 ratio to about
71.4/14.3/14.3 (an effect of the grouping constraint) and removes the overlap
completely.

## 3. Expert / generalist classifiers (`ame_derm/experts/`)

Classifiers built on DINOv2 and trained according to the paper's "Asymmetric
Anatomical MoE" design: a single whole-body generalist together with one
expert per anatomical zone, torso/limb/head (paper §"Anatomical Partitioning
and Experts").

- `train_classifier_generic.py` — the shared `GenericClassifier` base along
  with the full-loader evaluation helper.
- `train_classifier_phase2_final.py` — the precomputed-crop dataset and
  training loop that the generalist uses.
- `train_classifier_phase2_mixup.py` — introduces MixUp/CutMix together with
  the weighted soft-target loss.
- `train_specialist_expert.py` — `MultimodalClassifier`,
  `CrossAttentionFusion` (image features attending over a
  localization-metadata embedding), `EXPERT_MAPPING` (zone assignment), and
  the per-zone training loop.
- `train_moe_smart_router.py` — a baseline with a learnable router, serving as
  the comparison point for static fusion (see below).
- `train_phase3_hybrid_multimodal.py` — a generalist variant trained on a
  hybrid of real and boundary-augmented data, used as one component of the
  class-wise analysis in `moe_fusion/diagnose_classwise.py` below.
- `train_classifier_phase2_boundary_aug_v2.py` — the boundary-augmentation
  dataset builder that `train_phase3_hybrid_multimodal.py` consumes.

## 4. MoE fusion & the main ablation ladder (`ame_derm/moe_fusion/`)

Paper §"Static Fusion and Test-Time Augmentation", together with the main
results table.

- `eval_moe_pure_champion.py` — shared evaluation primitives (`tta_views`
  [6-view TTA: identity, h-flip, v-flip, 3 rotations], `extract_probs`,
  `build_exp_arrays`, `get_expert_name`) that almost every later
  evaluation/external-validation script reuses. It implements the static
  council fusion (a fixed scalar weight between the routed expert and the
  generalist).
- `eval_moe_champion_5fold.py` — the 5-fold evaluation that produces the
  headline generalist-only versus symmetric-MoE figures.
- `eval_adaptive_gater.py` — a three-way comparison among the symmetric static
  MoE, the asymmetric (large generalist + base experts) static MoE, and a
  trained `AdaptiveDualGater` MLP router. The paper's case for preferring
  static routing over a learnable gater originates here.
- `eval_hetero_torso_fold0.py` — paper §"Backbone Homogeneity and the
  Generalization Gap" (Table `tab:hetero`): at fold 0 it swaps in a Swin
  expert for the DINOv2 torso expert to ask whether backbone heterogeneity
  increases ensemble diversity. It does not — the Swin expert scores higher on
  validation but falls off sharply on held-out test.
- `diagnose_classwise.py` — paper §"Class-wise Analysis on the Lowest-F1
  Classes" (Table `tab:classwise`): per-class F1 for the symmetric MoE against
  AME-Derm on melanoma (MEL) and actinic keratosis (AKIEC), the two classes
  with the lowest F1.

## 5. Rare-class verification cascade (`ame_derm/vlm_cascade/`)

Paper §"Rare-Class Verification Cascade" / "Data-Driven Discovery of
VLM-Amenable Rare Classes".

**None of these scripts calls a model API.** Each constructs montage images
from candidate crops with hidden ground-truth keys, then expects a person (or
an interactive chat-based VLM session) to pass verdicts as a
`--reject_idx`-style argument to an "apply" step. A fully automated,
LoRA-adapted verifier that would eliminate this human-in-the-loop stage is
presented in the paper as future work; this release does not run it
end-to-end.

- `dump_bcn_preds.py` — outputs per-sample BCN predictions aligned with the
  lesion-grouped seed-42 split; that output is what the cascade scripts
  verify.
- `vlm_proto_eval.py`, `vlm_cascade_best.py`, `vlm_recall_recovery.py`,
  `vlm_bcn_vasc_verify.py` — construct the candidate montages and apply
  cascade verdicts across the various rare-class scenarios examined in the
  paper.

## 6. External validation (`ame_derm/external_validation/`)

Paper §"External Validation on an Independent Cohort".

- `retrain_bcn.py` — retrains the generalist and all three experts in full on
  BCN20000, following the same lesion-grouped, matched-seed protocol used for
  the internal HAM10000 experiments.
- `retrain_bcn_lpft.py`, `retrain_bcn_clean_crop.py` — variants covering
  linear-probe-then-fine-tune and clean-crop retraining.
- `external_zero_shot.py` / `external_fewshot_router.py` — zero-shot and
  few-shot baselines (no training on BCN/MSK) that serve as the reference
  point for the retrain comparison.
- `diagnose_bcn_f1.py` — per-class confusion and confidence diagnostics on the
  retrained model (the source of the "vascular lesions over-predicted" finding
  that motivates the verification cascade).
- `devignette_crop_bcn.py`, `prep_bcn_clahe.py`, `prep_bcn_crop_clahe.py`,
  `unsharp_crop_bcn.py` — ablations of preprocessing levers on the BCN cohort
  (de-vignetting with tight crop, CLAHE color normalization, unsharp masking).
- `prepare_isic2019_external.py`, `generate_isic2019_crops.py`,
  `pool_external_rare.py` — collect additional rare-class examples from
  ISIC2019/MSKCC/HIBA into the BCN retraining set.

## 7. PanDerm comparison (`baselines/panderm_comparison/`)

Paper §"Comparison with a Dermatology Foundation Model Under a Matched
Protocol" — the paper's most robust finding (5/5 folds, exact sign test
p=0.031), reproduced from the raw data in
[`results/data/panderm_baseline/`](../results/data/panderm_baseline) by
[`evaluation/statistical_significance.py`](../evaluation/statistical_significance.py);
see [Reproducing the results](../README.md#reproducing-the-results) in the
root README. Re-running from scratch requires the separate
[PanDerm](https://github.com/SiyuanYan1/PanDerm) repository together with its
pretrained weights; `config.AME_DERM_ROOT` (or the `AME_DERM_ROOT` environment
variable) indicates where these scripts should look for the shared fold CSVs
and checkpoints.

- `dinov2_official_protocol.py` — reproduces PanDerm's exact official
  fine-tuning recipe (layer-decay 0.65, mixup 0.8/cutmix 1.0, label smoothing
  0.1, 50 epochs) on the same data and splits, so the backbone comparison is
  isolated from protocol differences.
- `panderm_moe_prep.py` / `panderm_moe_eval.py` — build and score a
  PanDerm-backboned MoE (large generalist + base experts) using the same
  static-router logic as `moe_fusion/`, populating the PanDerm-MoE cell of the
  2x2 backbone × architecture comparison table.
- `moe_official_eval.py` — the matching official-protocol MoE evaluation for
  the DINOv2 backbone.

## 8. ABCD placebo ablation (`ame_derm/abcd_ablation/`)

Paper §"Manually Specified ABCD Features Are Redundant" (Table `tab:abcd`). It
asks whether classical ABCD descriptors (asymmetry, border, colour,
diameter/compactness), derived from the Phase-1 masks, contribute information
beyond what the DINOv2 representation already captures — under a
placebo-controlled design so that any apparent gain cannot simply be
attributed to the extra capacity of the head consuming those features.

- `extract_abcd_features.py` — derives the 8 ABCD descriptors from the Phase-1
  segmentation masks and caches them at
  `BIRNet_Research/outputs/abcd_features.csv`.
- `train_multimodal_ablation.py` — the architecturally identical 23-dim
  metadata head; its placebo variant supplies a constant null vector in place
  of real descriptors.
- `train_multimodal_abcd.py` — trains each ABCD subset (geometry / colour /
  all) through that same head, reading features from the CSV above.
- `eval_abcd_ablation.py` — computes the placebo-controlled effect δ for each
  subset and applies the reproducible-gain criterion (a subset must beat the
  placebo *and* replicate at a second fold) — no subset qualifies.

## 9. Statistical significance & inference latency (`evaluation/`)

- `statistical_significance.py` — paper §"Statistical Significance".
  Reproduces **both** headline tests: the paper's most robust finding
  (DINOv2/AME-Derm vs. PanDerm, 5/5 folds, exact sign test p=0.031, fully raw
  on both sides) and the smaller secondary internal ablation (asymmetric vs.
  symmetric MoE, paired t-test t(4)=2.30/p=0.041). Run it with
  `python evaluation/statistical_significance.py`; see
  [`results/README.md`](../results/README.md) for exactly which numbers are
  raw and which come from the manuscript.
- `inference_latency.py` — paper §"Inference Latency" (Table `tab:latency`): a
  wall-clock benchmark of the segmentation ensemble and Phase-2 classification
  (generalist + one routed expert) at fp16/fp32 on one GPU.

## Reproducibility notes

- `ame_derm.common.birnet_final.set_deterministic(seed)` pins the
  `random`/`numpy`/`torch`/cuDNN seeds and enforces deterministic algorithms.
  Feed the same seed into `build_augmentations(seed=...)`: otherwise
  albumentations retains an internal RNG of its own, separate from the global
  seeds, and leaving it out quietly reintroduces run-to-run non-determinism in
  the augmented outputs.
- A handful of secondary ablation values reported in the manuscript came from
  equivalent re-runs of the scripts in this repository, not from byte-for-byte
  recovery of the original experiments' archived logs. The headline results
  (main ablation ladder, external validation, PanDerm comparison, segmentation
  ensemble) can each be traced to the scripts listed above and their
  corresponding JSON output files.
