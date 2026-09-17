<div align="center">

# AME-Derm

**An Asymmetric Anatomical Mixture-of-Experts with a Data-Driven Rare-Class
Verification Algorithm for Dermoscopic Skin-Lesion Classification**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20code-orange)
![Reproducible](https://img.shields.io/badge/vs%20PanDerm-%2B4.7pts%2C%205%2F5%20folds%2C%20p%3D0.031-green)
![Datasets](https://img.shields.io/badge/datasets-HAM10000%2C%20BCN20000%2C%20ISIC%2C%20PH2-blueviolet)

</div>

---

AME-Derm segments each dermoscopic image, then classifies it with an
**asymmetric anatomical mixture-of-experts**: a large DINOv2 generalist plus
three anatomical-zone experts (torso/limb/head), fused by a **static** scalar
weight and six-view test-time augmentation. A data-driven **rare-class
verification cascade** recovers precision on classes too scarce to fix by
retraining. The method is validated internally on HAM10000 (lesion-grouped
5-fold CV) and externally on the independent BCN20000 cohort, and compared
against the PanDerm dermatology foundation model under a matched training
protocol.

This repository contains the implementation, the raw results behind the
paper's headline numbers, and a script that turns them into tables you can
check yourself — no training, no GPU, no API calls needed.

## Contents

| Path | Description |
|---|---|
| [`ame_derm/`](ame_derm) | The AME-Derm method: segmentation, anatomical experts, static-fusion MoE, the ABCD placebo ablation, and the VLM rare-class verification cascade |
| [`baselines/`](baselines) | The PanDerm foundation-model comparison, under a training protocol matched to AME-Derm |
| [`evaluation/`](evaluation) | Statistical significance testing (paired t-test, Wilcoxon signed-rank, exact sign test) and the inference-latency benchmark |
| [`pipeline/`](pipeline) | Numbered end-to-end scripts: crops → splits → verify → train → evaluate → external → baseline |
| [`results/`](results) | Raw per-fold/per-sample outputs and a `reproduce_tables.py` that regenerates CSV tables from them |
| [`docs/`](docs) | System documentation (paper-to-code mapping) and the data preparation guide |
| `config.py` | Dataset-path configuration, overridable via `.env` |

**Not included:** the source datasets, trained checkpoints, the PanDerm
repository itself, and an automated VLM API for the verification cascade
(it is human-in-the-loop by design — see [`ame_derm/vlm_cascade/`](ame_derm/vlm_cascade)).

## Quick start

**1. Install dependencies**

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**2. Configure paths** — copy `.env.example` to `.env` and point it at your
dataset/checkpoint layout (defaults match what produced the paper's numbers):

```bash
cp .env.example .env
```

**3. Add your datasets** — see [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md)
for the exact directory layout expected under `BIRNet_Research/dataset/`.

**4. Run the pipeline**

```bash
python pipeline/01_build_segmentation_crops.py
python pipeline/02_build_splits.py
python pipeline/03_verify_split_integrity.py
python pipeline/04_train_experts.py --help          # run once per zone/fold
python pipeline/05_run_moe_evaluation.py
python pipeline/06_external_validation.py
python pipeline/07_baseline_panderm_comparison.py   # requires the PanDerm repo
```

Every script forwards to its underlying `ame_derm`/`baselines` module and
accepts `--help` for its full argument list. Full stage-by-stage detail,
including which scripts back which paper table/figure, is in
[`docs/SYSTEM_DOCUMENTATION.md`](docs/SYSTEM_DOCUMENTATION.md).

> **Just want to check the reported numbers?** Skip straight to
> [Reproducible results](#reproducible-results) below — it regenerates from
> data already in this repository, no dataset download required.

## Datasets

| Dataset | Role | Images | Source |
|---|---|---|---|
| **HAM10000** | Primary classification benchmark, lesion-grouped 5-fold CV | 10,015 | [ISIC Archive collection 212](https://api.isic-archive.com/collections/212/) |
| **BCN20000** | Independent external classification cohort | 11,982 | [ISIC Archive collection 249](https://api.isic-archive.com/collections/249/) |
| **ISIC 2018 Task 1** | Segmentation: pooled training + evaluation | 2,594 / 100 / 1,000 | [ISIC 2018 challenge](https://challenge.isic-archive.com/data/#2018) |
| **ISIC 2017** | Segmentation: pooled training only | 2,000 | [ISIC 2017 challenge](https://challenge.isic-archive.com/data/#2017) |
| **ISIC 2016** | Segmentation: pooled training only | 900 | [ISIC 2016 challenge](https://challenge.isic-archive.com/data/#2016) |
| **PH2** | Segmentation: pooled training only | 200 | [PH2 Dataset](https://www.fc.up.pt/addi/ph2%20database.html) |
| **ISIC 2019** (rare classes only) | Tested external rare-class supply lever — **negative result**: costs 2.7 pts on the internal model and −1.1 to −1.8 pts on BCN20000 (Table `tab:ledger`) | subset | [ISIC 2019 challenge](https://challenge.isic-archive.com/data/#2019) |

Full directory layout in [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md).

## Reproducible results

The tables below are regenerated deterministically from the raw outputs in
[`results/data/`](results/data) by
[`results/scripts/reproduce_tables.py`](results/scripts/reproduce_tables.py)
— no training, no GPU. Full CSVs are in [`results/tables/`](results/tables).

```bash
cd results
python scripts/reproduce_tables.py
```

Every table below is captioned with the specific manuscript table it
reproduces or supports, e.g. *"→ paper Table 10 (`tab:2x2`)"*, using the
manuscript's own table numbers (in the order tables appear across the whole
paper, methods section first). Tables marked **"supporting data for"**
aren't themselves in the paper — they're the per-fold numbers behind a
paper table that only reports the mean.

### Headline result: DINOv2 beats the PanDerm foundation model

This is the paper's strongest claim, and the one we can verify most
cleanly: both sides are computed from scratch in this repository. PanDerm's
macro-F1 is recomputed here directly from its own raw per-sample
predictions, not copied from the manuscript.

```bash
python evaluation/statistical_significance.py
```

**Table R1** — DINOv2 vs. PanDerm generalist, per fold *(supporting data for
paper Table 7* `tab:main` *and Table 11* `tab:stats`*, row "DINOv2 vs.
PanDerm (backbone)")*:

| Fold | DINOv2 generalist | PanDerm generalist | Δ |
|---|---|---|---|
| 0 | 79.37 | 73.57 | +5.80 |
| 1 | 78.48 | 73.09 | +5.39 |
| 2 | 79.35 | 73.86 | +5.49 |
| 3 | 79.91 | 77.62 | +2.29 |
| 4 | 82.88 | 78.56 | +4.33 |
| **mean** | **80.00** | **75.34** | **+4.66** |

DINOv2 wins all 5 folds (exact sign test *p* = 0.031; paired *t*-test
*p* = 0.001), reproducing the manuscript's reported +4.7-point gap almost
exactly.

**Table R2** — Backbone × MoE, 2×2 *(→ paper Table 10,* `tab:2x2`*)*:

| Backbone | Single generalist | MoE |
|---|---|---|
| DINOv2 | 80.00 | **81.82** (AME-Derm) |
| PanDerm | 75.34 | 77.34 |

Every number here is raw except the AME-Derm cell — see the note under
Table R4 below for why. Full per-fold data:
[`results/tables/panderm_comparison.csv`](results/tables/panderm_comparison.csv).

### Secondary result: asymmetric vs. symmetric MoE

The manuscript's other headline comparison — does the asymmetric mixture
beat a capacity-matched symmetric one — is a real but much smaller effect.
The manuscript itself calls it "small and borderline," and we agree:

**Table R3** — Main ablation ladder, generalist vs. symmetric MoE vs.
AME-Derm *(→ paper Table 7,* `tab:main`*, three of its five rows)*:

| Method | Macro-F1 |
|---|---|
| Single DINOv2 generalist | 80.00 ± 1.69 |
| Symmetric MoE (base experts + base generalist) | 80.97 ± 1.58 |
| AME-Derm (asymmetric) | 81.82 ± 1.82 |

Table R3 only has means. **Table R4** below is the per-fold breakdown behind
it — this is where the "4 of 5 folds" claim in paper Table 11 (`tab:stats`,
row "AME-Derm vs. symmetric MoE") actually comes from:

**Table R4** — Table R3, per fold *(supporting data for paper Table 11,*
`tab:stats`*)*:

| Fold | Symmetric MoE | AME-Derm | Δ |
|---|---|---|---|
| 0 | 82.35 | 83.68 | +1.33 |
| 1 | 79.19 | 79.90 | +0.71 |
| 2 | 79.34 | 80.07 | +0.73 |
| 3 | 81.74 | 83.55 | +1.81 |
| 4 | 82.25 | 81.88 | **−0.37** |

Fold 4 is where AME-Derm actually loses — that's the one fold out of five
that doesn't favor it. Paired *t*-test on these five deltas:
*t*(4) = 2.30, *p* = 0.041. Wilcoxon signed-rank: *p* ≈ 0.062 (manuscript:
0.063). It's a real margin, but nowhere near as clean as the DINOv2-vs-
PanDerm result in Table R1/R2 above (which won every fold), which is why
the manuscript leans on mechanism and fold-wise consistency rather than
significance alone for this particular claim.

*Note: in Table R4, the Symmetric MoE column is raw (`champ_static_f1`,
straight from `moe_champion_5fold_results.json`). The AME-Derm column is
not an independent measurement — we couldn't recover the asymmetric
model's own per-fold output from archived logs, so it's reconstructed as
Symmetric MoE plus the per-fold delta the manuscript reports (which is also
exactly the Δ column above). Everything else on this page is raw. See
[`results/README.md`](results/README.md) for the full breakdown of what's
raw versus manuscript-sourced.*

### Split-representativeness audit

**Table R5** — Split-representativeness audit, averaged over folds
*(supporting data for paper Table 4,* `tab:protocol`*, and the
"lesion-grouped, metadata-representative" claims in §Experimental
Protocols)*.

The lesion-grouped 5-fold split has zero lesion leakage — a real fix from
an earlier, non-grouped split that leaked 34.8%/36.1% of test/val images
(see [`docs/SYSTEM_DOCUMENTATION.md`](docs/SYSTEM_DOCUMENTATION.md) for the
story). It also stays close to the global class/dataset/sex/localization/age
distribution across folds: mean deviation ≤0.7pp, worst case ≤3.1pp. Full
breakdown in
[`results/tables/split_audit_summary.csv`](results/tables/split_audit_summary.csv).

## Configuration

Dataset paths (`CROP_DIR`, `BCN_CSV`, `BCN_IMG`, `AME_DERM_ROOT`) are
centralized in [`config.py`](config.py) and overridable via `.env`
(see `.env.example`). Everything else (backbone choice, seeds, TTA views,
epochs, etc.) is a per-script CLI argument — run any script with `--help`.

## Contributors

- **Dwi Sunaryono** — Department of Informatics, Institut Teknologi Sepuluh Nopember, Surabaya 60111, Indonesia (corresponding author, dwi@its.ac.id)
- **Brama Seta Putra Satria** — Department of Informatics, Institut Teknologi Sepuluh Nopember, Surabaya 60111, Indonesia
- **Misbakhul Munir Irfan Subakti** — Department of Informatics, Institut Teknologi Sepuluh Nopember, Surabaya 60111, Indonesia
- **Laras Suciningtyas** — Department of Electrical Engineering, Institut Teknologi Sepuluh Nopember, Surabaya 60111, Indonesia
- **Shoffi Izza Sabilla** — Department of Medical Technology, Institut Teknologi Sepuluh Nopember, Surabaya 60111, Indonesia

## Citation

```bibtex
@article{sunaryono2026amederm,
  title   = {AME-Derm: An Asymmetric Anatomical Mixture-of-Experts with a
             Data-Driven Rare-Class Verification Algorithm for Dermoscopic
             Skin-Lesion Classification},
  author  = {Sunaryono, Dwi and Satria, Brama Seta Putra and Subakti,
             Misbakhul Munir Irfan and Suciningtyas, Laras and Sabilla,
             Shoffi Izza},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026},
  note    = {Submitted / under review}
}
```

> **Note.** This is a manuscript submitted to *Biomedical Signal Processing
> and Control* (Elsevier); **no DOI is assigned yet**. This citation is
> provided for attribution only — final bibliographic details (volume,
> pages, DOI) will follow acceptance. Corresponding author: **Dwi Sunaryono
> (dwi@its.ac.id)**.

## License

MIT — see [LICENSE](LICENSE).
