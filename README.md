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

Every dermoscopic image is first segmented by AME-Derm, after which an
**asymmetric anatomical mixture-of-experts** performs the classification: a
single large DINOv2 generalist combined with three zone-specific experts
(torso, limb, head), merged through a **static** scalar weight together with
six-view test-time augmentation. Precision on classes that are too scarce to
correct by retraining is then restored by a data-driven **rare-class
verification cascade**. Internal validation uses HAM10000 (lesion-grouped
5-fold CV); external validation uses the separate BCN20000 cohort; and a
protocol-matched comparison is drawn against the PanDerm dermatology
foundation model.

Inside this repository are the implementation, the unprocessed outputs behind
the paper's key figures, and a script that converts those outputs into tables
you can verify yourself — no GPU, no training, and no API calls required.

## Repository layout

| Path | Description |
|---|---|
| [`ame_derm/`](ame_derm) | The AME-Derm pipeline itself: segmentation, zone experts, static-fusion MoE, the placebo-controlled ABCD ablation, and the VLM rare-class verification cascade |
| [`baselines/`](baselines) | Comparison against the PanDerm foundation model, carried out under a training protocol matched to AME-Derm |
| [`evaluation/`](evaluation) | Significance testing (paired t-test, Wilcoxon signed-rank, exact sign test) alongside the inference-latency benchmark |
| [`pipeline/`](pipeline) | End-to-end scripts numbered in run order: crops → splits → verify → train → evaluate → external → baseline |
| [`results/`](results) | Unprocessed per-fold and per-sample outputs, plus a `reproduce_tables.py` that rebuilds the CSV tables from that data |
| [`docs/`](docs) | Documentation of the system (mapping paper sections to code) and the dataset-preparation guide |
| `config.py` | Configuration of dataset paths, which can be overridden through `.env` |

**What this repository does not ship:** the source datasets, trained
checkpoints, the PanDerm codebase, or an automated VLM API for the
verification cascade (the cascade is intentionally human-in-the-loop — see
[`ame_derm/vlm_cascade/`](ame_derm/vlm_cascade)).

## Getting started

**1. Install the dependencies**

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**2. Set the paths** — duplicate `.env.example` as `.env` and aim it at
wherever your datasets and checkpoints live (the defaults reproduce the layout
behind the paper's numbers):

```bash
cp .env.example .env
```

**3. Place your datasets** — the precise directory tree required under
`BIRNet_Research/dataset/` is given in
[`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md).

**4. Execute the pipeline**

```bash
python pipeline/01_build_segmentation_crops.py
python pipeline/02_build_splits.py
python pipeline/03_verify_split_integrity.py
python pipeline/04_train_experts.py --help          # run once per zone/fold
python pipeline/05_run_moe_evaluation.py
python pipeline/06_external_validation.py
python pipeline/07_baseline_panderm_comparison.py   # requires the PanDerm repo
```

Each script delegates to the corresponding `ame_derm`/`baselines` module and
responds to `--help` with its complete argument list. Stage-by-stage specifics,
including the script behind every table or figure in the paper, appear in
[`docs/SYSTEM_DOCUMENTATION.md`](docs/SYSTEM_DOCUMENTATION.md).

> **Only need to confirm the reported numbers?** Go directly to
> [Reproducing the results](#reproducing-the-results) below, which rebuilds
> them from data already committed here, so no dataset download is needed.

## Datasets

| Dataset | Role | Images | Source |
|---|---|---|---|
| **HAM10000** | Main classification benchmark; lesion-grouped 5-fold CV | 10,015 | [ISIC Archive collection 212](https://api.isic-archive.com/collections/212/) |
| **BCN20000** | Standalone external classification cohort | 11,982 | [ISIC Archive collection 249](https://api.isic-archive.com/collections/249/) |
| **ISIC 2018 Task 1** | Segmentation: pooled training and evaluation | 2,594 / 100 / 1,000 | [ISIC 2018 challenge](https://challenge.isic-archive.com/data/#2018) |
| **ISIC 2017** | Segmentation: pooled training only | 2,000 | [ISIC 2017 challenge](https://challenge.isic-archive.com/data/#2017) |
| **ISIC 2016** | Segmentation: pooled training only | 900 | [ISIC 2016 challenge](https://challenge.isic-archive.com/data/#2016) |
| **PH2** | Segmentation: pooled training only | 200 | [PH2 Dataset](https://www.fc.up.pt/addi/ph2%20database.html) |
| **ISIC 2019** (rare classes only) | Tried as an external rare-class supply lever — **negative result**: it loses 2.7 pts on the internal model and −1.1 to −1.8 pts on BCN20000 (Table `tab:ledger`) | subset | [ISIC 2019 challenge](https://challenge.isic-archive.com/data/#2019) |

The complete directory layout is in [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md).

## Reproducing the results

The tables below are rebuilt deterministically from the raw outputs in
[`results/data/`](results/data) by
[`results/scripts/reproduce_tables.py`](results/scripts/reproduce_tables.py)
— no GPU and no training required. The full CSVs reside in
[`results/tables/`](results/tables).

```bash
cd results
python scripts/reproduce_tables.py
```

Each table below carries a caption naming the manuscript table it reproduces
or backs, for instance *"→ paper Table 10 (`tab:2x2`)"*, and follows the
manuscript's own numbering (the sequence in which tables occur through the
whole paper, methods first). A table labeled **"supporting data for"** is not
itself part of the paper — it holds the per-fold values underlying a paper
table that reports only the mean.

### Primary result: DINOv2 outperforms the PanDerm foundation model

Among the paper's claims, this one is both the strongest and the cleanest to
verify, since both sides are recomputed from scratch inside this repository.
PanDerm's macro-F1 is recalculated here straight from its own unprocessed
per-sample predictions rather than copied out of the manuscript.

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

All 5 folds go to DINOv2 (exact sign test *p* = 0.031; paired *t*-test
*p* = 0.001), reproducing the +4.7-point gap stated in the manuscript to
within a hair.

**Table R2** — Backbone × MoE, 2×2 *(→ paper Table 10,* `tab:2x2`*)*:

| Backbone | Single generalist | MoE |
|---|---|---|
| DINOv2 | 80.00 | **81.82** (AME-Derm) |
| PanDerm | 75.34 | 77.34 |

All values here are raw apart from the AME-Derm cell — the note beneath
Table R4 explains why. Complete per-fold data:
[`results/tables/panderm_comparison.csv`](results/tables/panderm_comparison.csv).

### Secondary finding: asymmetric versus symmetric MoE

The remaining headline comparison in the manuscript — whether the asymmetric
mixture outperforms a capacity-matched symmetric counterpart — is genuine but
decidedly smaller. The manuscript describes it as "small and borderline," a
characterization we share:

**Table R3** — Main ablation ladder, generalist vs. symmetric MoE vs.
AME-Derm *(→ paper Table 7,* `tab:main`*, three of its five rows)*:

| Method | Macro-F1 |
|---|---|
| Single DINOv2 generalist | 80.00 ± 1.69 |
| Symmetric MoE (base experts + base generalist) | 80.97 ± 1.58 |
| AME-Derm (asymmetric) | 81.82 ± 1.82 |

Only means appear in Table R3. The per-fold split behind those means is
**Table R4** below — it is the actual origin of the "4 of 5 folds" statement
in paper Table 11 (`tab:stats`, row "AME-Derm vs. symmetric MoE"):

**Table R4** — Table R3, per fold *(supporting data for paper Table 11,*
`tab:stats`*)*:

| Fold | Symmetric MoE | AME-Derm | Δ |
|---|---|---|---|
| 0 | 82.35 | 83.68 | +1.33 |
| 1 | 79.19 | 79.90 | +0.71 |
| 2 | 79.34 | 80.07 | +0.73 |
| 3 | 81.74 | 83.55 | +1.81 |
| 4 | 82.25 | 81.88 | **−0.37** |

The single fold where AME-Derm comes out behind is fold 4 — the only one of
the five that does not favor it. A paired *t*-test over the five deltas gives
*t*(4) = 2.30, *p* = 0.041; Wilcoxon signed-rank gives *p* ≈ 0.062
(manuscript: 0.063). The margin is real, yet far less clean than the
DINOv2-vs-PanDerm outcome in Table R1/R2 above (which swept every fold), and
that is why for this particular claim the manuscript relies on mechanism and
fold-wise consistency rather than significance by itself.

*Note: within Table R4, the Symmetric MoE column is raw
(`champ_static_f1`, taken directly from `moe_champion_5fold_results.json`).
The AME-Derm column is not an independent measurement — the asymmetric
model's own per-fold output could not be recovered from the archived logs,
so it is rebuilt as Symmetric MoE plus the manuscript's per-fold delta (which
happens to equal the Δ column above). Every other value on this page is raw.
The complete accounting of what is raw versus manuscript-sourced is in
[`results/README.md`](results/README.md).*

### Audit of split representativeness

**Table R5** — Split-representativeness audit, averaged over folds
*(supporting data for paper Table 4,* `tab:protocol`*, and the
"lesion-grouped, metadata-representative" claims in §Experimental
Protocols)*.

No lesion leakage at all occurs in the lesion-grouped 5-fold split, a genuine
improvement over an earlier non-grouped split that leaked 34.8%/36.1% of
test/val images (the background is recounted in
[`docs/SYSTEM_DOCUMENTATION.md`](docs/SYSTEM_DOCUMENTATION.md)). Across folds
it also tracks the global class/dataset/sex/localization/age distribution
closely: mean deviation ≤0.7pp and worst case ≤3.1pp. The complete breakdown
is in
[`results/tables/split_audit_summary.csv`](results/tables/split_audit_summary.csv).

## Configuration

The dataset paths (`CROP_DIR`, `BCN_CSV`, `BCN_IMG`, `AME_DERM_ROOT`) are
gathered in [`config.py`](config.py) and can be overridden through `.env`
(see `.env.example`). All remaining settings — backbone, seeds, TTA views,
epochs, and so on — are supplied as per-script CLI arguments; invoke any
script with `--help`.

## Contributors

- **Dwi Sunaryono** — Department of Informatics, Institut Teknologi Sepuluh Nopember, Surabaya 60111, Indonesia (corresponding author, dwi@its.ac.id)
- **Brama Seta Putra Satria** — Department of Informatics, Institut Teknologi Sepuluh Nopember, Surabaya 60111, Indonesia
- **Misbakhul Munir Irfan Subakti** — Department of Informatics, Institut Teknologi Sepuluh Nopember, Surabaya 60111, Indonesia
- **Laras Suciningtyas** — Department of Electrical Engineering, Institut Teknologi Sepuluh Nopember, Surabaya 60111, Indonesia
- **Shoffi Izza Sabilla** — Department of Medical Technology, Institut Teknologi Sepuluh Nopember, Surabaya 60111, Indonesia

## How to cite

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

> **Note.** This manuscript is under submission to *Biomedical Signal
> Processing and Control* (Elsevier) and **no DOI has been assigned yet**.
> The citation above serves attribution purposes only; final bibliographic
> details (volume, pages, DOI) will be supplied once accepted. Corresponding
> author: **Dwi Sunaryono (dwi@its.ac.id)**.

## License

MIT — refer to [LICENSE](LICENSE).
