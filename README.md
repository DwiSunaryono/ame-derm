# AME-Derm

**AME-Derm: An Asymmetric Anatomical Mixture-of-Experts with a Data-Driven
Rare-Class Verification Algorithm for Dermoscopic Skin-Lesion Classification**
— Dwi Sunaryono, Brama Seta Putra Satria, Misbakhul Munir Irfan Subakti,
Laras Suciningtyas, Shoffi Izza Sabilla. Institut Teknologi Sepuluh Nopember,
Surabaya, Indonesia. Submitted to *Biomedical Signal Processing and Control*.

This repository is the code-availability companion to the manuscript: lesion
segmentation, an asymmetric anatomical mixture-of-experts classifier with
static fusion and test-time augmentation, a rare-class verification cascade,
external validation on BCN20000/ISIC2019, and a matched-protocol comparison
against the PanDerm foundation model.

It is extracted and reorganized from a larger iterative-experiment
repository, kept close to its original form so behavior matches what was run
for the paper — not repackaged as a general-purpose library.

## Included

- `ame_derm/segmentation/` — the Phase-1 lesion segmentation ensemble and its
  ablations.
- `ame_derm/data_splits/` — the lesion-grouped, metadata-stratified 5-fold
  splitter and its leakage audit.
- `ame_derm/experts/` — generalist and per-anatomical-zone expert classifier
  training.
- `ame_derm/moe_fusion/` — static-fusion MoE evaluation and the
  static-vs-learnable-gater ablation.
- `ame_derm/vlm_cascade/` — the rare-class verification cascade (montage
  construction + verdict application; see "Not Included" below).
- `ame_derm/external_validation/` — BCN20000/ISIC2019 external validation,
  zero/few-shot baselines, and preprocessing-lever ablations.
- `ame_derm/panderm_comparison/` — matched-protocol comparison against
  PanDerm.
- `config.py` — central dataset-path configuration (override via `.env`).

See [docs/SYSTEM_DOCUMENTATION.md](docs/SYSTEM_DOCUMENTATION.md) for how
these map onto the paper's sections and results.

## Not Included

- **Datasets.** HAM10000, BCN20000, ISIC2016/2017/2018/2019, PH2, MSKCC, and
  HIBA are all public but not redistributed here — see
  [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md).
- **Trained checkpoints.**
- **A model API for the VLM cascade.** The cascade scripts prepare candidate
  montages; verdicts are supplied by a human or an interactive chat-based VLM
  session, exactly as described as a current limitation in the paper. There
  is no automated "run the cascade end-to-end" entry point.
- **The PanDerm repository itself** — clone
  [PanDerm](https://github.com/SiyuanYan1/PanDerm) separately if you want to
  reproduce `ame_derm/panderm_comparison/`.
- Research-only exploratory scripts (~180 additional scripts from the
  original experiment repository) not needed to reproduce the results
  reported in the paper.

## Installation

Python 3.10+.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optionally copy `.env.example` to `.env` and point the paths at your own
dataset/checkpoint layout (defaults match what produced the paper's numbers):

```bash
cp .env.example .env
```

GPU with CUDA is strongly recommended for training.

## Running the pipeline

All modules live inside the `ame_derm` package and import each other by
absolute path, so run them with `-m` from the repository root:

```bash
# 1. Segmentation crops
python -m ame_derm.segmentation.generate_phase1_crops

# 2. Lesion-grouped, metadata-stratified split
python -m ame_derm.data_splits.export_representative_folds
python -m ame_derm.data_splits.audit_split_representativeness

# 3. Train generalist + anatomical experts
python -m ame_derm.experts.train_specialist_expert --help

# 4. Static-fusion MoE evaluation / main ablation ladder
python -m ame_derm.moe_fusion.eval_moe_champion_5fold --help
python -m ame_derm.moe_fusion.eval_adaptive_gater --help

# 5. External validation on BCN20000
python -m ame_derm.external_validation.retrain_bcn --help

# 6. PanDerm matched-protocol comparison (requires the PanDerm repo)
python -m ame_derm.panderm_comparison.dinov2_official_protocol --help
```

Every script accepts `--help` for its full argument list. Full stage-by-stage
detail, including which scripts back which paper table/figure, is in
[docs/SYSTEM_DOCUMENTATION.md](docs/SYSTEM_DOCUMENTATION.md).

## Documentation

- [System Documentation](docs/SYSTEM_DOCUMENTATION.md) — pipeline stages,
  paper-to-code mapping, reproducibility notes.
- [Data Preparation Guide](docs/DATA_PREPARATION.md) — dataset sources and
  expected directory layout.

## License

MIT — see [LICENSE](LICENSE).

## Citation

Citation details will be added once the manuscript is accepted.
