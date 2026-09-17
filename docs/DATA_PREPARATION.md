# Data Preparation Guide

No dataset is redistributed in this repository. All sources below are public;
download them yourself and lay them out as described so the default paths in
`config.py` and `ame_derm/common/birnet_final.py` resolve without edits.

## Sources

| Dataset | Role | Where to get it |
|---|---|---|
| HAM10000 | Primary training/val/test cohort (7-class) | [ISIC Archive collection 212](https://api.isic-archive.com/collections/212/) |
| ISIC 2018 Task 1 | Segmentation training data (train/val/test masks) | [ISIC 2018 challenge](https://challenge.isic-archive.com/data/#2018) |
| ISIC 2018 Task 3 | Cross-dataset classification check | Same challenge page |
| ISIC 2016 / 2017 | Supplementary segmentation data | [ISIC 2016](https://challenge.isic-archive.com/data/#2016) / [ISIC 2017](https://challenge.isic-archive.com/data/#2017) |
| ISIC 2019 | External rare-class pooling | [ISIC 2019 challenge](https://challenge.isic-archive.com/data/#2019) |
| PH2 | Supplementary segmentation data | [PH2 Dataset](https://www.fc.up.pt/addi/ph2%20database.html) |
| BCN20000 | Independent external-validation cohort | [ISIC Archive collection 249](https://api.isic-archive.com/collections/249/) |
| MSKCC, HIBA | External rare-class supply for BCN retraining | ISIC Archive (search by institution) |
| PanDerm | Foundation-model comparison baseline (code + weights, not data) | [PanDerm repository](https://github.com/SiyuanYan1/PanDerm) |

## Directory layout expected by `ame_derm/common/birnet_final.py::DatasetPaths`

Everything below is rooted at `<base_dir>/dataset` where `base_dir` defaults to
`./BIRNet_Research` (override with `--base_dir` on any script that builds a
`Config`):

```text
BIRNet_Research/dataset/
|-- ham10000/
|   |-- HAM10000_classification/
|   |   |-- HAM10000_images_part_1/
|   |   |-- HAM10000_images_part_2/
|   |   `-- HAM10000_metadata.csv
|   `-- HAM10000_segmentations_lesion_tschandl/
|-- isic2018/
|   |-- isic_task1/
|   |   |-- ISIC2018_Task1-2_Training_Input/
|   |   |-- ISIC2018_Task1_Training_GroundTruth/
|   |   |-- ISIC2018_Task1-2_Validation_Input/
|   |   |-- ISIC2018_Task1_Validation_GroundTruth/
|   |   |-- ISIC2018_Task1-2_Test_Input/
|   |   `-- ISIC2018_Task1_Test_GroundTruth/
|   `-- isic_task3/
|       |-- ISIC2018_Task3_Training_Input/
|       `-- ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv
|-- isic2016/
|   |-- ISBI2016_ISIC_Part1_Training_Data/
|   `-- ISBI2016_ISIC_Part1_Training_GroundTruth/
|-- isic2017/
|   |-- ISIC-2017_Training_Data/
|   `-- ISIC-2017_Training_Part1_GroundTruth/
`-- ph2/
    `-- PH2Dataset/
        |-- PH2 Dataset images/
        `-- PH2_dataset.txt
```

Phase-1 segmentation output (lesion crops consumed by every classifier script)
is written to and read from `CROP_DIR` (`config.py`, default
`BIRNet_Research/outputs/cropped_phase1_final`) via
`ame_derm/segmentation/generate_phase1_crops.py`.

## External-validation cohort layout

Used by `ame_derm/external_validation/*` and referenced by `config.py`
(`BCN_CSV`, `BCN_IMG`):

```text
panderm_datasets/
|-- BCN20000/BCN20000/
|   |-- bcn20000.csv
|   `-- images/
|-- MSKCC/MSKCC/
|   |-- mskcc.csv
|   `-- images/
`-- HIBA/HIBA/
    |-- hiba.csv
    `-- images/
```

ISIC 2019 external rare-class pooling additionally expects:

```text
BIRNet_Research/external_data/isic2019/
```
(see `ame_derm/external_validation/prepare_isic2019_external.py` for the
exact ground-truth CSV / image-folder names from the official ISIC 2019
archive layout).

## Overriding paths

Rather than editing source files, copy `.env.example` to `.env` and set
`CROP_DIR`, `BCN_CSV`, `BCN_IMG`, and/or `AME_DERM_ROOT` — `config.py` loads
`.env` automatically (via `python-dotenv`) and every module that needs these
paths imports them from `config.py`.

## Building Phase-1 crops

```bash
python -m ame_derm.segmentation.generate_phase1_crops
# equivalently: python pipeline/01_build_segmentation_crops.py
```

This runs the locked segmentation ensemble
(`ame_derm/segmentation/seg_phase1_final.py`) over HAM10000 and writes crops
to `CROP_DIR`, which every classifier/expert/MoE script downstream expects.

## Building the lesion-grouped split

```bash
python -m ame_derm.data_splits.export_representative_folds       # pipeline/02_build_splits.py
python -m ame_derm.data_splits.audit_split_representativeness    # pipeline/03_verify_split_integrity.py
```

The first exports the 5-fold, lesion-grouped, metadata-stratified split used
throughout the paper; the second verifies zero lesion leakage and balanced
metadata coverage across folds (see `docs/SYSTEM_DOCUMENTATION.md` for why
this matters).
