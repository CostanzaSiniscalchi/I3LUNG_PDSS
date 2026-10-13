# Prospective Cohort: Preprocessing and Evaluation

This guide covers running the DLIF models trained on the retrospective cohort against the prospective cohort. The prospective cohort is **evaluation only**: nothing is fit or retrained. Imputers and scalers fit on the retrospective TRAIN split are reused, and the checkpoints trained with `training_type: standard` are used for inference.

Run all commands from the repository root (`I3LUNG_PDSS_1/`).

## Prerequisites

- The retrospective pipeline has been run (see [README.md](README.md)), so `data/` contains the fitted artifacts: `*_imputer.pkl`, `*_scaler.pkl` and `*_norm_config.json`. These are written by `impute_and_normalize.py`, including the per-source artifacts for each `digital_pathology_<token>` variant.
- Standard-mode checkpoints exist under `dlif_pipeline/results/` for the modality combinations you want to evaluate.
- The prospective raw files are in `data/prospective/`.

## 1. Raw input files

Place these in `data/prospective/`:

| File | Notes |
|---|---|
| `cb.csv` | Required. Must have `Subject` and `CENTER`. It has **no** `SET` column (no train/test split), which is why step 4 passes `--default-dataset`. |
| `outcomes.csv` | Required for annotations (`DCR`, `os_months_6`, `os_months_24`, ...). |
| `genomics.csv` | Optional. |
| `pyradiomics.csv`, `fmrad.csv` | Optional radiomics sources. |
| `digital_pathology.csv` | Optional. Legacy/gigapath source. |
| `digital_pathology_titan.csv`, `digital_pathology_titan_coral.csv` | Optional. Titan and Titan-Coral variants. Pass each through `--dp-types` in step 2. |

Modalities that are missing are skipped with a warning. Their columns are left null, so the cohort can be onboarded one modality at a time.

## 2. Impute and normalize (transform only)

```bash
python dlif_pipeline/preprocessing/preprocess_new_cohort.py \
    --raw-data-dir data/prospective \
    --artifacts-dir data \
    --output-dir data/prospective \
    --dp-types digital_pathology digital_pathology_titan digital_pathology_titan_coral
```

This writes `<modality>_processed.csv` into `data/prospective/`. Useful options:

- `--modalities cb genomics`: restrict the fixed modalities processed (default: all of `cb`, `genomics`, `pyradiomics`, `fmrad`).
- `--dp-types ...`: digital pathology sources. Each `digital_pathology_<token>` reads `<raw-data-dir>/digital_pathology_<token>.csv`. The default is only the legacy `digital_pathology`.
- `--<modality>-path <file>`: override the raw file for one modality, e.g. `--genomics-path data/prospective/genomics_v2.csv`.

## 3. Build the feature parquets

`create_prospective_parquet.py` builds **one parquet per rad/dp combination**, so run it once for each combination you need. `cb` is mandatory and genomics is included automatically if `genomics_processed.csv` exists.

```bash
# pyradiomics + Titan DP  ->  features_dataset_radpy_fixed_dp-titan.parquet
python dlif_pipeline/preprocessing/create_prospective_parquet.py \
    --rad-type pyradiomics --dp-type digital_pathology_titan

# fmrad + Titan DP  ->  features_dataset_fmrad_dp-titan.parquet
python dlif_pipeline/preprocessing/create_prospective_parquet.py \
    --rad-type fmrad --dp-type digital_pathology_titan

# Titan-Coral variants
python dlif_pipeline/preprocessing/create_prospective_parquet.py \
    --rad-type pyradiomics --dp-type digital_pathology_titan_coral
python dlif_pipeline/preprocessing/create_prospective_parquet.py \
    --rad-type fmrad --dp-type digital_pathology_titan_coral
```

Omit `--rad-type` or `--dp-type` if that modality isn't available. Its `mod2` or `mod3` column is then null for every subject. Parquets are written to `data/prospective/` (override with `--data-dir`, or name the file with `--output-name`).

Parquet naming follows the retrospective convention: `features_dataset_<radpy_fixed|fmrad>[_dp-<token>].parquet`.

## 4. Create the annotation file

```bash
python dlif_pipeline/preprocessing/create_annotations.py \
    --data-dir data/prospective \
    --dp-types digital_pathology digital_pathology_titan digital_pathology_titan_coral \
    --default-dataset ext_val
```

- `--default-dataset ext_val` tags every subject as `ext_val`. This is required because the prospective `cb.csv` has no `SET` column. The eval config's `eval_dataset_split` must match this tag.
- `--dp-types` adds the `HAS_DP*` coverage flags for each digital pathology source.
- Output: `data/prospective/annotations.csv`.

## 5. Run the evaluation

Use the configs in [configs/prosp/](configs/prosp/):

| Config | Digital pathology source |
|---|---|
| `config-prospective-eval.yaml` | DP (gigapath) |
| `config-prospective-titan-eval.yaml` | DP-Titan |
| `config-prospective-titan_coral-eval.yaml` | DP-Titan-Coral |

```bash
python dlif_pipeline/pipeline/train.py \
    --config dlif_pipeline/configs/prosp/config-prospective-titan-eval.yaml \
    --base_dir dlif_pipeline/results
```

Key settings in these configs:

- `training_type: evaluation`: inference only.
- `eval_model_training_type: standard`: loads the checkpoints trained in standard mode.
- `eval_cohort_tag: prospective`: writes outputs to `evaluation_prospective/` rather than the shared `evaluation/` directory used by the retrospective external-validation eval.
- `eval_dataset_split: ext_val`: must match `--default-dataset` from step 4.
- `annotation_file` and `train_df`: point at the prospective `annotations.csv` and the parquets from step 3. Make sure `train_df` lists every parquet the `mods` entries need.
- `prepare_dataset: true` with `bags_dir: dlif_pipeline/bags_prospective`: builds the per-modality-combination bags into `bags_prospective/bags_<mods>/`, separate from the retrospective `bags/`.
- `mods`: the modality combinations to evaluate. Each needs a matching standard-mode checkpoint. If one is missing, drop that entry.

Per-modality results are written under
`dlif_pipeline/results/<cohort>/<outcome>/classification/evaluation_prospective/hypothesis_driven/pyrad-noimp/<modality>/seed_0/`
(including `eval_auc_ci.csv` and `eval_classification_metrics.csv`).
