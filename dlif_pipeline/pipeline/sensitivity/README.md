# Sensitivity Analysis

This directory contains scripts for three sensitivity analyses that evaluate the robustness of the trained models.

## 1. CB Model Tested on Different Patient Subsets

**Scripts:** `collect_subset_results.py`, `run_rwd_subset_analysis.sh`, `plot_rwd_subsets.py`

We tested the performance of the CB-only model on subsets of patients based on their data availability by modality. For each outcome, the dashed gray line indicates the "baseline" model performance on all patients. Different combinations of modality availability are then tested. For each modality combination:
- The **blue dot** represents the performance of the model on the subset of patients that have all the modalities listed available.
- The **red dot** represents the performance of the model on patients that do not have any of the modalities listed.

### How to Run

```bash
# Run subset analysis across all outcomes
bash dlif_pipeline/pipeline/sensitivity/run_rwd_subset_analysis.sh \
    dlif_pipeline/results/C2/CBR/classification/standard/hypothesis_driven/pyrad-noimp/rwd/seed_0

# Collect results into CSVs
python dlif_pipeline/pipeline/sensitivity/collect_subset_results.py \
    dlif_pipeline/results/C2/CBR/classification/standard/hypothesis_driven/pyrad-noimp/rwd/seed_0

# Plot results
python dlif_pipeline/pipeline/sensitivity/plot_rwd_subsets.py
```

---

## 2. All Models Tested on the Patient Cohort with All Data Modalities Available (Complete Data)

**Scripts:** `collect_all_models_on_full_population.py`, `run_all_models_on_full_population.sh`, `plot_subset_vs_baseline.py`

To further determine whether the subset of modality-complete patients was particularly enriched, we tested the performance across all models trained on different data modality subsets on the subset of patients that had all data modalities available.

### How to Run

```bash
# Evaluate all models on the full-population subset
bash dlif_pipeline/pipeline/sensitivity/run_all_models_on_full_population.sh \
    dlif_pipeline/results/C2/CBR/classification/standard/hypothesis_driven/pyrad-noimp

# Collect results into CSVs
python dlif_pipeline/pipeline/sensitivity/collect_all_models_on_full_population.py \
    dlif_pipeline/results/C2/CBR/classification/standard/hypothesis_driven/pyrad-noimp

# Plot results
python dlif_pipeline/pipeline/sensitivity/plot_subset_vs_baseline.py
```

---

## 3. Sensitivity Analysis Using Masked Modalities on a Patient Subset with All Data Modalities Available (Complete Data)

**Scripts:** `dlif_pipeline/pipeline/train/mask_modalities.py`, `collect_mask_results.py`, `plot_mask_results.py`

To further examine whether the models trained on all data modalities were using a particular modality to improve predictions, we examined the performance of these models (using modalities CB + RADPY/RADFM + DP + GENOMICS) on the subset of patients that had complete data across all modalities by completely masking different combinations of modalities.

In these plots:
- The **dashed gray line** represents the baseline model performance on all patients.
- The **dashed blue line** represents the performance of the model on the subset of patients with complete data.

### How to Run

**Step 1 – Generate masked bags** using `mask_modalities.py`:

```bash
# Generate all masked subset combinations for radpy source
python dlif_pipeline/pipeline/train/mask_modalities.py --source radpy

# Generate for both radiomics sources
python dlif_pipeline/pipeline/train/mask_modalities.py --source radpy radfm
```

**Step 2 – Evaluate models with masked inputs** using `train.py` in evaluation mode.
Create a config file (e.g., `dlif_pipeline/configs/mask-config.yaml`) with the following content:

```yaml
task: classification
task_settings:
  outcomes: ["CBR", "ORR", "os_months_24", "os_months_6", "DCR"]
  loss: mm_loss

training_type: evaluation
data-type: hypothesis_driven
source: pyrad
imp: noimp
seed: [0]

use_early_stopping: true
prepare_dataset: false

annotation_file: ../data/annotations.csv

# Keep mods matching the TRAINED model (for path resolution)
# (find the model trained on all modalities)
mods:
  - rwd: true
    radpy: true
    dp: true
    genomics: true
  - rwd: true
    radfm: true
    dp: true
    genomics: true

eval_dataset_split: "test"

masked_mods: # which modalities to mask (true = mask, rwd always kept)
  # rwd_mask_dp_rad_genomics
  - dp: true
    radpy: true
    radfm: true
    genomics: true
  # rwd_radpy_mask_dp_genomics
  - dp: true
    radpy: false
    radfm: true
    genomics: true
  # rwd_radfm_mask_dp_genomics
  - dp: true
    radpy: true
    radfm: false
    genomics: true
  # rwd_dp_mask_rad_genomics
  - dp: false
    radpy: true
    radfm: true
    genomics: true
  # rwd_radfm_dp_mask_genomics
  - dp: false
    radpy: true
    radfm: false
    genomics: true
  # rwd_radpy_dp_mask_genomics
  - dp: false
    radpy: false
    radfm: true
    genomics: true
  # rwd_genomics_mask_dp_rad
  - dp: true
    radpy: true
    radfm: true
    genomics: false
  # rwd_radfm_genomics_mask_dp
  - dp: true
    radpy: true
    radfm: false
    genomics: false
  # rwd_radfm_genomics_mask_dp
  - dp: true
    radpy: false
    radfm: true
    genomics: false
  # rwd_dp_genomics_mask_radpy
  - dp: false
    radpy: true
    radfm: false
    genomics: false
  # rwd_dp_genomics_mask_radfm
  - dp: false
    radpy: false
    radfm: true
    genomics: false

folds:
  cross_validation:
    - GHD
    - INT
    - MH
    - SZMC
    - VHIO
  standard:
    - ALL

hyperparameters_default:
  batch_size: 16
  reconstruction_weight: 0.1
  n_layers: 2
```

Then run:

```bash
python dlif_pipeline/pipeline/train.py \
    --config dlif_pipeline/configs/mask-config.yaml \
    --base_dir dlif_pipeline/results
```

**Step 3 – Collect and plot results:**

```bash
# Collect masked-modality results into CSVs
python dlif_pipeline/pipeline/sensitivity/collect_mask_results.py

# Plot results
python dlif_pipeline/pipeline/sensitivity/plot_mask_results.py
```

---

## Output Directories

| Analysis | Output Directory |
|---|---|
| RWD subset analysis | `dlif_pipeline/results/all_mods_subset_analysis/` |
| All models on full population | `dlif_pipeline/results/models_on_all_data/` |
| Masked modalities | `dlif_pipeline/results/mask_analysis/` |
