# DLIF Model Training Guide

This guide explains how to use the DLIF (Deep Learning Intermediate Fusion) pipeline for classification and survival analysis.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Structure](#output-structure)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Conda or Miniconda

### Setup
Run this from within the `I3LUNG_PDSS` directory
```bash
# Create and activate conda environment
conda create -n dlif python=3.9
conda activate dlif
conda install -c conda-forge spacy
# Install dependencies
pip install -r dlif_pipeline/requirements.txt

# Configure environment
export MPLBACKEND=Agg  # Required for headless plotting
```

# Data Requirements and Preparation

## Required Files

Place the following files in the `I3LUNG_PDSS/data` directory before running the pipelines:

- `/data/rwd.csv`
- `/data/genomics.csv`
- `/data/digital_pathology.csv`
- `/data/pyradiomics.csv`
- `/data/fmrad.csv`
- `/data/outcomes.csv`
- `/data/features.json`


## How to Create Parquet and Annotation Files

Please run all commands from within the  `I3LUNG_PDSS` directory.

1. **Impute and normalize the data**  
   Run:  
   `python dlif_pipeline/preprocessing/utils/impute_and_normalize.py`  
   This creates processed versions of all modality files with the `_processed` suffix in the `/data` directory.

2. **Create the Parquet files**  
   Run:  
   `python dlif_pipeline/preprocessing/create_parquet.py`  
   This generates the Parquet files in the `/data` directory.

3. **Create the annotation file**  
   Run:  
   `python dlif_pipeline/preprocessing/create_annotations.py`  
   The annotation file will be saved in `/data`.

   - **Note:** The `create_annotations.py` script includes a `use_subfolds` option.  
     To enable intra-center subanalysis, edit the script and set:  
     `ann = create_annotations(use_subfolds=True)`

These steps will produce:
- Two Parquet files (one with `mod2 = pyradiomics`, one with `mod2 = fmrad`)
- An annotation file with outcomes and subanalysis flags

Ensure all files are present before running downstream experiments.

## Project Structure

The expected directory structure is as follows:
```
I3LUNG_PDSS/
└── dlif_pipeline/                          # This repository
    ├── configs/                            # Configuration files
    ├── MIL/                                # Model
    ├── pipeline/                           # Training scripts
    ├── preprocessing/                      # Data preprocessing scripts
    ├── datasets.json/                      # Dataset config for MIL
    ├── bags/                               # Bag-level features
    │   ├── radfm/                          # RadFM modality bags
    │   └── radpy/                          # RadPy modality bags
    ├── results/                            # Training outputs (auto-generated)
    └── README.md
```

## Configuration

Configuration files are located in the `configs/` directory:

### Classification Tasks
- `00-config-classification-cv.yaml` - Cross-validation configuration
- `01-config-classification-standard.yaml` - Standard training configuration
- `02-config-classification-eval.yaml` - External validation configuration

### Survival Tasks
- `03-config-survival-cv.yaml` - Cross-validation configuration
- `04-config-survival-standard.yaml` - Standard training configuration
- `05-config-survival-eval.yaml` - External validation configuration

## Usage

Before running, please make sure you prepared the required data. You can find how to prepare the required data in the dedicated README.md in the preprocessing folder.
Run all commands from the `I3LUNG_PDSS` directory.

### Classification Tasks

#### 1. Cross-Validation (Leave-One-Center-Out)

Performs leave-one-center-out cross-validation across multiple centers:
- GHD
- INT
- MH
- SZMC
- VHIO
```bash
python dlif_pipeline/pipeline/train.py --config dlif_pipeline/configs/00-config-classification-cv.yaml --base_dir dlif_pipeline/results
```

#### 2. Standard Training

Trains on all data from the above sites and evaluates on held-out data from each site:
```bash
python dlif_pipeline/pipeline/train.py \
  --config dlif_pipeline/configs/01-config-classification-standard.yaml \
  --base_dir dlif_pipeline/results
```

#### 3. External Validation

Evaluates on external held-out center (UOC):
```bash
python dlif_pipeline/pipeline/train.py \
  --config dlif_pipeline/configs/02-config-classification-eval.yaml \
  --base_dir dlif_pipeline/results
```

### Survival Tasks

Use the same commands with survival configuration files:

#### 1. Cross-Validation
```bash
python dlif_pipeline/pipeline/train.py \
  --config dlif_pipeline/configs/03-config-survival-cv.yaml \
  --base_dir dlif_pipeline/results
```

#### 2. Standard Training
```bash
python dlif_pipeline/pipeline/train.py \
  --config dlif_pipeline/configs/04-config-survival-standard.yaml \
  --base_dir dlif_pipeline/results
```

#### 3. External Validation
```bash
python dlif_pipeline/pipeline/train.py \
  --config dlif_pipeline/configs/05-config-survival-eval.yaml \
  --base_dir dlif_pipeline/results
```

### Calculate Metrics

```bash
python dlif_pipeline/pipeline/metrics/compute_metrics_from_config.py \
  --config dlif_pipeline/configs/00-config-classification-cv.yaml \
  --base_dir dlif_pipeline/results

python dlif_pipeline/pipeline/metrics/compute_metrics_from_config.py \
  --config dlif_pipeline/configs/01-config-classification-standard.yaml \
  --base_dir dlif_pipeline/results
```

### Create Plots

```
bash
python dlif_pipeline/pipeline/plotting/plot_from_config.py \
  --config dlif_pipeline/configs/00-config-classification-cv.yaml \
  --base_dir dlif_pipeline/results

python dlif_pipeline/pipeline/plotting/plot_from_config.p \
  --config dlif_pipeline/configs/01-config-classification-standard.yaml \
  --base_dir dlif_pipeline/results
```

## Output Structure

Training outputs are organized hierarchically in the results directory:
```
results/
└── C23/
    └── os_months_24/
        └── classification/                     # or 'survival'
            └── cross_validation/               # or 'standard'/'evaluation'
                └── hypothesis_driven/
                    └── pyrad-noimp/
                        └── rwd_radpy/
                            └── seed_0/
                                ├── attention/
                                │   └── attention_weights.npz          # Raw attention weights
                                ├── eval/                              # Test set evaluation
                                │   └── 00000-mb_attention_mil/
                                │       ├── attention/
                                │       │   └── attention_weights.npz
                                │       ├── mil_params.json            # Model configuration
                                │       ├── predictions.parquet        # Test predictions
                                │       └── scores_test.csv            # Test metrics (survival only)
                                ├── models/
                                │   └── best_valid.pth                 # Best model checkpoint
                                ├── eval_cindex_ci.csv                 # C-index with CI (survival only)
                                ├── eval_auc_ci.csv                    # AUC with CI (classification only)
                                ├── eval_classification_metrics.csv    # F1, sensitivity, specificity (classification only)
                                ├── history.csv                        # Training history per epoch
                                ├── mil_params.json                    # Model hyperparameters
                                ├── predictions_train.parquet          # Training predictions
                                ├── predictions.parquet                # Test predictions
                                └── slide_manifest.csv                 # Slide-level metadata
```

### Key Output Files

**Model Artifacts:**
- `best_valid.pth` - Best performing model checkpoint based on validation metrics

**Predictions:**
- `predictions_train.parquet` - Training set predictions with true labels
- `predictions.parquet` - Test set predictions with true labels
- `slide_manifest.csv` - Slide-level metadata and predictions

**Metrics (Classification):**
- `eval_auc_ci.csv` - AUC with confidence intervals
- `eval_classification_metrics.csv` - F1 score, sensitivity, specificity

**Metrics (Survival):**
- `eval_cindex_ci.csv` - Concordance index with confidence intervals
- `scores_test.csv` - Test set survival metrics

**Training History:**
- `history.csv` - Loss and metrics tracked per epoch

**Attention Mechanisms:**
- `attention_weights.npz` - Raw attention weights for interpretability

**Configuration:**
- `mil_params.json` - Complete model configuration and hyperparameters
