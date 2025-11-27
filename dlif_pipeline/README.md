# Deep Learning Intermediate Fusion (DLIF) pipeline

A comprehensive framework for training and evaluating attention-based multiple instance learning (MIL) models on multimodal data, with support for both classification and survival analysis tasks. DLIF enables patient-level modelling without requiring complete data across all modalities.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Complete Workflow](#complete-workflow)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## Overview

This pipeline provides an end-to-end solution for MIL on medical data, featuring:

- **Multimodal Integration**: Combines radiomics (PyRadiomics/Foundation Models), clinical data (RWD), digital pathology (DP), and genomics
- **Multiple Training Strategies**: Standard training, cross-validation, hyperparameter tuning, and external validation
- **Task Support**: Binary/multi-class classification and survival analysis (Cox proportional hazards)
- **Attention Mechanisms**: Multi-bag attention-based architecture for interpretable predictions

## Features

### Data Modalities

| Modality       | Description                                                           |
|----------      |-------------                                                          |
| **RWD**        | Real-world clinical data and patient demographics (required baseline) |
| **RadPy**      | PyRadiomics-extracted radiomic features                               |
| **RadFM**      | Foundation model-based radiomic features                              |
| **DP**         | Deep learning features from digital pathology                         |
| **Genomics**   | Molecular and genetic biomarkers                                      |

### Training Modes

- **Standard**: Single train/validation/test split
- **Cross-Validation**: Leave-one-center-out cross-validation
- **Hyperparameter Tuning**: Grid search with nested cross-validation
- **External Validation**: Evaluation on held-out center (UOC)

### Model Architecture

The `mb_attention_mil` model implements:
- Attention-based multi-instance pooling
- Variable-length bag handling
- Multimodal feature fusion with learned weights
- Optional reconstruction loss for feature regularization

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
```

## Quick Start

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
     To enable uni-centric subanalysis, edit the script and set:  
     `ann = create_annotations(use_subfolds=True)`

These steps will produce:
- Two Parquet files (one with `mod2 = pyradiomics`, one with `mod2 = fmrad`)
- An annotation file with outcomes and subanalysis flags

Ensure all files are present before running downstream experiments.


### 2. Configure Experiment

Create a YAML configuration file (see [`configs/`](configs/) for examples):

```yaml
# configuration example
task: classification # classification/survival
task_settings:
  outcomes: ["os_months_24"]
  # events: DEATH_EVENT_OC # survival only variable
  loss: mm_loss # mm_survival_loss  
  # save_monitor: c_index  # survival only variable

training_type: cross_validation # hyperparameter_tuning  | evaluation | standard
data-type: hypothesis_driven  # hypothesis_driven/data_driven
source: pyrad  # pyrad/foundation
imp: noimp  # noimp/imp for imp should use a separate dataset that includes an additional column for each clinical feature to indicate whether the value was imputed (imp columns), allowing tracking of missing data across patients.
seed: [0]

use_early_stopping: true
prepare_dataset: true

train_df: [ "../data/features_dataset_radpy_fixed.parquet", "../data/features_dataset_fmrad.parquet" ]
annotation_file: ../data/annotations.csv

mods:
    # RWD
  - rwd: true
    radpy: false
    dp: false
    genomics: false
    # RWD_DP
  - rwd: true
    radpy: false
    dp: true
    genomics: false
    # RWD_FMRAD
  - rwd: true
    radfm: true
    dp: false
    genomics: false
    # RWD_PYRAD
  - rwd: true
    radpy: true
    dp: false
    genomics: false
    # RWD_DP_FMRAD
  - rwd: true
    radfm: true
    dp: true
    genomics: false
    # RWD_DP_PYRAD
  - rwd: true
    radpy: true
    dp: true
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

# default hyperparameters
hyperparameters_default:
  batch_size: 64
  reconstruction_weight: 0.1
  n_layers: 1


```


## Configuration Details

We now provide details on how to edit the configuration to run different analysis.

### Task Settings

```yaml
task_settings:
  outcomes: ["OS_MONTHS"]     # Target variable(s), for classificaiton you can also have multiple outcomes: ["os_months_6", "os_months_24","DCR", "ORR"] 
  events: DEATH_EVENT_OC         # Event indicator (survival only)
  loss: mm_survival_loss         # Loss function
  save_monitor: c_index         # survival only

```

**Supported Outcomes:**
- Classification: `os_months_6`, `os_months_24`, `DCR`, `ORR`
- Survival: `OS_MONTHS`

**Loss Functions:**
- `mm_loss`: Binary classification
- `mm_survival_loss`: Cox proportional hazards

### Training Options

```yaml
training_type: cross_validation  # standard | cross_validation | hyperparameter_tuning | evaluation
data_type: hypothesis_driven     # hypothesis_driven 
source: pyrad                    # pyrad 
imp: noimp                       # noimp 
seed: [0]                  # Random seeds for reproducibility

use_early_stopping: true
prepare_dataset: true  # Set to true on first run or when data changes
```

### Modality Combinations

```yaml
mods:
  - rwd: true      # Always required
    radpy: true
    dp: false
    genomics: false
  
  - rwd: true
    radpy: true
    dp: true
    genomics: false
```

> **Note**: RWD must be `true` in all configurations as it serves as the baseline modality.

You can add as many modalities combinations as you want (within the modalities supported: rwd, radpy, radfm, dp, genomics.), 
radpy and radfm are mutually exclusive.

### Hyperparameter Configuration

**For Grid Search:**

```yaml
hyperparameters:
  batch_size: [16, 32, 64]
  reconstruction_weight: [0.01, 0.1, 0.3]
  n_layers: [1, 2]
```

**For Standard Training:**

```yaml
hyperparameters_default:
  batch_size: 64
  reconstruction_weight: 0.1
  n_layers: 1
```

### Optional Filters

Apply dataset filters by uncommenting relevant flags:

```yaml
# Cohort selection
# USE_COHORT2_FILTER: [true]

# Histology subtype
# FILTER_SQUAMOUS: "1.0"  # or "0.0" for non-squamous
# ADENO: "1.0"

# Treatment regimen
# FILTER_CHEMO_IMMUNO: "1"  # or "0"

# Biomarker expression
# FILTER_PDL1: "high"  # or "low"

# Center-specific (use only one at a time)
# FILTER_INT: [true]
```

before applying subanalysis for unicenter, make sure to have generated the specific annotations.


### Complete Workflow

### 3. Train Model and Evaluate model

This is an example of how to train the model with the minimum configuration example just provided (you can find the same configuration at the directory dlif_pipeline/configs/00-config-classification-cv.yaml):

```bash
python dlif_pipeline/pipeline/train.py --config dlif_pipeline/configs/01-config-classification-standard.yaml --base_dir dlif_pipeline/results
```
Other examples are provided at dlif_pipeline/configs.

**Output Structure:**

```
results/
└── cohort2/
    └── mil/
        └── os_months_24/
            └── classification/
                └── cross_validation/
                    └── hypothesis_driven/
                        └── pyrad-noimp/
                            └── rwd_radpy/                                 
                                   └──seed_0/
                                      ├── attention/                          # Attention mechanism outputs
                                      │   ├── attention_weights.npz           # Raw attention weights
                                      ├── eval/                               # model on the test set
                                      │   ├── 00000-mb_attention_mil/
                                              ├── attention/                          # Attention mechanism outputs
                                                   ├── attention_weights.npz          # Raw attention weights
                                              ├── mil_params.json                     # Model configuration and hyperparameters
                                              ├── predictions.parquet                 # predictions in test
                                              ├── scores_test.csv                     # metrics in test (survival only)
                                      ├── models/                             # Model checkpoints
                                      │   ├── best_valid.pth                  # Best model (by validation metric)
                                      ├── eval_cindex_ci.csv                  # C-index with confidence intervals (survival only)
                                      ├── eval_auc_ci.csv                     # AUC with confidence intervals (classification only)
                                      ├── eval_classification_metrics.csv     # F1, sensitivity, specificity (classification only)
                                      ├── history.csv                         # Training history (loss, metrics per epoch)
                                      ├── mil_params.json                     # Model configuration and hyperparameters
                                      ├── predictions_train.parquet           # Training set predictions with true labels
                                      ├── predictions.parquet                 # Test set predictions with true labels
                                      └── slide_manifest.csv                  # Slide-level metadata and predictions
                                      ```
```

At the end of training, metrics are calculated automatically.

#### 2. Generate Plots

```bash
python dlif_pipeline/pipeline/plotting/plots_for_supplementary.py
```

**Plotting Script Configuration:**

if you want to plot for you experiment, edit the script ( as above ):

```python
training_type = 'standard'
#  training_type = 'standard'
sub0 = RESULTS_DIR
sub1 = 'C23'
sub2 = ''
task = 'classification'
# task = 'survival'
outcomes = ['os_months_24'] # os_months_6, DCR, OS_MONTHS (for survival)
path_pre = f'' # new_path
path_suf = f'{task}/{training_type}/hypothesis_driven/pyrad-noimp'
```

## Project Structure
```
Mil2/
├── mil_training/                     # Main training directory
│   ├── scripts/
│   │   └── pipeline/
│   │       ├── metrics/              # Metric calculation scripts
│   │       │   ├── __pycache__/
│   │       │   ├── metrics_utils/
│   │       │   │   ├── __pycache__/
│   │       │   │   ├── __init__.py
│   │       │   │   ├── classification_metrics.py
│   │       │   │   └── survival_metrics.py
│   │       │   ├── __init__.py
│   │       │   ├── calculate_average.py
│   │       │   ├── compute_scores.py
│   │       │   ├── delong_n.py               # AUC calculation with DeLong CI
│   │       │   ├── other_metrics.py          # F1, sensitivity, specificity
│   │       │   └── survival_cindex_ci.py     # C-index calculation
│   │       │
│   │       ├── plotting/             # Visualization scripts
│   │       │   ├── __pycache__/
│   │       │   ├── __init__.py
│   │       │   ├── line_no_plot_other_metrics.py
│   │       │   └── plots_for_supplementary.py
│   │       │
│   │       └── train/                # Training orchestration
│   │           ├── hyperparameters_tuning/
│   │           │   ├── __pycache__/
│   │           │   ├── grid_runner.py
│   │           │   └── hyperparam_tuning.py
│   │           ├── utils/
│   │           │   ├── __init__.py
│   │                   ├── config_utils.py
│   │                ├── dataset_utils.py
│   │           │   └── path_utils.py
│   │           ├── README
│   │           ├── run_training.py
│   │           ├── prepare_dataset.py
│   │           └── training_loop.py
│   │
│   └── utils/                        # General utilities
│       ├── __pycache__/
│       ├── __init__.py
│       ├── pipeline_utils.py
│       ├── df.parquet
│       ├── MIL.log
│       ├── README
│       ├── settings.json
│       ├── train.py
│       ├── __init__.py
│       └── pyrightconfig.py
│
├── MIL/                              # Core MIL package
```

## Troubleshooting

### Common Issues

|              Issue                  |            Solution                    |
|-------              ----------      |----------                 ----------   |
| **Matplotlib backend errors**       | Ensure `export MPLBACKEND=Agg` is set  |
| **Missing bag files**               | Set `prepare_dataset: true` in config  |
| **Out of memory errors**            | Reduce `batch_size` in hyperparameters |
| **Learning rate failures**          | Specify fixed `lr` in hyperparameters  |
| **Configuration validation errors** | Verify YAML syntax and required fields |



**Version**: 1.0.0  
**Last Updated**: November 2025 
