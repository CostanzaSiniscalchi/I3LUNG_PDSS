# MIL: Multiple Instance Learning for Medical Imaging

A deep learning pipeline for survival analysis and classification tasks using Multiple Instance Learning (MIL) on medical imaging data.

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
```bash
# Clone repository
git clone git@github.com:AI-ON-Laboratory/MIL.git
cd MIL

# Create and activate conda environment
conda create -n mil python=3.9
conda activate mil

# Install dependencies
pip install -r I3LUNG_PDSS/dlif_pipeline/requirements.txt

# Configure environment
export MPLBACKEND=Agg  # Required for headless plotting
```

## Project Structure

The expected directory structure is as follows:
```
I3LUNG_PDSS/
└── dlif_pipeline/                          # This repository
    ├── configs/                            # Configuration files
    ├── pipeline/                           # Training scripts
    ├── data/                               # Data directory
    │   ├── annotations/                    # Annotation files
    │   ├── features_dataset_radfm.parquet # RadFM feature dataset
    │   └── features_dataset_radpy_fixed.parquet # RadPy feature dataset
    ├── bags/                               # Bag-level features
    │   ├── radfm/                         # RadFM modality bags
    │   └── radpy/                         # RadPy modality bags
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
- `00-config-survival-cv.yaml` - Cross-validation configuration
- `01-config-survival-standard.yaml` - Standard training configuration
- `02-config-survival-eval.yaml` - External validation configuration

## Usage

### Classification Tasks

#### 1. Cross-Validation (Leave-One-Center-Out)

Performs leave-one-center-out cross-validation across multiple centers:
- GHD
- INT
- MH
- SZMC
- VHIO
```bash
python pipeline/train.py \
  --config configs/00-config-classification-cv.yaml \
  --base_dir ./results
```

#### 2. Standard Training

Trains on all data from the above sites and evaluates on held-out data from each site:
```bash
python pipeline/train.py \
  --config configs/01-config-classification-standard.yaml \
  --base_dir ./results
```

#### 3. External Validation

Evaluates on external held-out center (UOC):
```bash
python pipeline/train.py \
  --config configs/02-config-classification-eval.yaml \
  --base_dir ./results
```

### Survival Tasks

Use the same commands with survival configuration files:

#### 1. Cross-Validation
```bash
python pipeline/train.py \
  --config configs/00-config-survival-cv.yaml \
  --base_dir ./results
```

#### 2. Standard Training
```bash
python pipeline/train.py \
  --config configs/01-config-survival-standard.yaml \
  --base_dir ./results
```

#### 3. External Validation
```bash
python pipeline/train.py \
  --config configs/02-config-survival-eval.yaml \
  --base_dir ./results
```

### Calculate Metrics

```bash
python pipeline/metrics/compute_metrics_from_config.py \
  --config configs/00-config-classification-cv.yaml \
  --base_dir ./results
python pipeline/metrics/compute_metrics_from_config.py \
  --config configs/01-config-classification-standard.yaml \
  --base_dir ./results
```

### Create Plots

```
bash
python pipeline/plotting/plot_from_config.py \
  --config configs/00-config-classification-cv.yaml \
  --base_dir ./results
python pipeline/plotting/plot_from_config.p \
  --config configs/01-config-classification-standard.yaml \
  --base_dir ./results
```

## Output Structure

Training outputs are organized hierarchically in the results directory:
```
results/
└── cohort2/
    └── mil/
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

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{mil_pipeline_2024,
  title     = {Multi-Instance Learning Pipeline for Medical Imaging},
  author    = {Sacco, Matteo and Lerma, Ludovica},
  year      = {2024},
  url       = {https://github.com/AI-ON-Laboratory/MIL},
  publisher = {AI-ON Laboratory}
}
```

## Support

For assistance:

- **Issues**: [GitHub Issues](https://github.com/AI-ON-Laboratory/MIL/issues)
- **Documentation**: See `docs/` directory
- **Examples**: Review `configs-test/` for configuration templates
- **Contact**: @matte-esse, @LudoLe

## License

[Specify license here - e.g., MIT, Apache 2.0, etc.]

---

**Version**: 1.0.0  
**Last Updated**: October 2024  
**Maintainers**: Matteo Sacco, Ludovica Lerma