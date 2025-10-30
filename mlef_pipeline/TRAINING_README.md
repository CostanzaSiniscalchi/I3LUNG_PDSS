# MLEF Model Training Guide

This guide explains how to use the `train_mlef.py` script to train MLEF (Machine Learning Early Fusion) models.

## Overview

The script trains MLEF models for different data modality combinations and saves results in a structured folder format compatible with your analysis pipeline.

add what does it mean abbr

## Quick Start

### Basic Usage

Train all default modalities for OS_6 outcome with C23 subanalysis (main analysis):
```bash
python train_mlef.py --outcome OS_6 --subanalysis C23
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--outcome` | str | `OS_6` | Target outcome: `OS_6` or `OS_24` or `DCR` |
| `--subanalysis` | str | `C23` | Subgroup: `C23`, `C2`, `IO_ONLY`, `IO_CHT`, `LOW_PDL1`, `HIGH_PDL1`, `SQUAMOUS`, `ADENOCARCINOMA` |
| `--modalities` | list | all | Modalities to train: `RWD`, `RWD_DP`, `RWD_FMRAD`, `RWD_PYRAD`, `RWD_DP_FMRAD`, `RWD_DP_PYRAD` |
| `--model` | str | `LR` | Model type: `LR` (Logistic Regression), `RF` (Random Forest) |
| `--no-feature-selection` | flag | False | Disable automatic feature selection |
| `--output-dir` | str | `.` | Base output directory |

## Default Modality Combinations

When no `--modalities` argument is provided, the script trains these combinations in order:

1. **RWD** - Real-world data only (clinical features)
2. **RWD_DP** - RWD + Digital Pathology
3. **RWD_FMRAD** - RWD + First-order radiomic features
4. **RWD_PYRAD** - RWD + PyRadiomics features
5. **RWD_DP_FMRAD** - RWD + DP + First-order radiomic features
6. **RWD_DP_PYRAD** - RWD + DP + PyRadiomics features

### Common Examples

1. **Train for OS_24 outcome:**
```bash
python train_mlef.py --outcome OS_24 --subanalysis C23
```

2. **Train specific modalities only:**
```bash
python train_mlef.py --outcome OS_6 --subanalysis C23 --modalities RWD RWD_DP RWD_PYRAD
```

3. **Use Random Forest model:**
```bash
python train_mlef.py --outcome OS_6 --subanalysis C23 --model RF
```

4. **Train for different subanalysis:**
```bash
python train_mlef.py --outcome OS_6 --subanalysis IO_ONLY
```

5. **Disable feature selection:**
```bash
python train_mlef.py --outcome OS_6 --subanalysis C23 --no-feature-selection
```

## Output Structure

The script creates the following folder structure:

```
MLEF/
  C23/                          # Subanalysis name
    RWD/
      model_LR.pkl              # Trained model
      train_set.xlsx            # Training data (with FOLD column)
      test_set.xlsx             # Test data
      exval_set.xlsx            # External validation data (if available)
      prediction_CV.xlsx        # CV predictions (Subject, y_pred, y_true) add
      results.xlsx              # Performance metrics (CV, TEST, EXVAL AUCs)
    
    RWD_DP/
      model_LR.pkl
      train_set.xlsx
      test_set.xlsx
      exval_set.xlsx
      prediction_CV.xlsx
      results.xlsx
      rwd-only/                 # RWD-matched baseline model
        model_LR.pkl
        train_set.xlsx
        test_set.xlsx
        exval_set.xlsx
        prediction_CV.xlsx
        results.xlsx
    
    RWD_FMRAD/
      ... (same structure as RWD_DP)
    
    RWD_PYRAD/
      ... (same structure as RWD_DP)
    
    RWD_DP_FMRAD/
      ... (same structure as RWD_DP)
    
    RWD_DP_PYRAD/
      ... (same structure as RWD_DP)
    
    training_summary.xlsx       # Summary of all training runs
```

## Output Files Description

### Model Files
- **model_LR.pkl / model_RF.pkl** - Trained scikit-learn model (pickled)

### Dataset Files
- **train_set.xlsx** - Training set with features, outcome, and FOLD column for cross-validation
- **test_set.xlsx** - Held-out test set
- **exval_set.xlsx** - External validation set (UOC center data, if available)

### Prediction Files
- **prediction_CV.xlsx** - Cross-validated predictions on training set
  - Columns: `Subject`, `y_pred` (predicted probability), `y_true` (actual label)

### Results Files
- **results.xlsx** - Performance metrics for all data splits
  - Columns: `SET` (CV/TEST/EXVAL), `AUC` (mean ± std), `n` (sample size)
- **training_summary.xlsx** - Summary table of all trained models in the run

## RWD-Matched Models

For each multimodal combination (e.g., RWD_DP), the script also trains an **RWD-matched model** saved in the `rwd-only/` subfolder. This baseline model:
- Uses only RWD features
- Trains on the **same subjects** that have all modalities available
- Enables fair comparison between multimodal and RWD-only performance on the same dataset

The RWD modality itself doesn't have a `rwd-only/` subfolder (it would be redundant).

## Training Process

For each modality combination, the script performs:

1. **Data Loading** - Loads and merges modalities via early fusion
2. **Data Splitting** - Uses predefined train/test/external splits from `split.json`
4. **Imputation** - Fills missing values using iterative imputation
5. **Normalization** - Log-transforms skewed features and standardizes
6. **Feature Selection** - LASSO-based selection targeting 15±10 features (optional)
7. **Model Training** - Bayesian hyperparameter optimization with 5-fold LOCO-CV
8. **Evaluation** - Computes AUC with 95% CI on CV, test, and external sets
9. **Saving** - Exports model, datasets, predictions, and metrics

## Cross-Validation Strategy

The script uses **Leave-One-Center-Out Cross-Validation (LOCO-CV)**:
- Splits by patient center (INT, GHD, MH, SZMC, VHIO)
- Each fold holds out one center for validation
- Ensures robustness across different data sources
- Class balancing via sample weights in each fold

## Feature Selection

When enabled (default), the script:
1. Reduces FM-RAD features to ~100 via LASSO (if present)
2. Performs LASSO selection on all features targeting 15±10 features
3. Uses elbow method to find optimal feature count
4. Selects features that maximize discriminative power

## Model Training

The script supports three model types:

### Logistic Regression (LR) - Default
- Bayesian search over L1, L2, and ElasticNet penalties
- Hyperparameters: C, penalty, solver, l1_ratio
- 50 iterations of Bayesian optimization

### Random Forest (RF)
- Bayesian search over tree parameters
- Hyperparameters: n_estimators, max_depth, min_samples_split, etc.
- Out-of-bag scoring enabled

All models use:
- Balanced class weights
- Cross-validation scoring weighted by fold size
- Random state fixed at 10 for reproducibility

## Requirements

The script requires:
- Python 3.10+
- scikit-learn
- pandas
- numpy
- scipy
- joblib
- scikit-optimize


Data files needed in `mlef_pipeline/`:
- `split.json` - Train/test split definitions
- `submodel_features.json` - Features to exclude

Data files needed in `../data/` (relative to mlef_pipeline):
- `rwd.csv` - Real-world data
- `digital_pathology.csv` - Digital pathology features extracted ..
- `fmrad.csv` - Foundation model radiomic features
- `pyradiomics.csv` - PyRadiomics perturbation..features
- `outcomes.csv` - Outcome labels



## Advanced Usage

### Custom Modality Combinations

To train custom combinations, modify the `DEFAULT_MODALITIES` list in `train_mlef.py`:

```python
DEFAULT_MODALITIES = [
    [Mode.RWD],
    [Mode.RWD, Mode.DP],
    [Mode.RWD, Mode.FMRAD],
    # Add your custom combinations
    [Mode.RWD, Mode.DP, Mode.FMRAD, Mode.PYRAD],  # All modalities
]
```
### Integration with Visualization

After training, use the `plot_auc_from_folder` function from `graphs_i3lung.ipynb` to visualize results:

```python
from graphs_i3lung import plot_auc_from_folder

plot_auc_from_folder(
    root=r'c:\\Users\\Cristina\\AI-ON lab\\I3LUNG_PDSS',
    architecture="MLEF",
    analyses=["C23"],
    modality_order=["RWD", "RWD_DP", "RWD_FMRAD", "RWD_PYRAD", "RWD_DP_FMRAD", "RWD_DP_PYRAD"],
    show=True,
    save_dir="figures"
)
```

## Notes

- Training time varies by modality and model type (5-30 min per modality)
- Bayesian optimization uses 50 iterations per hyperparameter search
- Results are deterministic (random_state=10) for reproducibility
- External validation set (UOC) is automatically detected and evaluated if present
- The script is safe to interrupt - just restart to continue with remaining modalities


## Citation


