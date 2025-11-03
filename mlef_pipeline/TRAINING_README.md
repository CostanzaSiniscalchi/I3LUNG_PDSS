# MLEF Model Training Guide

This guide explains how to use the `train_mlef.py` script to train MLEF (Machine Learning Early Fusion) models.

## Overview

The script trains MLEF models for different data modality combinations and saves results in a structured folder format compatible with your analysis pipeline.

add what does it mean abbr

## Quick Start

### Step 1: Configure Models (Optional)

By default, the script uses a configuration file (`modality_models_config.json`) to specify which model to use for each modality combination. This allows:
- Different models for different modalities (e.g., LR for RWD, RF for RWD_DP)
- Different models for RWD-only matched analyses

Set your preferred model for each modality:
- `"LR"` = Logistic Regression 
- `"RF"` = Random Forest 

#### RWD-Matched Models (RWD-Only Analysis)

For each multimodal combination (e.g., RWD_DP), the script also trains an **RWD-matched model** saved in the `rwd-only/` subfolder. This baseline model:
- Uses only RWD features
- Trains on the **same subjects** that have all modalities available
- Enables fair comparison between multimodal and RWD-only performance on the same dataset

The RWD modality itself doesn't have a `rwd-only/` subfolder (it would be redundant).

#### Configuring RWD-Only Models

You can specify different models for RWD-only analyses in the configuration file:

```json
{
    "models": {
        "RWD_DP": "RF"
    },
    "rwd_only_models": {
        "RWD_DP": "LR"
    }
}
```

This configuration trains:
- Main RWD_DP model with Random Forest
- RWD-only matched model with Logistic Regression

#### Different Models for Main vs RWD-Only
If you want different models:
```json
{
    "models": {"RWD_DP": "RF"},
    "rwd_only_models": {"RWD_DP": "LR"}
}
```

If you want the same model for both, just specify in `models`:
```json
{
    "models": {"RWD_DP": "RF"}
}
```
The RWD-only will automatically use RF.

### Step 2: Run Training

Train all default modalities for OS_6 outcome with C23 subanalysis:
```bash
python train_mlef.py --outcome OS_6 --subanalysis C23
```

The script will automatically use the models specified in your configuration file.


## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--outcome` | str | `OS_6` | Target outcome: `OS_6` or `OS_24` or `DCR` |
| `--subanalysis` | str | `C23` | Subgroup: `C23`, `C2`, `IO_ONLY`, `IO_CHT`, `LOW_PDL1`, `HIGH_PDL1`, `SQUAMOUS`, `ADENOCARCINOMA` |
| `--modalities` | list | all | Modalities to train: `RWD`, `RWD_DP`, `RWD_FMRAD`, `RWD_PYRAD`, `RWD_DP_FMRAD`, `RWD_DP_PYRAD` |
| `--model` | str | `LR` | Default model type (used if not specified in config): `LR` (Logistic Regression), `RF` (Random Forest) |
| `--no-feature-selection` | flag | False | Disable automatic feature selection |
| `--output-dir` | str | `.` | Base output directory |

**Note:** Model selection now primarily uses the configuration file. The `--model` argument serves as a fallback default for any modalities not specified in the config.

## Default Modality Combinations

When no `--modalities` argument is provided, the script trains these combinations in order:

1. **RWD** - Real-world data only (clinical features)
2. **RWD_DP** - RWD + Digital Pathology
3. **RWD_FMRAD** - RWD + Foundation Model radiomic features
4. **RWD_PYRAD** - RWD + PyRadiomics features
5. **RWD_DP_FMRAD** - RWD + DP + Foundation Model radiomic features
6. **RWD_DP_PYRAD** - RWD + DP + PyRadiomics features

### Common Examples

1. **Train with configuration file (recommended):**
```bash
# Uses models specified in modality_models_config.json
python train_mlef.py --outcome OS_6 --subanalysis C23
```

2. **Train for OS_24 outcome:**
```bash
python train_mlef.py --outcome OS_24 --subanalysis C23
```

3. **Train specific modalities only:**
```bash
# Only trains RWD, RWD_DP, and RWD_PYRAD
python train_mlef.py --outcome OS_6 --subanalysis C23 --modalities RWD RWD_DP RWD_PYRAD
```

4. **Override default model (for unconfigured modalities):**
```bash
# Uses RF as default for any modality not in config
python train_mlef.py --outcome OS_6 --subanalysis C23 --model RF
```

5. **Train for different subanalysis:**
```bash
python train_mlef.py --outcome OS_6 --subanalysis IO_ONLY
```

6. **Disable feature selection:**
```bash
python train_mlef.py --outcome OS_6 --subanalysis C23 --no-feature-selection
```

7. **Custom output directory:**
```bash
python train_mlef.py --outcome OS_6 --subanalysis C23 --output-dir ./results
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

## Training Process

For each modality combination, the script performs:

1. **Data Loading** - Loads and merges modalities via early fusion
2. **Data Splitting** - Uses predefined train/test/external splits from `split.json`
3. **Imputation** - Fills missing values using iterative imputation
4. **Normalization** - Log-transforms skewed features and standardizes
5. **Feature Selection** - LASSO-based selection targeting 15±10 features (optional)
6. **Model Training** - Bayesian hyperparameter optimization with 5-fold LOCO-CV
7. **Evaluation** - Computes AUC with 95% CI on CV, test, and external sets
8. **Saving** - Exports model, datasets, predictions, and metrics

### Training Output

When you run the script, it displays the configuration and progress:

```
================================================================================
MLEF MODEL TRAINING
================================================================================
Outcome: OS_6
Subanalysis: C23
Default model type: LR
Feature selection: LASSO
Output directory: /home/user/I3LUNG_PDSS/mlef_pipeline

Modalities to train (6):
  - RWD (Model: LR)
  - RWD_DP (Model: RF, RWD-only: LR)
  - RWD_FMRAD (Model: LR, RWD-only: LR)
  - RWD_PYRAD (Model: LR, RWD-only: LR)
  - RWD_DP_FMRAD (Model: RF, RWD-only: LR)
  - RWD_DP_PYRAD (Model: RF, RWD-only: LR)
================================================================================

[1/6] Processing RWD...
Using model: LR
1. Loading data...
2. Splitting data...
3. Preparing features and target...
...
✓ Training completed successfully!
  CV AUC: 0.725 ± 0.045
  Test AUC: 0.698 ± 0.062
  External AUC: 0.712 ± 0.089

[2/6] Processing RWD_DP...
Using model: RF
...
Training RWD-only with model: LR
...
```

This shows which model is being used for each modality and its RWD-only analysis.

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

The script supports two model types, configured via `modality_models_config.json`:

### Logistic Regression (LR)
- Bayesian search over L1, L2, and ElasticNet penalties
- Hyperparameters: C, penalty, solver, l1_ratio
- 50 iterations of Bayesian optimization
- Best for: Interpretability, fewer features, linear relationships

### Random Forest (RF)
- Bayesian search over tree parameters
- Hyperparameters: n_estimators, max_depth, min_samples_split, etc.
- Out-of-bag scoring enabled
- Best for: Complex interactions, non-linear relationships, larger feature sets

All models use:
- Balanced class weights
- Cross-validation scoring weighted by fold size
- Random state fixed at 10 for reproducibility

### Configuring Models

Edit `mlef_pipeline/modality_models_config.json` to specify which model to use for each modality:

```json
{
    "models": {
        "RWD": "LR",
        "RWD_DP": "RF",
        "RWD_FMRAD": "LR",
        "RWD_PYRAD": "LR",
        "RWD_DP_FMRAD": "RF",
        "RWD_DP_PYRAD": "RF"
    },
    "rwd_only_models": {
        "RWD_DP": "LR",
        "RWD_FMRAD": "LR",
        "RWD_PYRAD": "LR",
        "RWD_DP_FMRAD": "LR",
        "RWD_DP_PYRAD": "LR"
    }
}
```

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



## Recommended Workflow

### 1. Configure Models
Edit `modality_models_config.json` to specify your model choices.


### 2. Run Training
```bash
python train_mlef.py --outcome OS_6 --subanalysis C23
```

### 3. Review Results
Check the `training_summary.xlsx` file for overview of all trained models:
```bash
# Summary location:
MLEF/OS_6/C23/training_summary.xlsx
```

### 4. Compare Modalities
Use the visualization notebook to create comparison plots:
```python
from graphs_classification import plot_auc_results
plot_auc_results(architecture='MLEF', outcome='OS_24', analyses=['C23'])
```

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

After training, use the `plot_auc_results` function from `graphs_classification.ipynb` to visualize results:

```python
from graphs_classification import plot_auc_results

plot_auc_results(
    architecture="MLEF",
    outcome='OS_24',
    analyses=["C23"],
    modality_order=["RWD", "RWD_DP", "RWD_FMRAD", "RWD_PYRAD", "RWD_DP_FMRAD", "RWD_DP_PYRAD"],
    show=True
)
```

## Troubleshooting

### Model Selection Priority
The script determines which model to use in this order:
1. **Configuration file** (`modality_models_config.json`) - Primary source
2. **Command-line `--model`** - Fallback for unconfigured modalities
3. **Default (LR)** - If neither is specified

### Configuration File Not Found
If you see `Warning: Config file modality_models_config.json not found`:
- The script will use the `--model` argument as default for all modalities
- Create the config file or use absolute path in the script

### Invalid Model in Configuration
If you see `Warning: Invalid model 'XYZ'`:
- Check that model names in config are exactly `"LR"` or `"RF"` (case-sensitive)
- The script will fall back to the default model



## Notes

- Training time varies by modality and model type (5-15 min per modality)
- Bayesian optimization uses 50 iterations per hyperparameter search
- Results are deterministic (random_state=10) for reproducibility
- External validation set (UOC) is automatically detected and evaluated if present
- The script is safe to interrupt - just restart to continue with remaining modalities
- Configuration changes take effect immediately on next run


## Citation


