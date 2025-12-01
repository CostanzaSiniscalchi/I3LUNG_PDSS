"""
MLEF Model Training Script

This script trains MLEF (Multi-Level Early Fusion) models for different modality combinations.
It saves models, datasets, predictions, and results in a structured folder format.

Usage:
    python mlef_pipeline/train_mlef.py --outcome OS_6 --subanalysis C23
"""

import os
import sys
import json
import argparse
import joblib
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.utils import compute_sample_weight
import random
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, confusion_matrix, make_scorer

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='skopt')

# Make runs deterministic across numpy and python's random where applicable
np.random.seed(10)
random.seed(10)
# Ensure hash-based operations are deterministic
os.environ['PYTHONHASHSEED'] = '10'

# Add mlef_pipeline to path
sys.path.insert(0, str(Path(__file__).parent / 'mlef_pipeline'))

from data_curation import DataLoader, SafeGroupKFold
from i3l_statistics import Statistics
from i3l_ml import ML
from enums import Mode, Outcome, Subanalysis, Model

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')


# Default modality combinations to train (in order)
DEFAULT_MODALITIES = [
    [Mode.RWD],
    [Mode.RWD, Mode.DP],
    [Mode.RWD, Mode.FMRAD],
    [Mode.RWD, Mode.PYRAD],
    [Mode.RWD, Mode.DP, Mode.FMRAD],
    [Mode.RWD, Mode.DP, Mode.PYRAD],
]

# Mapping from modality list to folder name
MODALITY_TO_FOLDER = {
    frozenset([Mode.RWD]): 'RWD',
    frozenset([Mode.RWD, Mode.DP]): 'RWD_DP',
    frozenset([Mode.RWD, Mode.FMRAD]): 'RWD_FMRAD',
    frozenset([Mode.RWD, Mode.PYRAD]): 'RWD_PYRAD',
    frozenset([Mode.RWD, Mode.DP, Mode.FMRAD]): 'RWD_DP_FMRAD',
    frozenset([Mode.RWD, Mode.DP, Mode.PYRAD]): 'RWD_DP_PYRAD',
}


def get_modality_folder_name(modes: List[Mode]) -> str:
    """Convert a list of modes to the corresponding folder name."""
    mode_set = frozenset(modes)
    if mode_set in MODALITY_TO_FOLDER:
        return MODALITY_TO_FOLDER[mode_set]
    
    # Fallback: join mode names
    return '_'.join(sorted([m.value for m in modes]))


def load_modality_models_config(outcome: str, subanalysis: str, config_path: str = 'mlef_pipeline/modality_models_config.json') -> Tuple[dict, dict]:
    """Load the modality models configuration file.
    
    Returns:
        Tuple of (models_config, rwd_only_models_config)
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            config = config.get(subanalysis.value, {})
            config = config.get(outcome.value, {})
        return config.get('models', {}), config.get('rwd_only_models', {})
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using default model.")
        return {}, {}


def get_model_for_modality(modality_folder: str, config: dict, default_model: Model = Model.LR.value) -> Model:
    """Get the model type for a specific modality from the config."""
    # Se default_model è stato passato esplicitamente, usalo sempre
    if hasattr(default_model, "_explicit") and default_model._explicit:
        print("EXPLICIT")
        return default_model
    model_str = config.get(modality_folder, default_model)
    try:
        return Model[model_str]
    except KeyError:
        print(f"Warning: Invalid model '{model_str}' for modality {modality_folder}. Using {default_model.value}.")
        return default_model


def get_rwd_only_model_for_modality(modality_folder: str, rwd_only_config: dict, 
                                     parent_model: Model) -> Model:
    """Get the model type for RWD-only analysis from the config.
    
    Args:
        modality_folder: The modality name (e.g., 'RWD_DP')
        rwd_only_config: The rwd_only_models configuration dictionary
        parent_model: The model used for the parent modality (fallback)
        default_model: The overall default model
    
    Returns:
        Model to use for RWD-only analysis
    """
    if modality_folder in rwd_only_config:
        model_str = rwd_only_config[modality_folder]
        try:
            return Model[model_str]
        except KeyError:
            print(f"Warning: Invalid RWD-only model '{model_str}' for modality {modality_folder}. Using parent model {parent_model.value}.")
            return parent_model
    else:
        # If not specified in rwd_only_config, use parent modality's model
        return parent_model


def create_output_dirs(base_path: Path, subanalysis: str, modality_folder: str, outcome: str) -> Path:
    """Create output directory structure and return the modality path."""
    output_dir = base_path / 'results' / outcome / subanalysis / modality_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create RWD-matched subdirectory (for non-RWD modalities)
    if modality_folder != 'RWD':
        rwd_only_dir = output_dir / 'rwd-only'
        rwd_only_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def save_datasets(output_dir: Path, train_set: pd.DataFrame, test_set: pd.DataFrame, 
                  ext_set: pd.DataFrame, train_folds: pd.Series):
    """Save train, test, and external validation sets."""
    # Add fold information to train set
    train_with_folds = train_set.copy()
    train_with_folds['FOLD'] = train_folds.values
    
    train_with_folds.to_csv(output_dir / 'train_set.csv', index=True)
    test_set.to_csv(output_dir / 'test_set.csv', index=True)
    
    if not ext_set.empty:
        ext_set.to_csv(output_dir / 'exval_set.csv', index=True)
    
    print(f"  ✓ Saved datasets to {output_dir}")


def compute_metrics_with_ci(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                           threshold: float = 0.5) -> dict:
    """
    Compute classification metrics with confidence intervals.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        Dictionary with metrics and their std (from CI)
    """
    stats = Statistics()
    
    # Calculate AUC with CI
    auc, ci = stats.auc_roc_ci(y_true, y_pred_proba, alpha=0.95)
    
    if pd.isna(auc) or pd.isna(ci).any():
        auc_std = np.nan
    else:
        auc_std = (ci[1] - ci[0]) / 2
    
    # Convert probabilities to binary predictions
    y_pred_binary = np.where(np.isnan(y_pred_proba), np.nan, (y_pred_proba >= threshold).astype(int))
    
    # Calculate F1 macro
    f1_macro = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
    
    # Calculate confusion matrix for sensitivity and specificity
    cm = confusion_matrix(y_true, y_pred_binary)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        
        # Sensitivity (recall, TPR)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Specificity (TNR)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = np.nan
        specificity = np.nan
    
    return {
        'auc': auc,
        'auc_std': auc_std,
        'f1_macro': f1_macro,
        'sensitivity': sensitivity,
        'specificity': specificity
    }


def compute_cv_predictions(model, X: pd.DataFrame, y: pd.Series, cv_splits) -> Tuple[pd.Series, dict]:
    """
    Compute cross-validation predictions using LOCO-CV.
    
    Returns:
        y_pred_cv: Cross-validated predictions (pd.Series with original index)
        cv_metrics: Dictionary with CV metrics (auc, auc_std, f1_macro, f1_macro_std, 
                    sensitivity, sensitivity_std, specificity, specificity_std)
    """
    from sklearn.base import clone
    
    # Initialize Series with NaN to clearly mark unprocessed folds
    y_pred_cv = pd.Series(np.nan, index=y.index, name='y_pred')
    
    # Generate predictions for each fold
    for train_idx, val_idx in cv_splits:
        # Clone model for each fold
        fold_model = clone(model)
        
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Check if training fold has only one class - skip if so
        if len(np.unique(y_train_fold)) < 2:
            print(f"  Warning: Skipping fold with only one class in training set")
            # Set predictions to NaN for this fold
            y_pred_cv.iloc[val_idx] = np.nan
            # Don't include this fold in metrics
            continue
        
        # Compute sample weights for this fold
        sample_weight = compute_sample_weight(class_weight='balanced', y=y_train_fold)
        
        # Fit model on this fold
        fold_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weight)
        
        # Predict on validation fold - use .iloc to set by position
        y_pred_proba_fold = fold_model.predict_proba(X_val_fold)[:, 1]
        y_pred_cv.iloc[val_idx] = y_pred_proba_fold
    
    # Compute overall CV metrics (global across all predictions)
    # Filter out NaN values (from skipped folds)
    valid_mask = ~np.isnan(y_pred_cv.values)
    
    if not valid_mask.any():
        # All folds were skipped - return NaN metrics
        print("  Warning: All folds were skipped due to single-class training sets")
        return y_pred_cv, {
            'auc': np.nan,
            'auc_std': np.nan,
            'f1_macro': np.nan,
            'f1_macro_std': np.nan,
            'sensitivity': np.nan,
            'sensitivity_std': np.nan,
            'specificity': np.nan,
            'specificity_std': np.nan
        }
    
    y_valid = y.values[valid_mask]
    y_pred_valid = y_pred_cv.values[valid_mask]
    
    stats = Statistics()
    auc, ci = stats.auc_roc_ci(y_valid, y_pred_valid, alpha=0.95)
    auc_std = (ci[1] - ci[0]) / 2
    
    # Define custom scorers for sensitivity and specificity
    def sensitivity_score(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def specificity_score(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Prepare sample weights for CV scoring
    sample_weights = compute_sample_weight(class_weight='balanced', y=y)
    
    # Create a function that returns fresh CV splits
    def get_cv_splits():
        return cv_splits
    
    # Compute weighted CV metrics using ML.get_weighted_cv
    ml = ML()
    
    f1_macro_mean, f1_macro_std = ml.get_weighted_cv(
        model, X, y, 
        cv_getter=get_cv_splits,
        scorer='f1_macro',
        sample_weight=sample_weights
    )
    
    sensitivity_mean, sensitivity_std = ml.get_weighted_cv(
        model, X, y,
        cv_getter=get_cv_splits,
        scorer=make_scorer(sensitivity_score),
        sample_weight=sample_weights
    )
    
    specificity_mean, specificity_std = ml.get_weighted_cv(
        model, X, y,
        cv_getter=get_cv_splits,
        scorer=make_scorer(specificity_score),
        sample_weight=sample_weights
    )
    
    return y_pred_cv, {
        'auc': auc,
        'auc_std': auc_std,
        'f1_macro': f1_macro_mean,
        'f1_macro_std': f1_macro_std,
        'sensitivity': sensitivity_mean,
        'sensitivity_std': sensitivity_std,
        'specificity': specificity_mean,
        'specificity_std': specificity_std
    }


def train_and_evaluate_modality(
    modes: List[Mode],
    outcome: Outcome,
    subanalysis: Subanalysis,
    model_type: Model = Model.LR,
    base_path: Path = Path('.'),
    select_features: bool = True
) -> dict:
    """
    Train and evaluate a model for a specific modality combination.
    
    Args:
        modes: List of data modalities to include
        outcome: Target outcome to predict
        subanalysis: Subgroup analysis to perform
        model_type: Type of model to train (LR, RF, XGB)
        base_path: Base directory for saving results
        select_features: Whether to perform feature selection
    
    Returns:
        Dictionary with results and metrics
    """
    print(f"\n{'='*80}")
    print(f"Training {get_modality_folder_name(modes)} - {outcome.value} - {subanalysis.value}")
    print(f"{'='*80}")
    
    # Initialize helpers
    dl = DataLoader()
    ml = ML()
    stats = Statistics()
    
    # Create output directory
    modality_folder = get_modality_folder_name(modes)
    output_dir = create_output_dirs(base_path, subanalysis.value, modality_folder, outcome.value)
    
    # 1. Load and prepare data
    print("1. Loading data...")
    dataset = dl.create_dataset(
        modes=modes,
        outcome=outcome.value,
        subanalysis=subanalysis
    )
    
    # 2. Split data
    print("2. Splitting data...")
    
    train_set = dataset[dataset['SET'] == 'TRAIN'].set_index('Subject').drop(columns=['SET'])
    test_set = dataset[dataset['SET'] == 'TEST'].set_index('Subject').drop(columns=['SET', 'CENTER'])
    ext_set = dataset[dataset['SET'] == 'EXVAL'].set_index('Subject').drop(columns=['SET', 'CENTER'])
    
    train_folds = train_set['CENTER']
    train_set = train_set.drop(columns=['CENTER'])

    # 3. Separate features and target
    print("3. Preparing features and target...")
    X_train, y_train = train_set.drop(columns=[outcome.value]), train_set[outcome.value]
    X_test, y_test = test_set.drop(columns=[outcome.value]), test_set[outcome.value]
    X_ext, y_ext = ext_set.drop(columns=[outcome.value]), ext_set[outcome.value]
    
    # Convert outcome to binary if needed
    # y_train = dl.get_outcome(y_train_raw, outcome)
    # y_test = dl.get_outcome(y_test_raw, outcome)
    # y_ext = dl.get_outcome(y_ext_raw, outcome) if not ext_set.empty else pd.Series()
    
    with open('mlef_pipeline/submodel_features.json', 'r') as f:
        submodel_features = json.load(f)
        submodel_features = [f for f in submodel_features if f in X_train.columns]
    
    X_train = X_train.drop(columns=submodel_features, errors='ignore')
    X_test = X_test.drop(columns=submodel_features, errors='ignore')
    X_ext = X_ext.drop(columns=submodel_features, errors='ignore')
    # 4. Imputation
    print("4. Imputing missing values...")

    print(X_train.shape)
    X_train_imputed, imputer = dl.impute_df(X_train)
    X_ext_imputed, _ = dl.impute_df(X_ext, imputer=imputer)
    X_test_imputed, _ = dl.impute_df(X_test, imputer=imputer)

    # 5. Normalization
    print("5. Normalizing features...")
    X_train_scaled, scaler, to_standard_normalize, to_log_normalize = dl.normalize(X_train_imputed)
    
    if not X_ext.empty:
        X_ext_scaled, _, _, _ = dl.normalize(
            X_ext_imputed, scaler=scaler,
            to_standard_normalize=to_standard_normalize,
            to_log_normalize=to_log_normalize
        )
    else:
        X_ext_scaled = X_ext

    X_test_scaled, _, _, _ = dl.normalize(
            X_test_imputed, scaler=scaler, 
            to_standard_normalize=to_standard_normalize, 
            to_log_normalize=to_log_normalize
        )
    
    # 6. Setup cross-validation
    print("6. Setting up cross-validation...")
    cv = SafeGroupKFold(n_splits=len(train_folds.unique()))
    
    def get_cv_splits():
        return list(cv.split(X_train_scaled, y_train, groups=train_folds))
    
    # 7. Train model
    print(f"7. Training {model_type.value} model...")
    model = ml.train_classification_model(
        X=X_train_scaled,
        y=y_train,
        model_name=model_type,
        cv=lambda: iter(get_cv_splits()),
        select_features=select_features
    )
    
    selected_features = model.feature_names_in_
    print(f"  ✓ Selected {len(selected_features)} features")
    
    # Filter to selected features
    X_train_final = X_train_scaled[selected_features]
    X_test_final = X_test_scaled[selected_features]
    X_ext_final = X_ext_scaled[selected_features] if not X_ext_scaled.empty else X_ext_scaled
    
    # 8. Compute cross-validation predictions
    print("8. Computing CV predictions...")
    # use GroupKFold with centers as groups
    cv = GroupKFold(n_splits=len(train_folds.unique()))
    def get_cv_splits():
        return list(cv.split(X_train_scaled, y_train, groups=train_folds))
    
    y_proba_cv, cv_metrics = compute_cv_predictions(
        model, X_train_final, y_train, get_cv_splits()
    )
    
    # 9. Make predictions on all sets
    print("9. Making predictions on test and external sets...")
    # Refit on full training set for final predictions
    # sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
    # model.fit(X_train_final, y_train, sample_weight=sample_weight)
    
    y_pred_test = model.predict_proba(X_test_final)[:, 1]
    test_metrics = compute_metrics_with_ci(y_test.values, y_pred_test)
    
    if not X_ext_final.empty:
        y_pred_ext = model.predict_proba(X_ext_final)[:, 1]
        ext_metrics = compute_metrics_with_ci(y_ext.values, y_pred_ext)
    else:
        y_pred_ext = np.array([])
        ext_metrics = {
            'auc': np.nan,
            'auc_std': np.nan,
            'f1_macro': np.nan,
            'sensitivity': np.nan,
            'specificity': np.nan
        }
    
    # 10. Save model
    print("10. Saving model...")
    model_path = output_dir / f'model_{model_type.value}.pkl'
    joblib.dump(model, model_path)
    print(f"  ✓ Model saved to {model_path}")

    # 11. Save datasets (with outcome)
    print("11. Saving datasets...")
    train_set_with_outcome = X_train_scaled.copy()
    train_set_with_outcome[outcome.value] = y_train
    test_set_with_outcome = X_test_scaled.copy()
    test_set_with_outcome[outcome.value] = y_test
    ext_set_with_outcome = X_ext_scaled.copy()
    if not ext_set.empty:
        ext_set_with_outcome[outcome.value] = y_ext
    
    save_datasets(output_dir, train_set_with_outcome, test_set_with_outcome, 
                  ext_set_with_outcome, train_folds)
    
    # 12. Save predictions
    print("12. Saving predictions...")
    
    # Save CV predictions
    cv_predictions = pd.DataFrame({
        'Subject': y_proba_cv.index,  # Now y_pred_cv is a Series with the correct index
        'y_proba': y_proba_cv.values,
        'y_pred': np.where(~y_proba_cv.isna(), (y_proba_cv >= 0.5).astype(int), np.nan),
        'y_true': y_train.values
    })
    cv_predictions.to_csv(output_dir / 'prediction_CV.csv', index=False)
    
    # Save TEST predictions
    test_predictions = pd.DataFrame({
        'Subject': X_test_final.index,
        'y_proba': y_pred_test,
        'y_pred': np.where(~pd.isna(y_pred_test), (y_pred_test >= 0.5).astype(int), np.nan),
        'y_true': y_test.values
    })
    test_predictions.to_csv(output_dir / 'prediction_TEST.csv', index=False)
    
    # Save EXVAL predictions (if available)
    if not X_ext_final.empty:
        exval_predictions = pd.DataFrame({
            'Subject': X_ext_final.index,
            'y_proba': y_pred_ext,
            'y_pred': np.where(~pd.isna(y_pred_ext), (y_pred_ext >= 0.5).astype(int), np.nan),
            'y_true': y_ext.values
        })
        exval_predictions.to_csv(output_dir / 'prediction_EXVAL.csv', index=False)

    # 13. Save results
    print("13. Saving results...")
    results = pd.DataFrame({
        'SET': ['CV', 'TEST', 'EXVAL'],
        'AUC': [
            f'{cv_metrics["auc"]:.2f} ± {cv_metrics["auc_std"]:.2f}',
            f'{test_metrics["auc"]:.2f} ± {test_metrics["auc_std"]:.2f}',
            f'{ext_metrics["auc"]:.2f} ± {ext_metrics["auc_std"]:.2f}' if not np.isnan(ext_metrics["auc"]) else 'N/A'
        ],
        'F1_MACRO': [
            f'{cv_metrics["f1_macro"]:.2f} ± {cv_metrics["f1_macro_std"]:.2f}',
            f'{test_metrics["f1_macro"]:.2f}',
            f'{ext_metrics["f1_macro"]:.2f}' if not np.isnan(ext_metrics["f1_macro"]) else 'N/A'
        ],
        'SENSITIVITY': [
            f'{cv_metrics["sensitivity"]:.2f} ± {cv_metrics["sensitivity_std"]:.2f}',
            f'{test_metrics["sensitivity"]:.2f}',
            f'{ext_metrics["sensitivity"]:.2f}' if not np.isnan(ext_metrics["sensitivity"]) else 'N/A'
        ],
        'SPECIFICITY': [
            f'{cv_metrics["specificity"]:.2f} ± {cv_metrics["specificity_std"]:.2f}',
            f'{test_metrics["specificity"]:.2f}',
            f'{ext_metrics["specificity"]:.2f}' if not np.isnan(ext_metrics["specificity"]) else 'N/A'
        ],
        'n': [len(y_train), len(y_test), len(y_ext) if not ext_set.empty else 0]
    })
    results.to_excel(output_dir / 'results.xlsx', index=False)
    
    print("\n✓ Training completed successfully!")
    print(f"  CV AUC: {cv_metrics['auc']:.3f} ± {cv_metrics['auc_std']:.3f}")
    print(f"  CV F1 Macro: {cv_metrics['f1_macro']:.3f} ± {cv_metrics['f1_macro_std']:.3f}")
    print(f"  CV Sensitivity: {cv_metrics['sensitivity']:.3f} ± {cv_metrics['sensitivity_std']:.3f}")
    print(f"  CV Specificity: {cv_metrics['specificity']:.3f} ± {cv_metrics['specificity_std']:.3f}")
    print(f"  Test AUC: {test_metrics['auc']:.3f} ± {test_metrics['auc_std']:.3f}")
    print(f"  Test F1 Macro: {test_metrics['f1_macro']:.3f}")
    print(f"  Test Sensitivity: {test_metrics['sensitivity']:.3f}")
    print(f"  Test Specificity: {test_metrics['specificity']:.3f}")
    if not np.isnan(ext_metrics['auc']):
        print(f"  External AUC: {ext_metrics['auc']:.3f} ± {ext_metrics['auc_std']:.3f}")
        print(f"  External F1 Macro: {ext_metrics['f1_macro']:.3f}")
        print(f"  External Sensitivity: {ext_metrics['sensitivity']:.3f}")
        print(f"  External Specificity: {ext_metrics['specificity']:.3f}")
    
    return {
        'modality': modality_folder,
        'cv_auc': cv_metrics['auc'],
        'cv_auc_std': cv_metrics['auc_std'],
        'cv_f1_macro': cv_metrics['f1_macro'],
        'cv_f1_macro_std': cv_metrics['f1_macro_std'],
        'cv_sensitivity': cv_metrics['sensitivity'],
        'cv_sensitivity_std': cv_metrics['sensitivity_std'],
        'cv_specificity': cv_metrics['specificity'],
        'cv_specificity_std': cv_metrics['specificity_std'],
        'test_auc': test_metrics['auc'],
        'test_auc_std': test_metrics['auc_std'],
        'test_f1_macro': test_metrics['f1_macro'],
        'test_sensitivity': test_metrics['sensitivity'],
        'test_specificity': test_metrics['specificity'],
        'ext_auc': ext_metrics['auc'],
        'ext_auc_std': ext_metrics['auc_std'],
        'ext_f1_macro': ext_metrics['f1_macro'],
        'ext_sensitivity': ext_metrics['sensitivity'],
        'ext_specificity': ext_metrics['specificity'],
        'n_features': len(selected_features),
        'n_train': len(y_train),
        'n_test': len(y_test),
        'n_ext': len(y_ext) if not ext_set.empty else 0
    }


def train_rwd_matched_model(
    modes: List[Mode],
    outcome: Outcome,
    subanalysis: Subanalysis,
    model_type: Model = Model.LR,
    base_path: Path = Path('.'),
    select_features: bool = True
):
    """
    Train RWD-matched model (using only subjects that have all modalities).
    This is saved in the rwd-only subfolder for non-RWD modalities.
    """
    modality_folder = get_modality_folder_name(modes)
    
    # Skip if this is already RWD-only
    if modality_folder == 'RWD':
        return None
    
    print(f"\n{'='*80}")
    print(f"Training RWD-matched model for {modality_folder}")
    print(f"{'='*80}")
    
    # Get subjects that have all modalities (intersection)
    dl = DataLoader()
    ml = ML()
    stats = Statistics()
    
    # Load data for the full modality
    full_dataset = dl.create_dataset(
        modes=modes,
        outcome=outcome.value,
        subanalysis=subanalysis
    )
    
    # Get subjects from full modality (these have all data)
    matched_subjects = set(full_dataset['Subject'].values)
    
    # Now create RWD-only dataset but filter to matched subjects
    rwd_dataset = dl.create_dataset(
        modes=[Mode.RWD],
        outcome=outcome.value,
        subanalysis=subanalysis
    )
    rwd_dataset = rwd_dataset[rwd_dataset['Subject'].isin(matched_subjects)]
    
    print(f"  Using {len(matched_subjects)} matched subjects")
    
    # Create output directory
    output_dir = base_path / 'results' / outcome.value / subanalysis.value / modality_folder / 'rwd-only'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_set = rwd_dataset[rwd_dataset['SET'] == 'TRAIN'].set_index('Subject').drop(columns=['SET'])
    test_set = rwd_dataset[rwd_dataset['SET'] == 'TEST'].set_index('Subject').drop(columns=['SET', 'CENTER'])
    ext_set = rwd_dataset[rwd_dataset['SET'] == 'EXVAL'].set_index('Subject').drop(columns=['SET', 'CENTER'])

    train_folds = train_set['CENTER']
    train_set = train_set.drop(columns=['CENTER'])

    X_train, y_train = train_set.drop(columns=[outcome.value]), train_set[outcome.value]
    X_test, y_test = test_set.drop(columns=[outcome.value]), test_set[outcome.value]
    X_ext, y_ext = ext_set.drop(columns=[outcome.value]), ext_set[outcome.value]

    with open('mlef_pipeline/submodel_features.json', 'r') as f:
        submodel_features = json.load(f)
        submodel_features = [f for f in submodel_features if f in X_train.columns]
    
    X_train = X_train.drop(columns=submodel_features, errors='ignore')
    X_test = X_test.drop(columns=submodel_features, errors='ignore')
    X_ext = X_ext.drop(columns=submodel_features, errors='ignore')
    
    X_train_imputed, imputer = dl.impute_df(X_train)
    X_ext_imputed, _ = dl.impute_df(X_ext, imputer=imputer) if not X_ext.empty else (X_ext, None)
    X_test_imputed, _ = dl.impute_df(X_test, imputer=imputer)
    
    X_train_scaled, scaler, to_standard_normalize, to_log_normalize = dl.normalize(X_train_imputed)
    X_test_scaled, _, _, _ = dl.normalize(X_test_imputed, scaler=scaler, 
                                           to_standard_normalize=to_standard_normalize,
                                           to_log_normalize=to_log_normalize)
    if not X_ext.empty:
        X_ext_scaled, _, _, _ = dl.normalize(X_ext_imputed, scaler=scaler,
                                              to_standard_normalize=to_standard_normalize,
                                              to_log_normalize=to_log_normalize)
    else:
        X_ext_scaled = X_ext
    
    cv = SafeGroupKFold(n_splits=len(train_folds.unique()))
    
    def get_cv_splits():
        return list(cv.split(X_train_scaled, y_train, groups=train_folds))
    
    model = ml.train_classification_model(
        X=X_train_scaled,
        y=y_train,
        model_name=model_type,
        cv=lambda: iter(get_cv_splits()),
        select_features=select_features
    )
    
    selected_features = model.feature_names_in_
    X_train_final = X_train_scaled[selected_features]
    X_test_final = X_test_scaled[selected_features]
    X_ext_final = X_ext_scaled[selected_features] if not X_ext_scaled.empty else X_ext_scaled

    cv = GroupKFold(n_splits=len(train_folds.unique()))
    def get_cv_splits():
        return list(cv.split(X_train_scaled, y_train, groups=train_folds))
    
    y_proba_cv, cv_metrics = compute_cv_predictions(
        model, X_train_final, y_train, get_cv_splits()
    )
    
    sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
    model.fit(X_train_final, y_train, sample_weight=sample_weight)
    
    y_pred_test = model.predict_proba(X_test_final)[:, 1]
    test_metrics = compute_metrics_with_ci(y_test.values, y_pred_test)
    
    y_pred_ext = model.predict_proba(X_ext_final)[:, 1]
    ext_metrics = compute_metrics_with_ci(y_ext.values, y_pred_ext)
    
    # Save everything
    joblib.dump(model, output_dir / f'model_{model_type.value}.pkl')
    
    train_set_with_outcome = X_train_scaled.copy()
    train_set_with_outcome['Subject'] = train_set.index
    train_set_with_outcome[outcome.value] = y_train
    test_set_with_outcome = X_test_scaled.copy()
    test_set_with_outcome['Subject'] = test_set.index
    test_set_with_outcome[outcome.value] = y_test
    ext_set_with_outcome = X_ext_scaled.copy()
    ext_set_with_outcome['Subject'] = ext_set.index
    if not ext_set.empty:
        ext_set_with_outcome[outcome.value] = y_ext
    
    save_datasets(output_dir, train_set_with_outcome, test_set_with_outcome,
                  ext_set_with_outcome, train_folds)
    
    # Save predictions
    cv_predictions = pd.DataFrame({
        'Subject': X_train_final.index,
        'y_proba': y_proba_cv,
        'y_pred': np.where(~y_proba_cv.isna(), (y_proba_cv >= 0.5).astype(int), np.nan),
        'y_true': y_train
    })
    cv_predictions.to_csv(output_dir / 'prediction_CV.csv', index=False)
    
    # Save TEST predictions
    test_predictions = pd.DataFrame({
        'Subject': X_test_final.index,
        'y_proba': y_pred_test,
        'y_pred': np.where(~pd.isna(y_pred_test), (y_pred_test >= 0.5).astype(int), np.nan),
        'y_true': y_test.values
    })
    test_predictions.to_csv(output_dir / 'prediction_TEST.csv', index=False)

    exval_predictions = pd.DataFrame({
        'Subject': X_ext_final.index,
        'y_proba': y_pred_ext,
        'y_pred': np.where(~pd.isna(y_pred_ext), (y_pred_ext >= 0.5).astype(int), np.nan),
        'y_true': y_ext.values
    })
    exval_predictions.to_csv(output_dir / 'prediction_EXVAL.csv', index=False)
    
    results = pd.DataFrame({
        'SET': ['CV', 'TEST', 'EXVAL'],
        'AUC': [
            f'{cv_metrics["auc"]:.2f} ± {cv_metrics["auc_std"]:.2f}',
            f'{test_metrics["auc"]:.2f} ± {test_metrics["auc_std"]:.2f}',
            f'{ext_metrics["auc"]:.2f} ± {ext_metrics["auc_std"]:.2f}' if not np.isnan(ext_metrics["auc"]) else 'N/A'
        ],
        'F1_MACRO': [
            f'{cv_metrics["f1_macro"]:.2f} ± {cv_metrics["f1_macro_std"]:.2f}',
            f'{test_metrics["f1_macro"]:.2f}',
            f'{ext_metrics["f1_macro"]:.2f}' if not np.isnan(ext_metrics["f1_macro"]) else 'N/A'
        ],
        'SENSITIVITY': [
            f'{cv_metrics["sensitivity"]:.2f} ± {cv_metrics["sensitivity_std"]:.2f}',
            f'{test_metrics["sensitivity"]:.2f}',
            f'{ext_metrics["sensitivity"]:.2f}' if not np.isnan(ext_metrics["sensitivity"]) else 'N/A'
        ],
        'SPECIFICITY': [
            f'{cv_metrics["specificity"]:.2f} ± {cv_metrics["specificity_std"]:.2f}',
            f'{test_metrics["specificity"]:.2f}',
            f'{ext_metrics["specificity"]:.2f}' if not np.isnan(ext_metrics["specificity"]) else 'N/A'
        ],
        'n': [len(y_train), len(y_test), len(y_ext) if not ext_set.empty else 0]
    })
    results.to_excel(output_dir / 'results.xlsx', index=False)
    
    print(f"\n✓ RWD-matched model completed!")
    print(f"  CV AUC: {cv_metrics['auc']:.3f} ± {cv_metrics['auc_std']:.3f}")
    print(f"  CV F1 Macro: {cv_metrics['f1_macro']:.3f} ± {cv_metrics['f1_macro_std']:.3f}")
    print(f"  CV Sensitivity: {cv_metrics['sensitivity']:.3f} ± {cv_metrics['sensitivity_std']:.3f}")
    print(f"  CV Specificity: {cv_metrics['specificity']:.3f} ± {cv_metrics['specificity_std']:.3f}")
    
    return {
        'modality': f'{modality_folder}/rwd-only',
        'cv_auc': cv_metrics['auc'],
        'cv_auc_std': cv_metrics['auc_std'],
        'cv_f1_macro': cv_metrics['f1_macro'],
        'cv_f1_macro_std': cv_metrics['f1_macro_std'],
        'cv_sensitivity': cv_metrics['sensitivity'],
        'cv_sensitivity_std': cv_metrics['sensitivity_std'],
        'cv_specificity': cv_metrics['specificity'],
        'cv_specificity_std': cv_metrics['specificity_std'],
        'test_auc': test_metrics['auc'],
        'test_auc_std': test_metrics['auc_std'],
        'test_f1_macro': test_metrics['f1_macro'],
        'test_sensitivity': test_metrics['sensitivity'],
        'test_specificity': test_metrics['specificity'],
        'ext_auc': ext_metrics['auc'],
        'ext_auc_std': ext_metrics['auc_std'],
        'ext_f1_macro': ext_metrics['f1_macro'],
        'ext_sensitivity': ext_metrics['sensitivity'],
        'ext_specificity': ext_metrics['specificity'],
        'n_features': len(selected_features),
        'n_train': len(y_train),
        'n_test': len(y_test),
        'n_ext': len(y_ext)
    }


def main():
    parser = argparse.ArgumentParser(description='Train MLEF models for different modality combinations')
    parser.add_argument('--outcome', type=str, default='OS_6',
                        choices=['OS_6', 'OS_24', 'DCR'],
                        help='Target outcome to predict (default: OS_6)')
    parser.add_argument('--subanalysis', type=str, default='C23',
                        choices=['C23', 'C2', 'IO_ONLY', 'IO_CHT', 'LOW_PDL1', 
                                'HIGH_PDL1', 'SQUAMOUS', 'ADENOCARCINOMA'],
                        help='Subgroup analysis to perform (default: C23)')
    parser.add_argument('--modalities', type=str, nargs='+', default=None,
                        help='Modalities to train (default: all combinations). '
                             'Options: RWD, RWD_DP, RWD_FMRAD, RWD_PYRAD, RWD_DP_FMRAD, RWD_DP_PYRAD')
    parser.add_argument('--model', type=str, default=None,
                        choices=['LR', 'RF'],
                        help='Model type to train (default: LR)')
    parser.add_argument('--no-feature-selection', action='store_true',
                        help='Disable feature selection')
    parser.add_argument('--output-dir', type=str, default='mlef_pipeline',
                        help='Base output directory (default: current directory)')
    
    args = parser.parse_args()
    
    # Convert string arguments to enums
    outcome = Outcome[args.outcome]
    subanalysis = Subanalysis[args.subanalysis]
    # Se il parametro model è stato passato esplicitamente, lo segno
    if args.model is not None:
        default_model_type = Model[args.model]
        default_model_type._explicit = True
    else:
        default_model_type = None
    base_path = Path(args.output_dir)
    select_features = not args.no_feature_selection
    
        # Load modality models configuration
    modality_models_config, rwd_only_models_config = load_modality_models_config(outcome, subanalysis)
    
    # Determine which modalities to train
    if args.modalities:
        # Parse user-specified modalities
        modalities_to_train = []
        for mod_str in args.modalities:
            mode_list = []
            for m in mod_str.split('_'):
                if m == 'RWD':
                    mode_list.append(Mode.RWD)
                elif m == 'DP':
                    mode_list.append(Mode.DP)
                elif m == 'FMRAD':
                    mode_list.append(Mode.FMRAD)
                elif m == 'PYRAD':
                    mode_list.append(Mode.PYRAD)
            if mode_list:
                modalities_to_train.append(mode_list)
    else:
        # Use default modalities
        modalities_to_train = DEFAULT_MODALITIES
    
    print("\n" + "="*80)
    print("MLEF MODEL TRAINING")
    print("="*80)
    print(f"Outcome: {outcome.value}")
    print(f"Subanalysis: {subanalysis.value}")
    # print(f"Default model type: {default_model_type.value}")
    print(f"Feature selection: {'LASSO' if select_features else 'disabled'}")
    print(f"Output directory: {base_path.resolve()}")
    print(f"\nModalities to train ({len(modalities_to_train)}):")
    for modes in modalities_to_train:
        modality_folder = get_modality_folder_name(modes)
        model_for_modality = get_model_for_modality(modality_folder, modality_models_config, default_model_type)
        if modality_folder != 'RWD' and len(modes) > 1:
            rwd_only_model = get_rwd_only_model_for_modality(modality_folder, rwd_only_models_config, 
                                                               model_for_modality)
            print(f"  - {modality_folder} (Model: {model_for_modality.value}, RWD-only: {rwd_only_model.value})")
        else:
            print(f"  - {modality_folder} (Model: {model_for_modality.value})")
    print("="*80 + "\n")
    
    # Store all results
    all_results = []
    
    # Train each modality combination
    for i, modes in enumerate(modalities_to_train, 1):
        modality_folder = get_modality_folder_name(modes)
        print(f"\n[{i}/{len(modalities_to_train)}] Processing {modality_folder}...")
        
        # Get the model type for this modality from config
        model_type = get_model_for_modality(modality_folder, modality_models_config, default_model_type)
        print(f"Using model: {model_type.value}")
        
        try:
            # Train main model
            result = train_and_evaluate_modality(
                modes=modes,
                outcome=outcome,
                subanalysis=subanalysis,
                model_type=model_type,
                base_path=base_path,
                select_features=select_features
            )
            all_results.append(result)
            
            # Train RWD-matched model if not RWD-only
            if len(modes) > 1:
                # Get the model type for RWD-only analysis
                rwd_only_model_type = get_rwd_only_model_for_modality(
                    modality_folder, rwd_only_models_config, model_type
                )
                print(f"Training RWD-only with model: {rwd_only_model_type.value}")
                
                rwd_result = train_rwd_matched_model(
                    modes=modes,
                    outcome=outcome,
                    subanalysis=subanalysis,
                    model_type=rwd_only_model_type,
                    base_path=base_path,
                    select_features=select_features
                )
                if rwd_result:
                    all_results.append(rwd_result)
            
        except Exception as e:
            print(f"\n❌ Error training {modality_folder}: {str(e)}")
            import traceback
            traceback.print_exc()
            break
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Successfully trained {len(all_results)} model(s)\n")
    
    summary_df = pd.DataFrame(all_results)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_path = base_path / 'results' / outcome.value / subanalysis.value / 'training_summary.xlsx'
        summary_df.to_excel(summary_path, index=False)
        print(f"\n✓ Summary saved to {summary_path}")
    
    print("\n" + "="*80)
    print("ALL DONE! 🎉")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
