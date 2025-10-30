"""
MLEF Model Training Script

This script trains MLEF (Multi-Level Early Fusion) models for different modality combinations.
It saves models, datasets, predictions, and results in a structured folder format.

Usage:
    python train_mlef.py --outcome OS_6 --subanalysis C23
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


def create_output_dirs(base_path: Path, subanalysis: str, modality_folder: str, outcome: str) -> Path:
    """Create output directory structure and return the modality path."""
    output_dir = base_path / 'MLEF' / outcome / subanalysis / modality_folder
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
    
    train_with_folds.to_excel(output_dir / 'train_set.xlsx', index=True)
    test_set.to_excel(output_dir / 'test_set.xlsx', index=True)
    
    if not ext_set.empty:
        ext_set.to_excel(output_dir / 'exval_set.xlsx', index=True)
    
    print(f"  ✓ Saved datasets to {output_dir}")


def compute_cv_predictions(model, X: pd.DataFrame, y: pd.Series, cv_splits, 
                           train_folds: pd.Series) -> Tuple[np.ndarray, float, float]:
    """
    Compute cross-validation predictions using LOCO-CV.
    
    Returns:
        y_pred_cv: Cross-validated predictions
        cv_auc_mean: Mean CV AUC
        cv_auc_std: Std of CV AUC
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.base import clone
    
    stats = Statistics()
    
    y_pred_cv = np.zeros(len(y))
    cv_aucs = []
    
    for train_idx, val_idx in cv_splits:
        # Clone model for each fold
        fold_model = clone(model)
        
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Compute sample weights for this fold
        sample_weight = compute_sample_weight(class_weight='balanced', y=y_train_fold)
        
        # Fit model on this fold
        fold_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weight)
        
        # Predict on validation fold
        y_pred_cv[val_idx] = fold_model.predict_proba(X_val_fold)[:, 1]
        
        # Compute fold AUC
        fold_auc = roc_auc_score(y_val_fold, y_pred_cv[val_idx])
        cv_aucs.append(fold_auc)
    
    # Compute overall CV AUC and confidence interval
    cv_auc, ci = stats.auc_roc_ci(y, y_pred_cv, alpha=0.95)
    cv_auc_std = (ci[1] - ci[0]) / 2
    
    return y_pred_cv, cv_auc, cv_auc_std


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
    with open('split.json', 'r') as f:
        split = json.load(f)
    
    train_set = dataset[dataset['Subject'].isin(split['TRAIN_SET'])].set_index('Subject')
    test_set = dataset[dataset['Subject'].isin(split['TEST_SET'])].set_index('Subject')
    ext_set = dataset[dataset['Subject'].str.startswith('UOC')].set_index('Subject')
    
    # 3. Separate features and target
    print("3. Preparing features and target...")
    X_train, y_train = train_set.drop(columns=[outcome.value]), train_set[outcome.value]
    X_test, y_test = test_set.drop(columns=[outcome.value]), test_set[outcome.value]
    X_ext, y_ext = ext_set.drop(columns=[outcome.value]), ext_set[outcome.value]
    
    # Convert outcome to binary if needed
    # y_train = dl.get_outcome(y_train_raw, outcome)
    # y_test = dl.get_outcome(y_test_raw, outcome)
    # y_ext = dl.get_outcome(y_ext_raw, outcome) if not ext_set.empty else pd.Series()
    
    with open('submodel_features.json', 'r') as f:
        submodel_features = json.load(f)
        submodel_features = [f for f in submodel_features if f in X_train.columns]
    
    X_train = X_train.drop(columns=submodel_features, errors='ignore')
    X_test = X_test.drop(columns=submodel_features, errors='ignore')
    X_ext = X_ext.drop(columns=submodel_features, errors='ignore')
    
    # 4. Imputation
    print("4. Imputing missing values...")
    X_train_imputed, imputer = dl.impute_df(X_train)
    X_test_imputed, _ = dl.impute_df(X_test, imputer=imputer)
    X_ext_imputed, _ = dl.impute_df(X_ext, imputer=imputer) if not X_ext.empty else (X_ext, None)
    
    # 5. Normalization
    print("5. Normalizing features...")
    X_train_scaled, scaler, to_standard_normalize, to_log_normalize = dl.normalize(X_train_imputed)
    X_test_scaled, _, _, _ = dl.normalize(
        X_test_imputed, scaler=scaler, 
        to_standard_normalize=to_standard_normalize, 
        to_log_normalize=to_log_normalize
    )
    if not X_ext.empty:
        X_ext_scaled, _, _, _ = dl.normalize(
            X_ext_imputed, scaler=scaler,
            to_standard_normalize=to_standard_normalize,
            to_log_normalize=to_log_normalize
        )
    else:
        X_ext_scaled = X_ext
    
    # 6. Setup cross-validation
    print("6. Setting up cross-validation...")
    train_folds = dl.get_loco_folds(pd.Series(train_set.index))
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
    y_pred_cv, cv_auc, cv_auc_std = compute_cv_predictions(
        model, X_train_final, y_train, get_cv_splits(), train_folds
    )
    
    # 9. Make predictions on all sets
    print("9. Making predictions on test and external sets...")
    # Refit on full training set for final predictions
    # sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
    # model.fit(X_train_final, y_train, sample_weight=sample_weight)
    
    y_pred_test = model.predict_proba(X_test_final)[:, 1]
    test_auc, test_ci = stats.auc_roc_ci(y_test, y_pred_test, alpha=0.95)
    test_auc_std = (test_ci[1] - test_ci[0]) / 2
    
    if not X_ext_final.empty:
        y_pred_ext = model.predict_proba(X_ext_final)[:, 1]
        ext_auc, ext_ci = stats.auc_roc_ci(y_ext, y_pred_ext, alpha=0.95)
        ext_auc_std = (ext_ci[1] - ext_ci[0]) / 2
    else:
        y_pred_ext = np.array([])
        ext_auc, ext_auc_std = np.nan, np.nan
    
    # 10. Save model
    print("10. Saving model...")
    model_path = output_dir / f'model_{model_type.value}.pkl'
    joblib.dump(model, model_path)
    print(f"  ✓ Model saved to {model_path}")

    # 11. Save datasets (with outcome)
    print("11. Saving datasets...")
    # train_set_with_outcome = train_set.copy()
    # train_set_with_outcome[outcome.value] = y_train
    # test_set_with_outcome = test_set.copy()
    # test_set_with_outcome[outcome.value] = y_test
    # ext_set_with_outcome = ext_set.copy()
    # if not ext_set.empty:
    #     ext_set_with_outcome[outcome.value] = y_ext
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
    
    # 12. Save CV predictions
    print("12. Saving CV predictions...")
    cv_predictions = pd.DataFrame({
        'Subject': X_train_final.index,
        'y_pred': y_pred_cv,
        'y_true': y_train
    })
    cv_predictions.to_excel(output_dir / 'prediction_CV.xlsx', index=False)

    # 13. Save results
    print("13. Saving results...")
    results = pd.DataFrame({
        'SET': ['CV', 'TEST', 'EXVAL'],
        'AUC': [
            f'{cv_auc:.2f} ± {cv_auc_std:.2f}',
            f'{test_auc:.2f} ± {test_auc_std:.2f}',
            f'{ext_auc:.2f} ± {ext_auc_std:.2f}' if not np.isnan(ext_auc) else 'N/A'
        ],
        'n': [len(y_train), len(y_test), len(y_ext) if not ext_set.empty else 0]
    })
    results.to_excel(output_dir / 'results.xlsx', index=False)
    
    print("\n✓ Training completed successfully!")
    print(f"  CV AUC: {cv_auc:.3f} ± {cv_auc_std:.3f}")
    print(f"  Test AUC: {test_auc:.3f} ± {test_auc_std:.3f}")
    if not np.isnan(ext_auc):
        print(f"  External AUC: {ext_auc:.3f} ± {ext_auc_std:.3f}")
    
    return {
        'modality': modality_folder,
        'cv_auc': cv_auc,
        'cv_auc_std': cv_auc_std,
        'test_auc': test_auc,
        'test_auc_std': test_auc_std,
        'ext_auc': ext_auc,
        'ext_auc_std': ext_auc_std,
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
        outcome='OS MONTHS',
        subanalysis=subanalysis
    )
    rwd_dataset = rwd_dataset[rwd_dataset['Subject'].isin(matched_subjects)]
    
    print(f"  Using {len(matched_subjects)} matched subjects")
    
    # Create output directory
    output_dir = base_path / 'MLEF' / outcome.value / subanalysis.value / modality_folder / 'rwd-only'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Follow same pipeline as main training
    with open('split.json', 'r') as f:
        split = json.load(f)
    
    train_set = rwd_dataset[rwd_dataset['Subject'].isin(split['TRAIN_SET'])].set_index('Subject')
    test_set = rwd_dataset[rwd_dataset['Subject'].isin(split['TEST_SET'])].set_index('Subject')
    ext_set = rwd_dataset[rwd_dataset['Subject'].str.startswith('UOC')].set_index('Subject')

    X_train, y_train_raw = train_set.drop(columns=[outcome.value]), train_set[outcome.value]
    X_test, y_test_raw = test_set.drop(columns=[outcome.value]), test_set[outcome.value]
    X_ext, y_ext_raw = ext_set.drop(columns=[outcome.value]), ext_set[outcome.value]

    y_train = dl.get_outcome(y_train_raw, outcome)
    y_test = dl.get_outcome(y_test_raw, outcome)
    y_ext = dl.get_outcome(y_ext_raw, outcome) if not ext_set.empty else pd.Series()
    
    with open('submodel_features.json', 'r') as f:
        submodel_features = json.load(f)
        submodel_features = [f for f in submodel_features if f in X_train.columns]
    
    X_train = X_train.drop(columns=submodel_features, errors='ignore')
    X_test = X_test.drop(columns=submodel_features, errors='ignore')
    X_ext = X_ext.drop(columns=submodel_features, errors='ignore')
    
    X_train_imputed, imputer = dl.impute_df(X_train)
    X_test_imputed, _ = dl.impute_df(X_test, imputer=imputer)
    X_ext_imputed, _ = dl.impute_df(X_ext, imputer=imputer) if not X_ext.empty else (X_ext, None)
    
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
    
    train_folds = dl.get_loco_folds(pd.Series(train_set.index))
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
    
    y_pred_cv, cv_auc, cv_auc_std = compute_cv_predictions(
        model, X_train_final, y_train, get_cv_splits(), train_folds
    )
    
    sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
    model.fit(X_train_final, y_train, sample_weight=sample_weight)
    
    y_pred_test = model.predict_proba(X_test_final)[:, 1]
    test_auc, test_ci = stats.auc_roc_ci(y_test, y_pred_test, alpha=0.95)
    test_auc_std = (test_ci[1] - test_ci[0]) / 2
    
    if not X_ext_final.empty:
        y_pred_ext = model.predict_proba(X_ext_final)[:, 1]
        ext_auc, ext_ci = stats.auc_roc_ci(y_ext, y_pred_ext, alpha=0.95)
        ext_auc_std = (ext_ci[1] - ext_ci[0]) / 2
    else:
        y_pred_ext = np.array([])
        ext_auc, ext_auc_std = np.nan, np.nan
    
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
    
    cv_predictions = pd.DataFrame({
        'Subject': X_train_final.index,
        'y_pred': y_pred_cv,
        'y_true': y_train
    })
    cv_predictions.to_excel(output_dir / 'prediction_CV.xlsx', index=False)
    
    results = pd.DataFrame({
        'SET': ['CV', 'TEST', 'EXVAL'],
        'AUC': [
            f'{cv_auc:.2f} ± {cv_auc_std:.2f}',
            f'{test_auc:.2f} ± {test_auc_std:.2f}',
            f'{ext_auc:.2f} ± {ext_auc_std:.2f}' if not np.isnan(ext_auc) else 'N/A'
        ],
        'n': [len(y_train), len(y_test), len(y_ext) if not ext_set.empty else 0]
    })
    results.to_excel(output_dir / 'results.xlsx', index=False)
    
    print(f"\n✓ RWD-matched model completed!")
    print(f"  CV AUC: {cv_auc:.3f} ± {cv_auc_std:.3f}")
    
    return {
        'modality': f'{modality_folder}/rwd-only',
        'cv_auc': cv_auc,
        'cv_auc_std': cv_auc_std,
        'n_train': len(y_train)
    }


def main():
    parser = argparse.ArgumentParser(description='Train MLEF models for different modality combinations')
    parser.add_argument('--outcome', type=str, default='OS_6',
                        choices=['OS_6', 'OS_24'],
                        help='Target outcome to predict (default: OS_6)')
    parser.add_argument('--subanalysis', type=str, default='C23',
                        choices=['C23', 'C2', 'IO_ONLY', 'IO_CHT', 'LOW_PDL1', 
                                'HIGH_PDL1', 'SQUAMOUS', 'ADENOCARCINOMA'],
                        help='Subgroup analysis to perform (default: C23)')
    parser.add_argument('--modalities', type=str, nargs='+', default=None,
                        help='Modalities to train (default: all combinations). '
                             'Options: RWD, RWD_DP, RWD_FMRAD, RWD_PYRAD, RWD_DP_FMRAD, RWD_DP_PYRAD')
    parser.add_argument('--model', type=str, default='LR',
                        choices=['LR', 'RF', 'XGB'],
                        help='Model type to train (default: LR)')
    parser.add_argument('--no-feature-selection', action='store_true',
                        help='Disable feature selection')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Base output directory (default: current directory)')
    
    args = parser.parse_args()
    
    # Convert string arguments to enums
    outcome = Outcome[args.outcome]
    subanalysis = Subanalysis[args.subanalysis]
    model_type = Model[args.model]
    base_path = Path(args.output_dir)
    select_features = not args.no_feature_selection
    
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
    print(f"Model type: {model_type.value}")
    print(f"Feature selection: {'LASSO' if select_features else 'disabled'}")
    print(f"Output directory: {base_path.resolve()}")
    print(f"\nModalities to train ({len(modalities_to_train)}):")
    for modes in modalities_to_train:
        print(f"  - {get_modality_folder_name(modes)}")
    print("="*80 + "\n")
    
    # Store all results
    all_results = []
    
    # Train each modality combination
    for i, modes in enumerate(modalities_to_train, 1):
        print(f"\n[{i}/{len(modalities_to_train)}] Processing {get_modality_folder_name(modes)}...")
        
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
            if get_modality_folder_name(modes) != 'RWD':
                rwd_result = train_rwd_matched_model(
                    modes=modes,
                    outcome=outcome,
                    subanalysis=subanalysis,
                    model_type=model_type,
                    base_path=base_path,
                    select_features=select_features
                )
                if rwd_result:
                    all_results.append(rwd_result)
            
        except Exception as e:
            print(f"\n❌ Error training {get_modality_folder_name(modes)}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Successfully trained {len(all_results)} model(s)\n")
    
    summary_df = pd.DataFrame(all_results)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_path = base_path / 'MLEF' / subanalysis.value / 'training_summary.xlsx'
        summary_df.to_excel(summary_path, index=False)
        print(f"\n✓ Summary saved to {summary_path}")
    
    print("\n" + "="*80)
    print("ALL DONE! 🎉")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
