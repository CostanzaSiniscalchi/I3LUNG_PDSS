"""
train_mlef_surv.py

MLEF Survival Analysis Training Script

This script trains CoxPH survival models for different modality combinations.
It saves models, datasets, predictions, and results in a structured folder format.

Usage:
    python train_mlef_surv.py --subanalysis C23
"""

from __future__ import annotations

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
from lifelines import CoxPHFitter
from sklearn.model_selection import GroupKFold

from data_curation import DataLoader, SafeGroupKFold
from i3l_statistics import Statistics
from i3l_ml import ML
from enums import Mode, Subanalysis

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='lifelines')


warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='lifelines')


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


def create_output_dirs(base_path: Path, subanalysis: str, modality_folder: str) -> Path:
    """Create output directory structure and return the modality path."""
    output_dir = base_path / 'results' / 'OS' / subanalysis / modality_folder
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
    
    # Ensure index name is 'Subject'
    train_with_folds.index.name = 'Subject'
    test_set.index.name = 'Subject'
    if not ext_set.empty:
        ext_set.index.name = 'Subject'
    
    train_with_folds.to_excel(output_dir / 'train_set.xlsx', index=True)
    test_set.to_excel(output_dir / 'test_set.xlsx', index=True)
    
    if not ext_set.empty:
        ext_set.to_excel(output_dir / 'exval_set.xlsx', index=True)
    
    print(f"  ✓ Saved datasets to {output_dir}")


def get_scores(values: dict) -> str:
    """Format c-index score with confidence interval."""
    return f"{values['c_index']:.2f} ± {(values['ci'][1] - values['ci'][0]) / 2 :.2f}"


def coxph_cv(train_set: pd.DataFrame, cv_getter: callable) -> np.ndarray:
    """Run cross-validated CoxPH and return per-sample predicted risk (NaN where not predicted).

    Parameters
    ----------
    train_set : pd.DataFrame
        DataFrame containing features + 'TIME' and 'EVENT' columns.
    cv_getter : callable
        Callable that returns an iterable of (train_index, test_index) splits (indices into train_set).
    
    Returns
    -------
    pred_risk : np.ndarray
        Cross-validated risk predictions for each sample.
    """
    pred_risk = np.empty(train_set.shape[0])
    pred_risk[:] = np.nan

    for index_train, index_test in cv_getter():
        # skip tiny folds
        if len(index_test) < 3:
            continue

        X_train_fold = train_set.iloc[index_train]
        X_val_fold = train_set.iloc[index_test]

        cph = CoxPHFitter(penalizer=0.5)
        cph.fit(X_train_fold, duration_col='TIME', event_col='EVENT')

        # lifelines returns a pandas Series; assign into numpy array
        pred_risk[index_test] = cph.predict_partial_hazard(X_val_fold)

    return pred_risk


def train_and_evaluate_modality(
    modes: List[Mode],
    subanalysis: Subanalysis,
    base_path: Path = Path('.'),
    select_features: bool = True
) -> dict:
    """
    Train and evaluate a CoxPH survival model for a specific modality combination.
    
    Args:
        modes: List of data modalities to include
        subanalysis: Subgroup analysis to perform
        base_path: Base directory for saving results
        select_features: Whether to perform feature selection
    
    Returns:
        Dictionary with results and metrics
    """
    print(f"\n{'='*80}")
    print(f"Training {get_modality_folder_name(modes)} - OS - {subanalysis.value}")
    print(f"{'='*80}")
    
    # Initialize helpers
    dl = DataLoader()
    ml = ML()
    stats = Statistics()
    
    # Create output directory
    modality_folder = get_modality_folder_name(modes)
    output_dir = create_output_dirs(base_path, subanalysis.value, modality_folder)
    
    # 1. Load and prepare data
    print("1. Loading data...")
    dataset = dl.create_dataset(
        modes=modes,
        outcome='OS MONTHS',
        subanalysis=subanalysis,
        is_survival=True
    )
    
    # 2. Split data
    print("2. Splitting data...")
    with open('mlef_pipeline/split.json', 'r') as f:
        split = json.load(f)
    
    train_set = dataset[dataset['Subject'].isin(split['TRAIN_SET'])].set_index('Subject')
    test_set = dataset[dataset['Subject'].isin(split['TEST_SET'])].set_index('Subject')
    ext_set = dataset[dataset['Subject'].str.startswith('UOC')].set_index('Subject')
    
    # 3. Remove test samples with TIME > max train TIME
    print("3. Adjusting test set for survival analysis...")
    removed = 0
    while True:
        max_train_time = train_set['OS MONTHS'].max()
        max_test_time = test_set['OS MONTHS'].max()
        if max_test_time <= max_train_time:
            break
        removed += 1
        max_test_index = test_set['OS MONTHS'].idxmax()
        test_set = test_set.drop(max_test_index)
    
    if removed > 0:
        print(f"  ✓ Removed {removed} test sample(s) with TIME > max train TIME")
    
    # 4. Separate features and target
    print("4. Preparing features and target...")
    X_train = train_set.drop(columns=['OS MONTHS', 'DEATH EVENT'])
    y_train = train_set[['OS MONTHS', 'DEATH EVENT']]
    X_test = test_set.drop(columns=['OS MONTHS', 'DEATH EVENT'])
    y_test = test_set[['OS MONTHS', 'DEATH EVENT']]
    X_ext = ext_set.drop(columns=['OS MONTHS', 'DEATH EVENT'])
    y_ext = ext_set[['OS MONTHS', 'DEATH EVENT']]
    
    # Rename columns to TIME/EVENT for lifelines
    y_train = y_train.rename(columns={'OS MONTHS': 'TIME', 'DEATH EVENT': 'EVENT'})
    y_test = y_test.rename(columns={'OS MONTHS': 'TIME', 'DEATH EVENT': 'EVENT'})
    y_ext = y_ext.rename(columns={'OS MONTHS': 'TIME', 'DEATH EVENT': 'EVENT'})
    
    # Drop submodel features if present
    with open('mlef_pipeline/submodel_features.json', 'r') as f:
        submodel_features = json.load(f)
        submodel_features = [f for f in submodel_features if f in X_train.columns]
    
    if len(submodel_features) > 0:
        X_train = X_train.drop(columns=submodel_features)
        X_test = X_test.drop(columns=submodel_features)
        X_ext = X_ext.drop(columns=submodel_features)
    
    # 5. Imputation
    print("5. Imputing missing values...")
    X_train_imputed, imputer = dl.impute_df(X_train)
    X_test_imputed, _ = dl.impute_df(X_test, imputer=imputer)
    X_ext_imputed, _ = dl.impute_df(X_ext, imputer=imputer) if not X_ext.empty else (X_ext, None)
    
    # 6. Normalization
    print("6. Normalizing features...")
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
    
    # 7. Setup cross-validation
    print("7. Setting up cross-validation...")
    train_folds = dl.get_loco_folds(pd.Series(train_set.index))
    cv = GroupKFold(n_splits=len(train_folds.unique()))
    
    def cv_getter():
        return cv.split(X_train_scaled, y_train, groups=train_folds)
    
    # 8. Ensure EVENT dtype
    y_train['EVENT'] = y_train['EVENT'].astype(bool)
    y_train = y_train[['EVENT', 'TIME']]
    
    # 9. Feature selection (coxnet)
    fm_rad_features = [col for col in X_train_scaled.columns if col.startswith('pred_') or col.startswith('log_pred_')]
    if len(fm_rad_features) > 0:
        selected_fm_rad_features = ml.coxnet_selection(
            X_train=X_train_scaled[fm_rad_features],
            y_train=y_train.to_records(index=False),
            cv=cv_getter,
            folds=train_folds,
            target_features=100,
            uncertainty=10,
        )
        X_train_scaled = X_train_scaled.drop(columns=[col for col in fm_rad_features if col not in selected_fm_rad_features])
        X_test_scaled = X_test_scaled.drop(columns=[col for col in fm_rad_features if col not in selected_fm_rad_features])
        X_ext_scaled = X_ext_scaled.drop(columns=[col for col in fm_rad_features if col not in selected_fm_rad_features]) if not X_ext_scaled.empty else X_ext_scaled
        print(f"  ✓ Selected {len(selected_fm_rad_features)} radiomics features")
    
    if select_features:
        print("8. Performing feature selection...")
        features = ml.coxnet_selection(
            X_train=X_train_scaled,
            y_train=y_train.to_records(index=False),
            cv=cv_getter,
            folds=train_folds,
            target_features=15,
            uncertainty=5,
        )
        print(f"  ✓ Selected {len(features)} features")
    else:
        print("8. Skipping feature selection (using all features)...")
        features = X_train_scaled.columns.tolist()
    
    # Select features
    X_train_sel = X_train_scaled[features]
    X_test_sel = X_test_scaled[features]
    X_ext_sel = X_ext_scaled[features] if not X_ext_scaled.empty else X_ext_scaled
    
    # 10. Build combined tables for lifelines
    print("9. Preparing datasets for CoxPH...")
    train_tbl = pd.concat([X_train_sel, pd.DataFrame(y_train, index=X_train_sel.index)], axis=1)
    test_tbl = pd.concat([X_test_sel, pd.DataFrame(y_test, index=X_test_sel.index)], axis=1)
    ext_tbl = pd.concat([X_ext_sel, pd.DataFrame(y_ext, index=X_ext_sel.index)], axis=1) if not X_ext_sel.empty else pd.DataFrame()
    
    # 11. Fit CoxPH model
    print("10. Training CoxPH model...")
    cph = CoxPHFitter(penalizer=0.5)
    cph.fit(train_tbl, duration_col='TIME', event_col='EVENT')
    
    # 12. Compute predictions and c-index
    print("11. Computing c-index scores...")
    
    # Train set
    risk_scores_train = cph.predict_log_partial_hazard(train_tbl).values
    cindex_train = stats.compute_c_index_and_ci(
        time=train_tbl['TIME'],
        event=train_tbl['EVENT'],
        risk_score=risk_scores_train,
    )
    
    # Test set
    risk_scores_test = cph.predict_log_partial_hazard(test_tbl).values
    cindex_test = stats.compute_c_index_and_ci(
        time=test_tbl['TIME'],
        event=test_tbl['EVENT'],
        risk_score=risk_scores_test,
    )
    
    # External set
    if not ext_tbl.empty:
        risk_scores_ext = cph.predict_log_partial_hazard(ext_tbl).values
        cindex_ext = stats.compute_c_index_and_ci(
            time=ext_tbl['TIME'],
            event=ext_tbl['EVENT'],
            risk_score=risk_scores_ext,
        )
    else:
        risk_scores_ext = np.array([])
        cindex_ext = {'c_index': np.nan, 'ci': (np.nan, np.nan)}
    
    # Cross-validation
    print("12. Computing CV predictions...")
    cv_risks = coxph_cv(
        train_set=train_tbl,
        cv_getter=cv_getter,
    )
    cindex_cv = stats.compute_c_index_and_ci(
        time=train_tbl['TIME'],
        event=train_tbl['EVENT'],
        risk_score=cv_risks,
    )
    
    # 13. Save model
    print("13. Saving model...")
    model_path = output_dir / 'model_COX.pkl'
    joblib.dump(cph, model_path)
    print(f"  ✓ Model saved to {model_path}")
    
    # 14. Save datasets (with TIME and EVENT)
    print("14. Saving datasets...")
    train_set_with_outcome = X_train_scaled.copy()
    train_set_with_outcome.index = train_set.index
    train_set_with_outcome['TIME'] = y_train['TIME'].values
    train_set_with_outcome['EVENT'] = y_train['EVENT'].values
    
    test_set_with_outcome = X_test_scaled.copy()
    test_set_with_outcome.index = test_set.index
    test_set_with_outcome['TIME'] = y_test['TIME'].values
    test_set_with_outcome['EVENT'] = y_test['EVENT'].values
    
    ext_set_with_outcome = X_ext_scaled.copy()
    if not ext_set.empty:
        ext_set_with_outcome.index = ext_set.index
        ext_set_with_outcome['TIME'] = y_ext['TIME'].values
        ext_set_with_outcome['EVENT'] = y_ext['EVENT'].values
    
    save_datasets(output_dir, train_set_with_outcome, test_set_with_outcome, 
                  ext_set_with_outcome, train_folds)
    
    # 15. Save predictions
    print("15. Saving predictions...")
    cv_predictions = pd.DataFrame({
        'Subject': X_train_sel.index,
        'risk_score': cv_risks,
        'TIME': y_train['TIME'].values,
        'EVENT': y_train['EVENT'].values
    })
    cv_predictions.to_excel(output_dir / 'prediction_CV.xlsx', index=False)
    
    test_predictions = pd.DataFrame({
        'Subject': X_test_sel.index,
        'risk_score': risk_scores_test,
        'TIME': y_test['TIME'].values,
        'EVENT': y_test['EVENT'].astype(bool).values
    })
    test_predictions.to_excel(output_dir / 'prediction_TEST.xlsx', index=False)
    
    if not ext_tbl.empty:
        ext_predictions = pd.DataFrame({
            'Subject': X_ext_sel.index,
            'risk_score': risk_scores_ext,
            'TIME': y_ext['TIME'].values,
            'EVENT': y_ext['EVENT'].astype(bool).values
        })
        ext_predictions.to_excel(output_dir / 'prediction_EXVAL.xlsx', index=False)
    
    # 16. Save results
    print("16. Saving results...")
    cv_auc_std = (cindex_cv['ci'][1] - cindex_cv['ci'][0]) / 2
    test_auc_std = (cindex_test['ci'][1] - cindex_test['ci'][0]) / 2
    ext_auc_std = (cindex_ext['ci'][1] - cindex_ext['ci'][0]) / 2 if not np.isnan(cindex_ext['c_index']) else np.nan
    
    results = pd.DataFrame({
        'SET': ['CV', 'TEST', 'EXVAL'],
        'C-INDEX': [
            f"{cindex_cv['c_index']:.2f} ± {cv_auc_std:.2f}",
            f"{cindex_test['c_index']:.2f} ± {test_auc_std:.2f}",
            f"{cindex_ext['c_index']:.2f} ± {ext_auc_std:.2f}" if not np.isnan(cindex_ext['c_index']) else 'N/A'
        ],
        'n': [len(y_train), len(y_test), len(y_ext) if not ext_set.empty else 0]
    })
    results.to_excel(output_dir / 'results.xlsx', index=False)
    
    print("\n✓ Training completed successfully!")
    print(f"  CV C-Index: {cindex_cv['c_index']:.3f} ± {cv_auc_std:.3f}")
    print(f"  Test C-Index: {cindex_test['c_index']:.3f} ± {test_auc_std:.3f}")
    if not np.isnan(cindex_ext['c_index']):
        print(f"  External C-Index: {cindex_ext['c_index']:.3f} ± {ext_auc_std:.3f}")
    
    return {
        'modality': modality_folder,
        'cv_cindex': cindex_cv['c_index'],
        'cv_cindex_std': cv_auc_std,
        'test_cindex': cindex_test['c_index'],
        'test_cindex_std': test_auc_std,
        'ext_cindex': cindex_ext['c_index'],
        'ext_cindex_std': ext_auc_std,
        'n_features': len(features),
        'n_train': len(y_train),
        'n_test': len(y_test),
        'n_ext': len(y_ext) if not ext_set.empty else 0
    }


def train_rwd_matched_model(
    modes: List[Mode],
    subanalysis: Subanalysis,
    base_path: Path = Path('.'),
    select_features: bool = True
):
    """
    Train RWD-matched CoxPH model (using only subjects that have all modalities).
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
        outcome='OS MONTHS',
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
    output_dir = base_path / 'results' / 'OS' / subanalysis.value / modality_folder / 'rwd-only'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Follow same pipeline as main training
    with open('mlef_pipeline/split.json', 'r') as f:
        split = json.load(f)
    
    train_set = rwd_dataset[rwd_dataset['Subject'].isin(split['TRAIN_SET'])].set_index('Subject')
    test_set = rwd_dataset[rwd_dataset['Subject'].isin(split['TEST_SET'])].set_index('Subject')
    ext_set = rwd_dataset[rwd_dataset['Subject'].str.startswith('UOC')].set_index('Subject')
    
    # Remove test samples with TIME > max train TIME
    removed = 0
    while True:
        max_train_time = train_set['OS MONTHS'].max()
        max_test_time = test_set['OS MONTHS'].max()
        if max_test_time <= max_train_time:
            break
        removed += 1
        max_test_index = test_set['OS MONTHS'].idxmax()
        test_set = test_set.drop(max_test_index)
    
    X_train = train_set.drop(columns=['OS MONTHS', 'DEATH EVENT'])
    y_train = train_set[['OS MONTHS', 'DEATH EVENT']]
    X_test = test_set.drop(columns=['OS MONTHS', 'DEATH EVENT'])
    y_test = test_set[['OS MONTHS', 'DEATH EVENT']]
    X_ext = ext_set.drop(columns=['OS MONTHS', 'DEATH EVENT'])
    y_ext = ext_set[['OS MONTHS', 'DEATH EVENT']]
    
    y_train = y_train.rename(columns={'OS MONTHS': 'TIME', 'DEATH EVENT': 'EVENT'})
    y_test = y_test.rename(columns={'OS MONTHS': 'TIME', 'DEATH EVENT': 'EVENT'})
    y_ext = y_ext.rename(columns={'OS MONTHS': 'TIME', 'DEATH EVENT': 'EVENT'})
    
    with open('mlef_pipeline/submodel_features.json', 'r') as f:
        submodel_features = json.load(f)
        submodel_features = [f for f in submodel_features if f in X_train.columns]
    
    if len(submodel_features) > 0:
        X_train = X_train.drop(columns=submodel_features)
        X_test = X_test.drop(columns=submodel_features)
        X_ext = X_ext.drop(columns=submodel_features)
    
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
    cv = GroupKFold(n_splits=len(train_folds.unique()))
    
    def cv_getter():
        return cv.split(X_train_scaled, y_train, groups=train_folds)
    
    y_train['EVENT'] = y_train['EVENT'].astype(bool)
    y_train = y_train[['EVENT', 'TIME']]
    
    # Feature selection
    if select_features:
        features = ml.coxnet_selection(
            X_train=X_train_scaled,
            y_train=y_train.to_records(index=False),
            cv=cv_getter,
            folds=train_folds,
            target_features=15,
            uncertainty=5,
        )
    else:
        features = X_train_scaled.columns.tolist()
    
    X_train_sel = X_train_scaled[features]
    X_test_sel = X_test_scaled[features]
    X_ext_sel = X_ext_scaled[features] if not X_ext_scaled.empty else X_ext_scaled
    
    train_tbl = pd.concat([X_train_sel, pd.DataFrame(y_train, index=X_train_sel.index)], axis=1)
    test_tbl = pd.concat([X_test_sel, pd.DataFrame(y_test, index=X_test_sel.index)], axis=1)
    ext_tbl = pd.concat([X_ext_sel, pd.DataFrame(y_ext, index=X_ext_sel.index)], axis=1) if not X_ext_sel.empty else pd.DataFrame()
    
    cph = CoxPHFitter(penalizer=0.5)
    cph.fit(train_tbl, duration_col='TIME', event_col='EVENT')
    
    risk_scores_train = cph.predict_log_partial_hazard(train_tbl).values
    cindex_train = stats.compute_c_index_and_ci(
        time=train_tbl['TIME'],
        event=train_tbl['EVENT'],
        risk_score=risk_scores_train,
    )
    
    risk_scores_test = cph.predict_log_partial_hazard(test_tbl).values
    cindex_test = stats.compute_c_index_and_ci(
        time=test_tbl['TIME'],
        event=test_tbl['EVENT'],
        risk_score=risk_scores_test,
    )
    
    if not ext_tbl.empty:
        risk_scores_ext = cph.predict_log_partial_hazard(ext_tbl).values
        cindex_ext = stats.compute_c_index_and_ci(
            time=ext_tbl['TIME'],
            event=ext_tbl['EVENT'],
            risk_score=risk_scores_ext,
        )
    else:
        risk_scores_ext = np.array([])
        cindex_ext = {'c_index': np.nan, 'ci': (np.nan, np.nan)}
    
    cv_risks = coxph_cv(train_set=train_tbl, cv_getter=cv_getter)
    cindex_cv = stats.compute_c_index_and_ci(
        time=train_tbl['TIME'],
        event=train_tbl['EVENT'],
        risk_score=cv_risks,
    )
    
    # Save everything
    joblib.dump(cph, output_dir / 'model_COX.pkl')
    
    train_set_with_outcome = X_train_scaled.copy()
    train_set_with_outcome.index = train_set.index
    train_set_with_outcome['TIME'] = y_train['TIME'].values
    train_set_with_outcome['EVENT'] = y_train['EVENT'].values
    
    test_set_with_outcome = X_test_scaled.copy()
    test_set_with_outcome.index = test_set.index
    test_set_with_outcome['TIME'] = y_test['TIME'].values
    test_set_with_outcome['EVENT'] = y_test['EVENT'].values
    
    ext_set_with_outcome = X_ext_scaled.copy()
    if not ext_set.empty:
        ext_set_with_outcome.index = ext_set.index
        ext_set_with_outcome['TIME'] = y_ext['TIME'].values
        ext_set_with_outcome['EVENT'] = y_ext['EVENT'].values
    
    save_datasets(output_dir, train_set_with_outcome, test_set_with_outcome,
                  ext_set_with_outcome, train_folds)
    
    cv_predictions = pd.DataFrame({
        'Subject': X_train_sel.index,
        'risk_score': cv_risks,
        'TIME': y_train['TIME'].values,
        'EVENT': y_train['EVENT'].values
    })
    cv_predictions.to_excel(output_dir / 'prediction_CV.xlsx', index=False)
    
    test_predictions = pd.DataFrame({
        'Subject': X_test_sel.index,
        'risk_score': risk_scores_test,
        'TIME': y_test['TIME'].values,
        'EVENT': y_test['EVENT'].astype(bool).values
    })
    test_predictions.to_excel(output_dir / 'prediction_TEST.xlsx', index=False)
    
    if not ext_tbl.empty:
        ext_predictions = pd.DataFrame({
            'Subject': X_ext_sel.index,
            'risk_score': risk_scores_ext,
            'TIME': y_ext['TIME'].values,
            'EVENT': y_ext['EVENT'].astype(bool).values
        })
        ext_predictions.to_excel(output_dir / 'prediction_EXVAL.xlsx', index=False)
    
    cv_auc_std = (cindex_cv['ci'][1] - cindex_cv['ci'][0]) / 2
    test_auc_std = (cindex_test['ci'][1] - cindex_test['ci'][0]) / 2
    ext_auc_std = (cindex_ext['ci'][1] - cindex_ext['ci'][0]) / 2 if not np.isnan(cindex_ext['c_index']) else np.nan
    
    results = pd.DataFrame({
        'SET': ['CV', 'TEST', 'EXVAL'],
        'C-INDEX': [
            f"{cindex_cv['c_index']:.2f} ± {cv_auc_std:.2f}",
            f"{cindex_test['c_index']:.2f} ± {test_auc_std:.2f}",
            f"{cindex_ext['c_index']:.2f} ± {ext_auc_std:.2f}" if not np.isnan(cindex_ext['c_index']) else 'N/A'
        ],
        'n': [len(y_train), len(y_test), len(y_ext) if not ext_set.empty else 0]
    })
    results.to_excel(output_dir / 'results.xlsx', index=False)
    
    print(f"\n✓ RWD-matched model completed!")
    print(f"  CV C-Index: {cindex_cv['c_index']:.3f} ± {cv_auc_std:.3f}")
    print(f"  Test C-Index: {cindex_test['c_index']:.3f} ± {test_auc_std:.3f}")
    if not np.isnan(cindex_ext['c_index']):
        print(f"  External C-Index: {cindex_ext['c_index']:.3f} ± {ext_auc_std:.3f}")
    
    return {
        'modality': f'{modality_folder}/rwd-only',
        'cv_cindex': cindex_cv['c_index'],
        'cv_cindex_std': cv_auc_std,
        'test_cindex': cindex_test['c_index'],
        'test_cindex_std': test_auc_std,
        'ext_cindex': cindex_ext['c_index'],
        'ext_cindex_std': ext_auc_std,
        'n_features': len(features),
        'n_train': len(y_train),
        'n_test': len(y_test),
        'n_ext': len(y_ext) if not ext_set.empty else 0
    }


def main():
    parser = argparse.ArgumentParser(description='Train MLEF CoxPH survival models for different modality combinations')
    parser.add_argument('--subanalysis', type=str, default='C23',
                        choices=['C23', 'C2', 'IO_ONLY', 'IO_CHT', 'LOW_PDL1', 
                                'HIGH_PDL1', 'SQUAMOUS', 'ADENOCARCINOMA'],
                        help='Subgroup analysis to perform (default: C23)')
    parser.add_argument('--modalities', type=str, nargs='+', default=None,
                        help='Modalities to train (default: all combinations). '
                             'Options: RWD, RWD_DP, RWD_FMRAD, RWD_PYRAD, RWD_DP_FMRAD, RWD_DP_PYRAD')
    parser.add_argument('--no-feature-selection', action='store_true',
                        help='Disable feature selection')
    parser.add_argument('--output-dir', type=str, default='mlef_pipeline',
                        help='Base output directory (default: mlef_pipeline)')
    
    args = parser.parse_args()
    
    # Convert string arguments to enums
    subanalysis = Subanalysis[args.subanalysis]
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
    print("MLEF SURVIVAL ANALYSIS TRAINING")
    print("="*80)
    print(f"Outcome: OS (Overall Survival)")
    print(f"Model: CoxPH")
    print(f"Subanalysis: {subanalysis.value}")
    print(f"Feature selection: {'CoxNet' if select_features else 'disabled'}")
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
                subanalysis=subanalysis,
                base_path=base_path,
                select_features=select_features
            )
            all_results.append(result)
            
            # Train RWD-matched model if not RWD-only
            if get_modality_folder_name(modes) != 'RWD':
                rwd_result = train_rwd_matched_model(
                    modes=modes,
                    subanalysis=subanalysis,
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
        summary_path = base_path / 'results' / 'OS' / subanalysis.value / 'training_summary.xlsx'
        summary_df.to_excel(summary_path, index=False)
        print(f"\n✓ Summary saved to {summary_path}")
    
    print("\n" + "="*80)
    print("ALL DONE! 🎉")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

