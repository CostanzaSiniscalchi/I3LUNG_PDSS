import pandas as pd
import numpy as np
import math
import pickle
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent.parent / 'data'

def impute_df(df: pd.DataFrame, imputer=None) -> Tuple[pd.DataFrame, IterativeImputer]:
    cols_to_impute = [col for col in df.columns if col not in ['CENTER', 'SET']]
    metadata = df[['CENTER', 'SET']].copy() if 'CENTER' in df.columns else None
    df_to_impute = df[cols_to_impute]
    
    categorical_features = [col for col in df_to_impute.columns if df_to_impute[col].nunique() <= 10]

    if imputer is None:
        imputer = IterativeImputer(
            missing_values=np.nan,
            random_state=10,
            n_nearest_features=3,
            initial_strategy='median',
            max_iter=10,
            sample_posterior=True
        )
        imputer = imputer.fit(df_to_impute)

    imputed_df = pd.DataFrame(imputer.transform(df_to_impute), columns=df_to_impute.columns, index=df_to_impute.index)

    for col in categorical_features:
        min_value = df_to_impute[col].min()
        max_value = df_to_impute[col].max()
        imputed_df[col] = imputed_df[col].clip(upper=max_value, lower=min_value)
        imputed_df[col] = imputed_df[col].round(0).astype(int)

    if metadata is not None:
        imputed_df = pd.concat([metadata, imputed_df], axis=1)
    
    return imputed_df, imputer

def normalize(df: pd.DataFrame, scaler: StandardScaler=None, to_standard_normalize: list=None, to_log_normalize: list=None) -> Tuple[pd.DataFrame, StandardScaler, list, list]:
    metadata_cols = ['CENTER', 'SET']
    metadata = df[[col for col in metadata_cols if col in df.columns]].copy()
    df_to_norm = df.drop(columns=[col for col in metadata_cols if col in df.columns])
    
    features_names = list(df_to_norm.columns)
    if to_log_normalize is None:
        categorical_features = [col for col in df_to_norm.columns if df_to_norm[col].nunique() <= 10]
        to_log_normalize = [
            column for column in df_to_norm.columns
            if abs(df_to_norm[column].skew()) > 0.9
        ]
        to_log_normalize = [col for col in to_log_normalize if col not in categorical_features]

    for col in to_log_normalize:
        if df_to_norm[col].min() < 0:
            df_to_norm[col] = df_to_norm[col] - df_to_norm[col].min()
    df_to_norm[to_log_normalize] = df_to_norm[to_log_normalize].map(lambda x: math.log(x + 1))

    if to_standard_normalize is None:
        categorical_features = [col for col in df_to_norm.columns if df_to_norm[col].nunique() <= 10]
        to_standard_normalize = [col for col in features_names if col not in categorical_features]

    if scaler is None and len(to_standard_normalize) > 0:
        scaler = StandardScaler()
        scaler.fit(df_to_norm[to_standard_normalize])

    if len(to_standard_normalize) > 0:
        df_to_norm[to_standard_normalize] = scaler.transform(df_to_norm[to_standard_normalize])
    df_to_norm = df_to_norm.rename(columns={col: f'log_{col}' for col in to_log_normalize})

    result = pd.concat([metadata, df_to_norm], axis=1)
    return result, scaler, to_standard_normalize, to_log_normalize

# Load RWD
rwd_train = pd.read_csv(DATA_DIR / 'split' / 'rwd_train.csv', index_col='Subject')
rwd_test = pd.read_csv(DATA_DIR / 'split' / 'rwd_test.csv', index_col='Subject')
rwd_ext_val = pd.read_csv(DATA_DIR / 'split' / 'rwd_ext_val.csv', index_col='Subject')

# Impute and normalize RWD
print("Processing RWD...")
rwd_train_imputed, rwd_imputer = impute_df(rwd_train)
rwd_test_imputed, _ = impute_df(rwd_test, imputer=rwd_imputer)
rwd_ext_val_imputed, _ = impute_df(rwd_ext_val, imputer=rwd_imputer)

rwd_train_proc, rwd_scaler, rwd_to_std, rwd_to_log = normalize(rwd_train_imputed)
rwd_test_proc, _, _, _ = normalize(rwd_test_imputed, scaler=rwd_scaler, to_standard_normalize=rwd_to_std, to_log_normalize=rwd_to_log)
rwd_ext_val_proc, _, _, _ = normalize(rwd_ext_val_imputed, scaler=rwd_scaler, to_standard_normalize=rwd_to_std, to_log_normalize=rwd_to_log)

# Save RWD
rwd_train_proc.to_csv(DATA_DIR / 'split' / 'rwd_train_processed.csv')
rwd_test_proc.to_csv(DATA_DIR / 'split' / 'rwd_test_processed.csv')
rwd_ext_val_proc.to_csv(DATA_DIR / 'split' / 'rwd_ext_val_processed.csv')

# Load Genomics
gen_train = pd.read_csv(DATA_DIR / 'split' / 'genomics_train.csv', index_col='Subject')
gen_test = pd.read_csv(DATA_DIR / 'split' / 'genomics_test.csv', index_col='Subject')
gen_ext_val = pd.read_csv(DATA_DIR / 'split' / 'genomics_ext_val.csv', index_col='Subject')

# Impute and normalize Genomics
print("Processing Genomics...")
gen_train_imputed, gen_imputer = impute_df(gen_train)
gen_test_imputed, _ = impute_df(gen_test, imputer=gen_imputer)
gen_ext_val_imputed, _ = impute_df(gen_ext_val, imputer=gen_imputer)

gen_train_proc, gen_scaler, gen_to_std, gen_to_log = normalize(gen_train_imputed)
gen_test_proc, _, _, _ = normalize(gen_test_imputed, scaler=gen_scaler, to_standard_normalize=gen_to_std, to_log_normalize=gen_to_log)
gen_ext_val_proc, _, _, _ = normalize(gen_ext_val_imputed, scaler=gen_scaler, to_standard_normalize=gen_to_std, to_log_normalize=gen_to_log)

# Save Genomics
gen_train_proc.to_csv(DATA_DIR / 'split' / 'genomics_train_processed.csv')
gen_test_proc.to_csv(DATA_DIR / 'split' / 'genomics_test_processed.csv')
gen_ext_val_proc.to_csv(DATA_DIR / 'split' / 'genomics_ext_val_processed.csv')

# Standardize other modalities
for modality in ['digital_pathology', 'pyradiomics', 'fmrad']:
    print(f"Processing {modality}...")
    train = pd.read_csv(DATA_DIR / 'split' / f'{modality}_train.csv', index_col='Subject')
    test = pd.read_csv(DATA_DIR / 'split' / f'{modality}_test.csv', index_col='Subject')
    ext_val = pd.read_csv(DATA_DIR / 'split' / f'{modality}_ext_val.csv', index_col='Subject')

    metadata_cols = ['CENTER', 'SET']
    train_metadata = train[[col for col in metadata_cols if col in train.columns]]
    test_metadata = test[[col for col in metadata_cols if col in test.columns]]
    ext_val_metadata = ext_val[[col for col in metadata_cols if col in ext_val.columns]]
    
    train_features = train.drop(columns=[col for col in metadata_cols if col in train.columns])
    test_features = test.drop(columns=[col for col in metadata_cols if col in test.columns])
    ext_val_features = ext_val.drop(columns=[col for col in metadata_cols if col in ext_val.columns])

    scaler = StandardScaler()
    train_proc = pd.DataFrame(scaler.fit_transform(train_features), columns=train_features.columns, index=train_features.index)
    test_proc = pd.DataFrame(scaler.transform(test_features), columns=test_features.columns, index=test_features.index)
    ext_val_proc = pd.DataFrame(scaler.transform(ext_val_features), columns=ext_val_features.columns, index=ext_val_features.index)

    train_proc = pd.concat([train_metadata, train_proc], axis=1)
    test_proc = pd.concat([test_metadata, test_proc], axis=1)
    ext_val_proc = pd.concat([ext_val_metadata, ext_val_proc], axis=1)

    train_proc.to_csv(DATA_DIR / 'split' / f'{modality}_train_processed.csv')
    test_proc.to_csv(DATA_DIR / 'split' / f'{modality}_test_processed.csv')
    ext_val_proc.to_csv(DATA_DIR / 'split' / f'{modality}_ext_val_processed.csv')

print("Done!")