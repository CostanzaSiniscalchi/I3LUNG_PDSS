import pandas as pd
import numpy as np
import math
import json
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from typing import Tuple

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent.parent / 'data'

# Load features.json
with open(DATA_DIR / 'features.json', 'r') as f:
    features_dict = json.load(f)

def impute_df(df: pd.DataFrame, imputer=None) -> Tuple[pd.DataFrame, IterativeImputer]:
    cols_to_impute = [col for col in df.columns if col not in ['CENTER', 'SET']]
    metadata = df[['CENTER', 'SET']].copy()
    df_to_impute = df[cols_to_impute].copy()
    
    for col in df_to_impute.columns:
        df_to_impute[col] = pd.to_numeric(df_to_impute[col], errors='coerce')
    
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

    imputed_df = pd.concat([metadata, imputed_df], axis=1)
    return imputed_df, imputer

def normalize(df: pd.DataFrame, scaler: StandardScaler=None, to_standard_normalize: list=None, to_log_normalize: list=None) -> Tuple[pd.DataFrame, StandardScaler, list, list]:
    metadata = df[['CENTER', 'SET']].copy()
    df_to_norm = df.drop(columns=['CENTER', 'SET']).copy()
    
    for col in df_to_norm.columns:
        df_to_norm[col] = pd.to_numeric(df_to_norm[col], errors='coerce')
    
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

# CB
print("Processing CB...")
cb = pd.read_csv(DATA_DIR / 'cb.csv', index_col='Subject')
cb_cols = ['SET', 'CENTER'] + [f for f in features_dict['CB'] if f in cb.columns]
cb = cb[cb_cols]

train_mask = cb['SET'] == 'TRAIN'
cb_train_imputed, cb_imputer = impute_df(cb[train_mask])
cb_train_proc, cb_scaler, cb_to_std, cb_to_log = normalize(cb_train_imputed)

test_mask = cb['SET'] == 'TEST'
cb_test_imputed, _ = impute_df(cb[test_mask], imputer=cb_imputer)
cb_test_proc, _, _, _ = normalize(cb_test_imputed, scaler=cb_scaler, to_standard_normalize=cb_to_std, to_log_normalize=cb_to_log)

ext_mask = cb['SET'] == 'EXVAL'
cb_ext_imputed, _ = impute_df(cb[ext_mask], imputer=cb_imputer)
cb_ext_proc, _, _, _ = normalize(cb_ext_imputed, scaler=cb_scaler, to_standard_normalize=cb_to_std, to_log_normalize=cb_to_log)

cb_result = pd.concat([cb_train_proc, cb_test_proc, cb_ext_proc])
cb_result.to_csv(DATA_DIR / 'cb_processed.csv')

# Genomics
print("Processing Genomics...")
gen = pd.read_csv(DATA_DIR / 'genomics.csv', index_col='Subject')
gen_cols = ['SET', 'CENTER'] + [f for f in features_dict['GEN'] if f in gen.columns]
gen = gen[gen_cols]

train_mask = gen['SET'] == 'TRAIN'
gen_train_imputed, gen_imputer = impute_df(gen[train_mask])
gen_train_proc, gen_scaler, gen_to_std, gen_to_log = normalize(gen_train_imputed)

test_mask = gen['SET'] == 'TEST'
gen_test_imputed, _ = impute_df(gen[test_mask], imputer=gen_imputer)
gen_test_proc, _, _, _ = normalize(gen_test_imputed, scaler=gen_scaler, to_standard_normalize=gen_to_std, to_log_normalize=gen_to_log)

ext_mask = gen['SET'] == 'EXVAL'
gen_ext_imputed, _ = impute_df(gen[ext_mask], imputer=gen_imputer)
gen_ext_proc, _, _, _ = normalize(gen_ext_imputed, scaler=gen_scaler, to_standard_normalize=gen_to_std, to_log_normalize=gen_to_log)

gen_result = pd.concat([gen_train_proc, gen_test_proc, gen_ext_proc])
gen_result.to_csv(DATA_DIR / 'genomics_processed.csv')

# Other modalities
for modality in ['digital_pathology', 'pyradiomics', 'fmrad']:
    print(f"Processing {modality}...")
    df = pd.read_csv(DATA_DIR / f'{modality}.csv', index_col='Subject')
    
    train_mask = df['SET'] == 'TRAIN'
    train_proc, scaler, to_std, to_log = normalize(df[train_mask])
    
    test_mask = df['SET'] == 'TEST'
    test_proc, _, _, _ = normalize(df[test_mask], scaler=scaler, to_standard_normalize=to_std, to_log_normalize=to_log)
    
    ext_mask = df['SET'] == 'EXVAL'
    ext_proc, _, _, _ = normalize(df[ext_mask], scaler=scaler, to_standard_normalize=to_std, to_log_normalize=to_log)
    
    result = pd.concat([train_proc, test_proc, ext_proc])
    result.to_csv(DATA_DIR / f'{modality}_processed.csv')

print("Done!")