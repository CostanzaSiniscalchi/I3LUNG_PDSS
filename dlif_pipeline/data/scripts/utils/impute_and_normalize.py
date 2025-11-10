import pandas as pd
import numpy as np
import math
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def impute_df(df: pd.DataFrame, imputer=None) -> Tuple[pd.DataFrame, IterativeImputer]:
    categorical_features = [col for col in df.columns if df[col].nunique() <= 10]

    if imputer is None:
        imputer = IterativeImputer(
            missing_values=np.nan,
            random_state=10,
            n_nearest_features=3,
            initial_strategy='median',
            max_iter=10,
            sample_posterior=True
        )
        imputer = imputer.fit(df)

    imputed_df = pd.DataFrame(imputer.transform(df), columns=df.columns, index=df.index)

    for col in categorical_features:
        min_value = df[col].min()
        max_value = df[col].max()
        imputed_df[col] = imputed_df[col].clip(upper=max_value, lower=min_value)
        imputed_df[col] = imputed_df[col].round(0).astype(int)

    return imputed_df, imputer

def normalize(df: pd.DataFrame, scaler: StandardScaler=None, to_standard_normalize: list=None, to_log_normalize: list=None) -> Tuple[pd.DataFrame, StandardScaler, list, list]:
    features_names = list(df.columns)
    if to_log_normalize is None:
        categorical_features = [col for col in df.columns if df[col].nunique() <= 10]
        to_log_normalize = [
            column for column in df.columns
            if abs(df[column].skew()) > 0.9
        ]
        to_log_normalize = [col for col in to_log_normalize if col not in categorical_features]

    for col in to_log_normalize:
        if df[col].min() < 0:
            df[col] = df[col] - df[col].min()
    df[to_log_normalize] = df[to_log_normalize].applymap(lambda x: math.log(x + 1))

    if to_standard_normalize is None:
        categorical_features = [col for col in df.columns if df[col].nunique() <= 10]
        to_standard_normalize = [col for col in features_names if col not in categorical_features]

    if scaler is None and len(to_standard_normalize) > 0:
        scaler = StandardScaler()
        scaler.fit(df[to_standard_normalize])

    if len(to_standard_normalize) > 0:
        df[to_standard_normalize] = scaler.transform(df[to_standard_normalize])
    df = df.rename(columns={col: f'log_{col}' for col in to_log_normalize})

    return df, scaler, to_standard_normalize, to_log_normalize

# Load RWD
rwd_train = pd.read_csv('Mil2/data/data/split/rwd_train.csv', index_col='Subject')
rwd_test = pd.read_csv('Mil2/data/data/split/rwd_test.csv', index_col='Subject')
rwd_uoc = pd.read_csv('Mil2/data/data/split/rwd_uoc.csv', index_col='Subject')

# Impute and normalize RWD
print("Processing RWD...")
rwd_train_imputed, rwd_imputer = impute_df(rwd_train)
rwd_test_imputed, _ = impute_df(rwd_test, imputer=rwd_imputer)
rwd_uoc_imputed, _ = impute_df(rwd_uoc, imputer=rwd_imputer)

rwd_train_proc, rwd_scaler, rwd_to_std, rwd_to_log = normalize(rwd_train_imputed)
rwd_test_proc, _, _, _ = normalize(rwd_test_imputed, scaler=rwd_scaler, to_standard_normalize=rwd_to_std, to_log_normalize=rwd_to_log)
rwd_uoc_proc, _, _, _ = normalize(rwd_uoc_imputed, scaler=rwd_scaler, to_standard_normalize=rwd_to_std, to_log_normalize=rwd_to_log)

# Save RWD
rwd_train_proc.to_csv('Mil2/data/data/split/rwd_train_processed.csv')
rwd_test_proc.to_csv('Mil2/data/data/split/rwd_test_processed.csv')
rwd_uoc_proc.to_csv('Mil2/data/data/split/rwd_uoc_processed.csv')

# Load Genomics
gen_train = pd.read_csv('Mil2/data/data/split/genomics_train.csv', index_col='Subject')
gen_test = pd.read_csv('Mil2/data/data/split/genomics_test.csv', index_col='Subject')
gen_uoc = pd.read_csv('Mil2/data/data/split/genomics_uoc.csv', index_col='Subject')

# Impute and normalize Genomics
print("Processing Genomics...")
gen_train_imputed, gen_imputer = impute_df(gen_train)
gen_test_imputed, _ = impute_df(gen_test, imputer=gen_imputer)
gen_uoc_imputed, _ = impute_df(gen_uoc, imputer=gen_imputer)

gen_train_proc, gen_scaler, gen_to_std, gen_to_log = normalize(gen_train_imputed)
gen_test_proc, _, _, _ = normalize(gen_test_imputed, scaler=gen_scaler, to_standard_normalize=gen_to_std, to_log_normalize=gen_to_log)
gen_uoc_proc, _, _, _ = normalize(gen_uoc_imputed, scaler=gen_scaler, to_standard_normalize=gen_to_std, to_log_normalize=gen_to_log)

# Save Genomics
gen_train_proc.to_csv('Mil2/data/data/split/genomics_train_processed.csv')
gen_test_proc.to_csv('Mil2/data/data/split/genomics_test_processed.csv')
gen_uoc_proc.to_csv('Mil2/data/data/split/genomics_uoc_processed.csv')

# Standardize other modalities
for modality in ['digital_pathology', 'pyradiomics', 'fmrad']:
    print(f"Processing {modality}...")
    train = pd.read_csv(f'Mil2/data/data/split/{modality}_train.csv', index_col='Subject')
    test = pd.read_csv(f'Mil2/data/data/split/{modality}_test.csv', index_col='Subject')
    uoc = pd.read_csv(f'Mil2/data/data/split/{modality}_uoc.csv', index_col='Subject')
    
    scaler = StandardScaler()
    train_proc = pd.DataFrame(scaler.fit_transform(train), columns=train.columns, index=train.index)
    test_proc = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)
    uoc_proc = pd.DataFrame(scaler.transform(uoc), columns=uoc.columns, index=uoc.index)
    
    train_proc.to_csv(f'Mil2/data/data/split/{modality}_train_processed.csv')
    test_proc.to_csv(f'Mil2/data/data/split/{modality}_test_processed.csv')
    uoc_proc.to_csv(f'Mil2/data/data/split/{modality}_uoc_processed.csv')

print("Done!")