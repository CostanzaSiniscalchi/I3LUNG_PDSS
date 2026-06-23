import pandas as pd
import numpy as np
import math
import json
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import joblib

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent.parent / 'data'

# Load features.json
with open(DATA_DIR / 'features.json', 'r') as f:
    features_dict = json.load(f)

CB_CATEGORICAL_FEATURES = [
    "ECOG PS",
    "SEX",
    "SMOKING CURRENT",
    "SMOKING FORMER", 
    "SMOKING NEVER",
    "PDL1 CATEGORY",
    "PARENCHYMAL BRAIN METS AT IO START",
    "LIVER METS AT IO START",
    "BONE METS AT IO START",
]

CB_CATEGORICAL_BOUNDS: dict = {
    "ECOG PS":                            (0, 4),
    "SEX":                                (0, 1),
    "SMOKING CURRENT":                    (0, 1),
    "SMOKING FORMER":                     (0, 1),
    "SMOKING NEVER":                      (0, 1),
    "PDL1 CATEGORY":                      (0, 2),
    "PARENCHYMAL BRAIN METS AT IO START": (0, 1),
    "LIVER METS AT IO START":             (0, 1),
    "BONE METS AT IO START":              (0, 1),
}


def impute_df(df: pd.DataFrame, imputer=None, categorical_features: list = None, categorical_bounds: dict = None) -> Tuple[pd.DataFrame, IterativeImputer]:
    # use deterministic imputation
    cols_to_impute = [col for col in df.columns if col not in ['CENTER', 'SET']]
    metadata = df[['CENTER', 'SET']].copy()
    df_to_impute = df[cols_to_impute].copy()

    for col in df_to_impute.columns:
        df_to_impute[col] = pd.to_numeric(df_to_impute[col], errors='coerce')

    if categorical_features is None:
        categorical_features = []
    if categorical_bounds is None:
        categorical_bounds = {}

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

    was_sampling = imputer.sample_posterior
    imputer.sample_posterior = False
    try:
        imputed_df = pd.DataFrame(imputer.transform(df_to_impute), columns=df_to_impute.columns, index=df_to_impute.index)
    finally:
        imputer.sample_posterior = was_sampling

    for col in categorical_features:
        if col not in imputed_df.columns:
            continue
        lo, hi = categorical_bounds.get(col, (df_to_impute[col].min(), df_to_impute[col].max()))
        imputed_df[col] = imputed_df[col].clip(lower=lo, upper=hi).round(0).astype(int)

    imputed_df = pd.concat([metadata, imputed_df], axis=1)
    return imputed_df, imputer

def normalize(df: pd.DataFrame, scaler: StandardScaler=None, to_standard_normalize: list=None, to_log_normalize: list=None, min_shifts: dict=None) -> Tuple[pd.DataFrame, StandardScaler, list, list, dict]:
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

    # When called on train (min_shifts=None), compute shifts from train data.
    # When called on test/exval, use the train-derived shifts and clip any remaining negatives to 0.
    if min_shifts is None:
        min_shifts = {}
        for col in to_log_normalize:
            col_min = df_to_norm[col].min()
            if col_min < 0:
                min_shifts[col] = float(-col_min)
    for col in to_log_normalize:
        shift = min_shifts.get(col, 0.0)
        df_to_norm[col] = (df_to_norm[col] + shift).clip(lower=0)
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
    return result, scaler, to_standard_normalize, to_log_normalize, min_shifts

# CB
print("Processing CB...")
cb = pd.read_csv(DATA_DIR / 'cb.csv', index_col='Subject')
cb_cols = ['SET', 'CENTER'] + [f for f in features_dict['CB'] if f in cb.columns]
cb = cb[cb_cols]

train_mask = cb['SET'] == 'TRAIN'
cb_train_imputed, cb_imputer = impute_df(cb[train_mask], categorical_features=CB_CATEGORICAL_FEATURES, categorical_bounds=CB_CATEGORICAL_BOUNDS)
cb_train_proc, cb_scaler, cb_to_std, cb_to_log, cb_min_shifts = normalize(cb_train_imputed)

joblib.dump(cb_imputer, DATA_DIR / 'cb_imputer.pkl')
joblib.dump(cb_scaler,  DATA_DIR / 'cb_scaler.pkl')
with open(DATA_DIR / 'cb_norm_config.json', 'w') as f:
    json.dump({'to_standard_normalize': cb_to_std, 'to_log_normalize': cb_to_log, 'min_shifts': cb_min_shifts}, f)


test_mask = cb['SET'] == 'TEST'
cb_test_imputed, _ = impute_df(cb[test_mask], imputer=cb_imputer, categorical_features=CB_CATEGORICAL_FEATURES, categorical_bounds=CB_CATEGORICAL_BOUNDS)
cb_test_proc, _, _, _, _ = normalize(cb_test_imputed, scaler=cb_scaler, to_standard_normalize=cb_to_std, to_log_normalize=cb_to_log, min_shifts=cb_min_shifts)

ext_mask = cb['SET'] == 'EXVAL'
cb_ext_imputed, _ = impute_df(cb[ext_mask], imputer=cb_imputer, categorical_features=CB_CATEGORICAL_FEATURES, categorical_bounds=CB_CATEGORICAL_BOUNDS)
cb_ext_proc, _, _, _, _ = normalize(cb_ext_imputed, scaler=cb_scaler, to_standard_normalize=cb_to_std, to_log_normalize=cb_to_log, min_shifts=cb_min_shifts)

cb_result = pd.concat([cb_train_proc, cb_test_proc, cb_ext_proc])
cb_result.to_csv(DATA_DIR / 'cb_processed.csv')

# Genomics
print("Processing Genomics...")
gen = pd.read_csv(DATA_DIR / 'genomics.csv', index_col='Subject')
gen_cols = ['SET', 'CENTER'] + [f for f in features_dict['GEN'] if f in gen.columns]
gen = gen[gen_cols]

train_mask = gen['SET'] == 'TRAIN'
gen_train_imputed, gen_imputer = impute_df(gen[train_mask])
gen_train_proc, gen_scaler, gen_to_std, gen_to_log, gen_min_shifts = normalize(gen_train_imputed)

joblib.dump(gen_imputer, DATA_DIR / 'gen_imputer.pkl')
joblib.dump(gen_scaler,  DATA_DIR / 'gen_scaler.pkl')
with open(DATA_DIR / 'gen_norm_config.json', 'w') as f:
    json.dump({'to_standard_normalize': gen_to_std, 'to_log_normalize': gen_to_log, 'min_shifts': gen_min_shifts}, f)

test_mask = gen['SET'] == 'TEST'
gen_test_imputed, _ = impute_df(gen[test_mask], imputer=gen_imputer)
gen_test_proc, _, _, _, _ = normalize(gen_test_imputed, scaler=gen_scaler, to_standard_normalize=gen_to_std, to_log_normalize=gen_to_log, min_shifts=gen_min_shifts)

ext_mask = gen['SET'] == 'EXVAL'
gen_ext_imputed, _ = impute_df(gen[ext_mask], imputer=gen_imputer)
gen_ext_proc, _, _, _, _ = normalize(gen_ext_imputed, scaler=gen_scaler, to_standard_normalize=gen_to_std, to_log_normalize=gen_to_log, min_shifts=gen_min_shifts)

gen_result = pd.concat([gen_train_proc, gen_test_proc, gen_ext_proc])
gen_result.to_csv(DATA_DIR / 'genomics_processed.csv')

# Other modalities
for modality in ['digital_pathology', 'pyradiomics', 'fmrad']:
    print(f"Processing {modality}...")
    df = pd.read_csv(DATA_DIR / f'{modality}.csv', index_col='Subject')

    train_mask = df['SET'] == 'TRAIN'
    train_proc, scaler, to_std, to_log, mod_min_shifts = normalize(df[train_mask])

    test_mask = df['SET'] == 'TEST'
    test_proc, _, _, _, _ = normalize(df[test_mask], scaler=scaler, to_standard_normalize=to_std, to_log_normalize=to_log, min_shifts=mod_min_shifts)

    ext_mask = df['SET'] == 'EXVAL'
    ext_proc, _, _, _, _ = normalize(df[ext_mask], scaler=scaler, to_standard_normalize=to_std, to_log_normalize=to_log, min_shifts=mod_min_shifts)

    result = pd.concat([train_proc, test_proc, ext_proc])
    result.to_csv(DATA_DIR / f'{modality}_processed.csv')
    joblib.dump(scaler, DATA_DIR / f'{modality}_scaler.pkl')
    with open(DATA_DIR / f'{modality}_norm_config.json', 'w') as f:
        json.dump({'to_standard_normalize': to_std, 'to_log_normalize': to_log, 'min_shifts': mod_min_shifts}, f)

print("Done!")