import argparse
import pandas as pd
import json
import sys
from pathlib import Path
import joblib

SCRIPT_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent.parent.parent / 'data'
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.preprocessing import (
    impute_df,
    normalize,
    CB_CATEGORICAL_FEATURES,
    CB_CATEGORICAL_BOUNDS,
    GEN_CATEGORICAL_FEATURES,
    GEN_CATEGORICAL_BOUNDS,
)


def impute_and_normalize(
    data_dir: Path = DEFAULT_DATA_DIR,
    features_json_path: Path = None,
    other_modalities=('digital_pathology', 'pyradiomics', 'fmrad'),
):
    data_dir = Path(data_dir)
    if features_json_path is None:
        features_json_path = data_dir / 'features.json'

    # Load features.json
    with open(features_json_path, 'r') as f:
        features_dict = json.load(f)

    # CB
    print("Processing CB...")
    cb = pd.read_csv(data_dir / 'cb.csv', index_col='Subject')
    # only keep columns that are in features.json and 'SET' and 'CENTER'
    cb_cols = ['SET', 'CENTER'] + [f for f in features_dict['CB'] if f in cb.columns]
    cb = cb[cb_cols]

    train_mask = cb['SET'] == 'TRAIN'
    cb_train_imputed, cb_imputer = impute_df(cb[train_mask], categorical_features=CB_CATEGORICAL_FEATURES, categorical_bounds=CB_CATEGORICAL_BOUNDS)
    cb_train_proc, cb_scaler, cb_to_std, cb_to_log, cb_min_shifts = normalize(cb_train_imputed, categorical_features=CB_CATEGORICAL_FEATURES)

    joblib.dump(cb_imputer, data_dir / 'cb_imputer.pkl')
    joblib.dump(cb_scaler, data_dir / 'cb_scaler.pkl')
    with open(data_dir / 'cb_norm_config.json', 'w') as f:
        json.dump({'to_standard_normalize': cb_to_std, 'to_log_normalize': cb_to_log, 'min_shifts': cb_min_shifts}, f)

    test_mask = cb['SET'] == 'TEST'
    cb_test_imputed, _ = impute_df(cb[test_mask], imputer=cb_imputer, categorical_features=CB_CATEGORICAL_FEATURES, categorical_bounds=CB_CATEGORICAL_BOUNDS)
    cb_test_proc, _, _, _, _ = normalize(cb_test_imputed, scaler=cb_scaler, to_standard_normalize=cb_to_std, to_log_normalize=cb_to_log, min_shifts=cb_min_shifts)

    ext_mask = cb['SET'] == 'EXVAL'
    cb_ext_imputed, _ = impute_df(cb[ext_mask], imputer=cb_imputer, categorical_features=CB_CATEGORICAL_FEATURES, categorical_bounds=CB_CATEGORICAL_BOUNDS)
    cb_ext_proc, _, _, _, _ = normalize(cb_ext_imputed, scaler=cb_scaler, to_standard_normalize=cb_to_std, to_log_normalize=cb_to_log, min_shifts=cb_min_shifts)

    cb_result = pd.concat([cb_train_proc, cb_test_proc, cb_ext_proc])
    cb_result.to_csv(data_dir / 'cb_processed.csv')

    # Genomics
    print("Processing Genomics...")
    gen = pd.read_csv(data_dir / 'genomics.csv', index_col='Subject')
    gen_cols = ['SET', 'CENTER'] + [f for f in features_dict['GEN'] if f in gen.columns]
    gen = gen[gen_cols]

    train_mask = gen['SET'] == 'TRAIN'
    gen_train_imputed, gen_imputer = impute_df(gen[train_mask], categorical_features=GEN_CATEGORICAL_FEATURES, categorical_bounds=GEN_CATEGORICAL_BOUNDS)
    gen_train_proc, gen_scaler, gen_to_std, gen_to_log, gen_min_shifts = normalize(gen_train_imputed, categorical_features=GEN_CATEGORICAL_FEATURES)

    joblib.dump(gen_imputer, data_dir / 'gen_imputer.pkl')
    joblib.dump(gen_scaler, data_dir / 'gen_scaler.pkl')
    with open(data_dir / 'gen_norm_config.json', 'w') as f:
        json.dump({'to_standard_normalize': gen_to_std, 'to_log_normalize': gen_to_log, 'min_shifts': gen_min_shifts}, f)

    test_mask = gen['SET'] == 'TEST'
    gen_test_imputed, _ = impute_df(gen[test_mask], imputer=gen_imputer, categorical_features=GEN_CATEGORICAL_FEATURES, categorical_bounds=GEN_CATEGORICAL_BOUNDS)
    gen_test_proc, _, _, _, _ = normalize(gen_test_imputed, scaler=gen_scaler, to_standard_normalize=gen_to_std, to_log_normalize=gen_to_log, min_shifts=gen_min_shifts)

    ext_mask = gen['SET'] == 'EXVAL'
    gen_ext_imputed, _ = impute_df(gen[ext_mask], imputer=gen_imputer, categorical_features=GEN_CATEGORICAL_FEATURES, categorical_bounds=GEN_CATEGORICAL_BOUNDS)
    gen_ext_proc, _, _, _, _ = normalize(gen_ext_imputed, scaler=gen_scaler, to_standard_normalize=gen_to_std, to_log_normalize=gen_to_log, min_shifts=gen_min_shifts)

    gen_result = pd.concat([gen_train_proc, gen_test_proc, gen_ext_proc])
    gen_result.to_csv(data_dir / 'genomics_processed.csv')

    # Other modalities (digital pathology, pyradiomics, fmrad, and any
    # alternate-source variant such as digital_pathology_titan): assumed
    # complete, so the imputation step is skipped — only normalization runs.
    for modality in other_modalities:
        raw_path = data_dir / f'{modality}.csv'
        if not raw_path.exists():
            print(f"[SKIP] {modality}: {raw_path} not found")
            continue

        print(f"Processing {modality}...")
        df = pd.read_csv(raw_path, index_col='Subject')

        train_mask = df['SET'] == 'TRAIN'
        train_proc, scaler, to_std, to_log, mod_min_shifts = normalize(df[train_mask])

        test_mask = df['SET'] == 'TEST'
        test_proc, _, _, _, _ = normalize(df[test_mask], scaler=scaler, to_standard_normalize=to_std, to_log_normalize=to_log, min_shifts=mod_min_shifts)

        ext_mask = df['SET'] == 'EXVAL'
        ext_proc, _, _, _, _ = normalize(df[ext_mask], scaler=scaler, to_standard_normalize=to_std, to_log_normalize=to_log, min_shifts=mod_min_shifts)

        result = pd.concat([train_proc, test_proc, ext_proc])
        result.to_csv(data_dir / f'{modality}_processed.csv')
        joblib.dump(scaler, data_dir / f'{modality}_scaler.pkl')
        with open(data_dir / f'{modality}_norm_config.json', 'w') as f:
            json.dump({'to_standard_normalize': to_std, 'to_log_normalize': to_log, 'min_shifts': mod_min_shifts}, f)

    print("Done!")


def _parse_args():
    parser = argparse.ArgumentParser(description="Impute and normalize the raw DLIF modality CSVs (CB, genomics, digital pathology, pyradiomics, fmrad).")
    parser.add_argument('--data-dir', default=str(DEFAULT_DATA_DIR), help=f"Directory containing the raw modality CSVs and features.json; processed outputs are written here too (default: {DEFAULT_DATA_DIR})")
    parser.add_argument('--features-json-path', default=None, help="Path to features.json (default: <data-dir>/features.json)")
    parser.add_argument(
        '--other-modalities', nargs='+', default=['digital_pathology', 'pyradiomics', 'fmrad'],
        help="Normalize-only modalities to process, each read from '<data-dir>/<name>.csv'. "
             "Add an alternate-source variant here (e.g. 'digital_pathology_titan') to produce "
             "its own <name>_processed.csv / _scaler.pkl / _norm_config.json for use with "
             "create_parquet.py's --dp-types (default: digital_pathology pyradiomics fmrad)."
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    impute_and_normalize(data_dir=Path(args.data_dir), features_json_path=args.features_json_path, other_modalities=args.other_modalities)
