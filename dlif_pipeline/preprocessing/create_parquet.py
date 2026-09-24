import argparse
import pandas as pd
import json
from pathlib import Path

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent.parent / 'data'

def create_feature_dataset_from_processed(
    rad_type: str = 'pyradiomics',
    dp_type: str = 'digital_pathology',
    data_dir: Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    data_dir = Path(data_dir)

    # Load processed data
    cb = pd.read_csv(data_dir / 'cb_processed.csv', index_col='Subject')
    rad = pd.read_csv(data_dir / f'{rad_type}_processed.csv', index_col='Subject')
    dp = pd.read_csv(data_dir / f'{dp_type}_processed.csv', index_col='Subject')
    genomics = pd.read_csv(data_dir / 'genomics_processed.csv', index_col='Subject')
    
    # Get CB subjects (master list)
    cb_subjects = set(cb.index)
    
    # Filter other modalities to keep only CB subjects
    rad = rad[rad.index.isin(cb_subjects)]
    dp = dp[dp.index.isin(cb_subjects)]
    genomics = genomics[genomics.index.isin(cb_subjects)]
    
    # Drop SET and CENTER from features
    cols_to_drop = ['SET', 'CENTER']
    cb_features = cb.drop(columns=[c for c in cols_to_drop if c in cb.columns])
    rad_features = rad.drop(columns=[c for c in cols_to_drop if c in rad.columns])
    dp_features = dp.drop(columns=[c for c in cols_to_drop if c in dp.columns])
    gen_features = genomics.drop(columns=[c for c in cols_to_drop if c in genomics.columns])
    
    # Create base dataframe
    df = pd.DataFrame(index=cb.index)
    df['Subject'] = df.index
    
    # Add mod1 (CB - always present)
    df['mod1'] = cb_features.apply(lambda r: r.tolist(), axis=1)
    
    # Add mod2 (radiomics) - None if subject not in rad
    df['mod2'] = df['Subject'].apply(
        lambda s: rad_features.loc[s].tolist() if s in rad_features.index else None
    )
    
    # Add mod3 (digital pathology) - None if subject not in dp
    df['mod3'] = df['Subject'].apply(
        lambda s: dp_features.loc[s].tolist() if s in dp_features.index else None
    )
    
    # Add mod4 (genomics) - None if subject not in genomics OR in exclusion list
    df['mod4'] = df['Subject'].apply(
        lambda s: gen_features.loc[s].tolist() if s in gen_features.index else None
    )
    
    # Null-out all-NaN lists
    for mod in ['mod1', 'mod2', 'mod3', 'mod4']:
        df[mod] = df[mod].apply(
            lambda lst: None if isinstance(lst, list) and all(pd.isna(x) for x in lst) else lst
        )
    
    print(f"Final shape: {df.shape}")
    for mod in ['mod1', 'mod2', 'mod3', 'mod4']:
        non_null = df[mod].notna().sum()
        print(f"{mod}: {non_null} non-null entries")
    
    return df

# Filenames kept as-is for backward compatibility with existing configs/data.
RAD_OUTPUT_STEMS = {
    'pyradiomics': 'features_dataset_radpy_fixed',
    'fmrad': 'features_dataset_fmrad',
}

# Suffix appended to the output filename for each dp_type. '' for the
# legacy/default gigapath source (predates this split, no token in the
# name); everything else must contain a unique token — this must match
# what prepare_dataset.py's SOURCE_VARIANT_GROUPS looks for.
DP_OUTPUT_SUFFIXES = {
    'digital_pathology': '',
    'digital_pathology_titan': '_dp-titan',
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Build the DLIF multimodal feature parquet files from processed modality CSVs.")
    parser.add_argument('--data-dir', default=str(DEFAULT_DATA_DIR), help=f"Directory containing the *_processed.csv files and where the parquet outputs are written (default: {DEFAULT_DATA_DIR})")
    parser.add_argument('--rad-types', nargs='+', default=['pyradiomics', 'fmrad'], help="Radiomics source(s) to build a parquet file for. Each produces '<data-dir>/features_dataset_<name>.parquet' (pyradiomics is saved as 'features_dataset_radpy_fixed.parquet' for backward compatibility).")
    parser.add_argument('--dp-types', nargs='+', default=['digital_pathology'], choices=list(DP_OUTPUT_SUFFIXES), help="Digital pathology source(s) to build a parquet file for. 'digital_pathology' (gigapath, default) keeps the legacy filename; 'digital_pathology_titan' appends '_dp-titan' so it doesn't collide with the gigapath parquet.")
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    data_dir = Path(args.data_dir)

    for rad_type in args.rad_types:
        rad_stem = RAD_OUTPUT_STEMS.get(rad_type, f'features_dataset_{rad_type}')
        for dp_type in args.dp_types:
            print(f"Creating {rad_type} / {dp_type} version...")
            df = create_feature_dataset_from_processed(rad_type=rad_type, dp_type=dp_type, data_dir=data_dir)
            output_name = f"{rad_stem}{DP_OUTPUT_SUFFIXES[dp_type]}.parquet"
            df.to_parquet(data_dir / output_name, index=False)

    print("\nDone!")