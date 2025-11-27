import pandas as pd
import json
from pathlib import Path

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / 'data'

def create_feature_dataset_from_processed(
    rad_type: str = 'pyradiomics'
) -> pd.DataFrame:
    
    # Load processed data
    rwd = pd.read_csv(DATA_DIR / 'rwd_processed.csv', index_col='Subject')
    rad = pd.read_csv(DATA_DIR / f'{rad_type}_processed.csv', index_col='Subject')
    dp = pd.read_csv(DATA_DIR / 'digital_pathology_processed.csv', index_col='Subject')
    genomics = pd.read_csv(DATA_DIR / 'genomics_processed.csv', index_col='Subject')
    
    # Get RWD subjects (master list)
    rwd_subjects = set(rwd.index)
    
    # Filter other modalities to keep only RWD subjects
    rad = rad[rad.index.isin(rwd_subjects)]
    dp = dp[dp.index.isin(rwd_subjects)]
    genomics = genomics[genomics.index.isin(rwd_subjects)]
    
    # Drop SET and CENTER from features
    cols_to_drop = ['SET', 'CENTER']
    rwd_features = rwd.drop(columns=[c for c in cols_to_drop if c in rwd.columns])
    rad_features = rad.drop(columns=[c for c in cols_to_drop if c in rad.columns])
    dp_features = dp.drop(columns=[c for c in cols_to_drop if c in dp.columns])
    gen_features = genomics.drop(columns=[c for c in cols_to_drop if c in genomics.columns])
    
    # Create base dataframe
    df = pd.DataFrame(index=rwd.index)
    df['Subject'] = df.index
    
    # Add mod1 (RWD - always present)
    df['mod1'] = rwd_features.apply(lambda r: r.tolist(), axis=1)
    
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

# Create both versions
print("Creating pyradiomics version...")
df_pyrad = create_feature_dataset_from_processed(rad_type='pyradiomics')
df_pyrad.to_parquet(DATA_DIR / 'features_dataset_radpy_fixed.parquet', index=False)

print("\nCreating fmrad version...")
df_fmrad = create_feature_dataset_from_processed(rad_type='fmrad')
df_fmrad.to_parquet(DATA_DIR / 'features_dataset_fmrad.parquet', index=False)

print("\nDone!") 