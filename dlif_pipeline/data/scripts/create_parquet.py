import pandas as pd
import json

def create_feature_dataset_from_processed(
    base_path: str = 'Mil2/data/data/split',
    rad_type: str = 'pyradiomics'
) -> pd.DataFrame:
    
    # Load genomics exclusion lists
    with open('Mil2/data/data/no_genomics_train.json', 'r') as f:
        no_gen_train = json.load(f)
    with open('Mil2/data/data/no_genomics_test.json', 'r') as f:
        no_gen_test = json.load(f)
    with open('Mil2/data/data/no_genomics_ext_val.json', 'r') as f:
        no_gen_ext = json.load(f)
    
    exclude_genomics = set(no_gen_train + no_gen_test + no_gen_ext)
    
    dfs = []
    
    for split in ['train', 'test', 'uoc']:
        # Load processed data
        rwd = pd.read_csv(f'{base_path}/rwd_{split}_processed.csv')
        rad = pd.read_csv(f'{base_path}/{rad_type}_{split}_processed.csv')
        dp = pd.read_csv(f'{base_path}/digital_pathology_{split}_processed.csv')
        genomics = pd.read_csv(f'{base_path}/genomics_{split}_processed.csv')
        
        # Get RWD subjects (master list)
        rwd_subjects = set(rwd['Subject'])
        
        # Filter other modalities to keep only RWD subjects
        rad = rad[rad['Subject'].isin(rwd_subjects)]
        dp = dp[dp['Subject'].isin(rwd_subjects)]
        genomics = genomics[genomics['Subject'].isin(rwd_subjects)]
        
        # Create base dataframe with all RWD subjects
        df_split = pd.DataFrame({'Subject': rwd['Subject']})
        
        # Add mod1 (RWD - always present)
        df_split['mod1'] = rwd.drop('Subject', axis=1).apply(lambda r: r.tolist(), axis=1)
        
        # Add mod2 (radiomics) - None if subject not in rad
        df_split['mod2'] = df_split['Subject'].apply(
            lambda s: rad[rad['Subject'] == s].drop('Subject', axis=1).values[0].tolist() 
            if s in rad['Subject'].values else None
        )
        
        # Add mod3 (digital pathology) - None if subject not in dp
        df_split['mod3'] = df_split['Subject'].apply(
            lambda s: dp[dp['Subject'] == s].drop('Subject', axis=1).values[0].tolist() 
            if s in dp['Subject'].values else None
        )
        
        # Add mod4 (genomics) - None if subject not in genomics OR in exclusion list
        df_split['mod4'] = df_split['Subject'].apply(
            lambda s: genomics[genomics['Subject'] == s].drop('Subject', axis=1).values[0].tolist() 
            if (s in genomics['Subject'].values and s not in exclude_genomics) else None
        )
        
        dfs.append(df_split)
    
    # Merge all splits
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Null-out all-NaN lists
    for mod in ['mod1', 'mod2', 'mod3', 'mod4']:
        full_df[mod] = full_df[mod].apply(
            lambda lst: None if isinstance(lst, list) and all(pd.isna(x) for x in lst) else lst
        )
    
    print(f"Final shape: {full_df.shape}")
    for mod in ['mod1', 'mod2', 'mod3', 'mod4']:
        non_null = full_df[mod].notna().sum()
        print(f"{mod}: {non_null} non-null entries")
    
    return full_df

# Create both versions
print("Creating pyradiomics version...")
df_pyrad = create_feature_dataset_from_processed(rad_type='pyradiomics')
df_pyrad.to_parquet('Mil2/data/data/features_dataset_radpy_fixed.parquet', index=False)

print("\nCreating fmrad version...")
df_fmrad = create_feature_dataset_from_processed(rad_type='fmrad')
df_fmrad.to_parquet('Mil2/data/data/features_dataset_fmrad.parquet', index=False)

print("\nDone!")