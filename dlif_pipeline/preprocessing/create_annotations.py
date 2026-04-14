import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / 'data'

def _to_str_flag(series):
    """Convert a binary flag column (int/float/mixed) to clean '0'/'1'/'' strings."""
    def _convert(x):
        if pd.isna(x) or x == '':
            return ''
        return str(int(float(x)))
    return series.apply(_convert)


def create_annotations(
    outcomes_path=None,
    cb_path=None,
    features_path=None,
    output_path=None,
    n_sub=5,
    use_subfolds=False,
    val_split=0.10,
    seed=42
):
    # Set default paths if not provided
    if outcomes_path is None:
        outcomes_path = DATA_DIR / 'outcomes.csv'
    if cb_path is None:
        cb_path = DATA_DIR / 'rwd.csv'
    if features_path is None:
        features_path = DATA_DIR / 'features_dataset_radpy_fixed.parquet'
    if output_path is None:
        output_path = DATA_DIR / 'annotations.csv'
    
    # Load data
    outcomes = pd.read_csv(outcomes_path)
    cb = pd.read_csv(cb_path)
    features = pd.read_parquet(features_path)
    
    # Rename outcomes columns
    outcomes = outcomes.rename(columns={
        'OS_6': 'os_months_6',
        'OS_24': 'os_months_24',
        'DEATH EVENT': 'DEATH_EVENT_OC',
        'PROGRESSION EVENT': 'PROGRESSION_EVENT_OC',
        'THERAPY END EVENT': 'THERAPY_END_EVENT_OC',
        'BEST RESPONSE': 'BEST_RESPONSE',
        'PFS MONTHS': 'PFS_MONTHS',
        'OS MONTHS': 'OS_MONTHS',
        'TTF MONTHS': 'TTF_MONTHS',
    })
    
    # Filter only CB subjects
    cb_subjects = set(cb['Subject'])
    ann = outcomes[outcomes['Subject'].isin(cb_subjects)].copy()
    
    
    # Get FOLD and dataset from CB
    cb_info = cb[['Subject', 'CENTER', 'SET']].copy()
    ann = ann.merge(cb_info, on='Subject', how='left')  # merge su Subject direttamente
    ann = ann.rename(columns={'CENTER': 'FOLD', 'SET': 'dataset'})
    
    # Map dataset values
    ann['dataset'] = ann['dataset'].map({
        'TRAIN': 'train',
        'TEST': 'test',
        'EXVAL': 'ext_val'
    })
    
    # Get outcome columns
    outcome_cols = [c for c in ann.columns if c not in ['Subject', 'FOLD', 'dataset']]

    # Normalize binary outcome columns to string '0'/'1' (keep NaN as NaN)
    binary_outcomes = ['os_months_6', 'os_months_24', 'DCR', 'ORR', 'CBR']
    for col in binary_outcomes:
        if col in ann.columns:
            ann[col] = ann[col].apply(lambda x: str(int(float(x))) if pd.notna(x) else np.nan)

    # Create dataset_ and fold_ for each outcome
    for outcome in outcome_cols:
        ann[f'dataset_{outcome}'] = np.where(ann[outcome].notna(), ann['dataset'], None)
        ann[f'fold_{outcome}'] = np.where(ann[outcome].notna(), ann['FOLD'], None)
    
    # Create subfolds (overwrites fold_ columns) - OPTIONAL
    if use_subfolds:
        print("Creating subfolds...")
        rng = np.random.default_rng(seed)
        for outcome in outcome_cols:
            fold_col = f'fold_{outcome}'
            mask = ann[fold_col].notna()
            
            for center, idxs in ann.loc[mask].groupby(fold_col).groups.items():
                ids = list(idxs)
                rng.shuffle(ids)
                for i, split_ids in enumerate(np.array_split(ids, n_sub), start=1):
                    ann.loc[split_ids, fold_col] = f"{center}_sub{i}"
        print(f"Created {n_sub} subfolds per center")
    
    # Merge additional flags from CB - only use columns that exist
    flag_cols = ['Subject', 'PDL1 CATEGORY', 'HISTOLOGY ADENOCARCINOMA', 'HISTOLOGY SQUAMOUS', 'IO LINE', 'IO IOCHT']
    available_flag_cols = [col for col in flag_cols if col in cb.columns]
    
    cb_flags = cb[available_flag_cols].copy()
    
    # Rename columns
    rename_map = {
        'HISTOLOGY ADENOCARCINOMA': 'NSCLC_HISTOLOGY_ADENOCARCINOMA',
        'HISTOLOGY SQUAMOUS': 'NSCLC_HISTOLOGY_SQUAMOUS',
        'PDL1 CATEGORY': 'PDL1_CATEGORY',
        'IO LINE': 'IO_LINE',
        'IO IOCHT': 'IO_CHT',
    }
    
    cb_flags = cb_flags.rename(columns={k: v for k, v in rename_map.items() if k in cb_flags.columns})
    
    ann = ann.merge(cb_flags, on='Subject', how='left')  # merge su Subject
    
    # Map PDL1 to low/high if column exists
    if 'PDL1_CATEGORY' in ann.columns:
        ann['PDL1_GROUP'] = ann['PDL1_CATEGORY'].map({
            0: 'low',
            1: 'low',
            2: 'high'
        }).fillna('')
        ann['PDL1_CATEGORY'] = ann['PDL1_CATEGORY'].astype('Int64').astype('string').replace('<NA>', '')
    else:
        ann['PDL1_GROUP'] = ''
    
    # Normalize binary flag columns to clean string '0'/'1'/''
    if 'NSCLC_HISTOLOGY_SQUAMOUS' in ann.columns:
        ann['NSCLC_HISTOLOGY_SQUAMOUS'] = _to_str_flag(ann['NSCLC_HISTOLOGY_SQUAMOUS'])
    else:
        ann['NSCLC_HISTOLOGY_SQUAMOUS'] = ''

    if 'NSCLC_HISTOLOGY_ADENOCARCINOMA' in ann.columns:
        ann['NSCLC_HISTOLOGY_ADENOCARCINOMA'] = _to_str_flag(ann['NSCLC_HISTOLOGY_ADENOCARCINOMA'])
    else:
        ann['NSCLC_HISTOLOGY_ADENOCARCINOMA'] = ''

    if 'IO_CHT' not in ann.columns:
        ann['IO_CHT'] = ''
    else:
        ann['IO_CHT'] = _to_str_flag(ann['IO_CHT'])
    
    # COHORT_2 flag: IO LINE == 1 if column exists
    if 'IO_LINE' in ann.columns:
        ann['COHORT_2'] = (ann['IO_LINE'] == 1).astype(int).astype(str)
        ann.drop(columns=['IO_LINE'], inplace=True)
    else:
        ann['COHORT_2'] = '0'

    # HAS_{mod} flags based on which subjects appear in each raw data file
    modality_files = {
        'HAS_CB': DATA_DIR / 'cb.csv',
        'HAS_RADPY': DATA_DIR / 'pyradiomics.csv',
        'HAS_FMRAD': DATA_DIR / 'fmrad.csv',
        'HAS_DP': DATA_DIR / 'digital_pathology.csv',
        'HAS_GENOMICS': DATA_DIR / 'genomics.csv',
    }
    for col_name, filepath in modality_files.items():
        if filepath.exists():
            subjects = set(pd.read_csv(filepath, usecols=['Subject'])['Subject'])
            ann[col_name] = ann['Subject'].isin(subjects).astype(int).astype(str)
            print(f"{col_name}: {(ann[col_name] == '1').sum()} patients")
        else:
            print(f"WARNING: {filepath} not found, setting {col_name} = 0")
            ann[col_name] = '0'

    # Add early stopping for ALL outcomes
    np.random.seed(seed)
    train_df = ann[ann['dataset'] == 'train']
    
    total_n = int(len(train_df) * val_split)
    early_stop_indices = train_df.sample(n=total_n, random_state=seed).index.tolist()
    print(f"Generated {total_n} early stopping subjects randomly")

    for outcome in outcome_cols:
        fold_col = f'fold_{outcome}'
        early_stop_col = f'early_stopping_{outcome}'
        
        if fold_col not in ann.columns:
            print(f"Skipping {outcome}: missing column '{fold_col}'")
            continue
        
        ann[early_stop_col] = "no"
        
        for idx in early_stop_indices:
            if idx in ann.index and pd.notna(ann.at[idx, fold_col]):
                ann.at[idx, early_stop_col] = "yes"
        
        print(f"Added early stopping for {outcome}")

    # Add folds for INT-only
    with open('data/int_cv_splits.json', 'r') as f:
        int_only = json.load(f)
    for fold_idx, slide_list in int_only.items():
        ann.loc[ann['Subject'].isin(slide_list), 'INT_ONLY_FOLDS'] = fold_idx

    # Rinomina Subject in slide
    ann = ann.rename(columns={'Subject': 'patient'})
    ann['slide'] = ann['patient']

    base_cols = ['slide', 'FOLD', 'dataset'] + outcome_cols + ['patient']
    
    
    split_cols = []
    for outcome in outcome_cols:
        split_cols.extend([f'dataset_{outcome}', f'fold_{outcome}'])
    
    flag_cols = ['PDL1_GROUP', 'PDL1_CATEGORY', 'NSCLC_HISTOLOGY_ADENOCARCINOMA', 'NSCLC_HISTOLOGY_SQUAMOUS', 'IO_CHT', 'COHORT_2', 'HAS_CB', 'HAS_RADPY', 'HAS_FMRAD', 'HAS_DP', 'HAS_GENOMICS', 'INT_ONLY_FOLDS']
    
    early_cols = [f'early_stopping_{o}' for o in outcome_cols]
    
    final_cols = base_cols + split_cols + flag_cols + early_cols
    
    ann = ann[final_cols]
    
    # Save
    ann.to_csv(output_path, index=False)
    print(f"Saved annotations to {output_path}")
    print(f"Shape: {ann.shape}")
    
    return ann

# Run
ann = create_annotations(use_subfolds=False)