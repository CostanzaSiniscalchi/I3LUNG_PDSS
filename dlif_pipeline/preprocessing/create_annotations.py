import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / 'data'

def create_annotations(
    outcomes_path=None,
    rwd_path=None,
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
    if rwd_path is None:
        rwd_path = DATA_DIR / 'rwd.csv'
    if features_path is None:
        features_path = DATA_DIR / 'features_dataset_radpy_fixed.parquet'
    if output_path is None:
        output_path = DATA_DIR / 'annotations.csv'
    
    # Load data
    outcomes = pd.read_csv(outcomes_path)
    rwd = pd.read_csv(rwd_path)
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
    
    # Filter only RWD subjects
    rwd_subjects = set(rwd['Subject'])
    ann = outcomes[outcomes['Subject'].isin(rwd_subjects)].copy()
    
    # Rename Subject to slide and patient
    ann = ann.rename(columns={'Subject': 'slide'})
    ann['patient'] = ann['slide']
    
    # Get FOLD and dataset from RWD
    rwd_info = rwd[['Subject', 'CENTER', 'SET']].copy()
    ann = ann.merge(rwd_info, left_on='slide', right_on='Subject', how='left')
    ann = ann.rename(columns={'CENTER': 'FOLD', 'SET': 'dataset'})
    ann.drop(columns=['Subject'], inplace=True)
    
    # Map dataset values
    ann['dataset'] = ann['dataset'].map({
        'TRAIN': 'train',
        'TEST': 'test',
        'EXVAL': 'ext_val'
    })
    
    # Get outcome columns
    outcome_cols = [c for c in ann.columns if c not in ['slide', 'patient', 'FOLD', 'dataset']]
    
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
    
    # Merge additional flags from RWD
    rwd_flags = rwd[['Subject', 'PDL1 CATEGORY', 'HISTOLOGY ADENOCARCINOMA', 'HISTOLOGY SQUAMOUS', 'IO IOCT', 'IO LINE']].copy()
    rwd_flags = rwd_flags.rename(columns={
        'HISTOLOGY ADENOCARCINOMA': 'NSCLC_HISTOLOGY_ADENOCARCINOMA',
        'HISTOLOGY SQUAMOUS': 'NSCLC_HISTOLOGY_SQUAMOUS',
        'IO IOCT': 'IO_IOCT',
        'PDL1 CATEGORY': 'PDL1_CATEGORY',
        'IO LINE': 'IO_LINE'
    })
    
    ann = ann.merge(rwd_flags, left_on='patient', right_on='Subject', how='left')
    ann.drop(columns=['Subject'], inplace=True)
    
    # Map PDL1 to low/high
    ann['PDL1_GROUP'] = ann['PDL1_CATEGORY'].map({
        0: 'low',
        1: 'low',
        2: 'high'
    }).fillna('')
    
    ann.drop(columns=['PDL1_CATEGORY'], inplace=True)
    
    # Fill NaN in flags with empty string
    ann['NSCLC_HISTOLOGY_SQUAMOUS'] = ann['NSCLC_HISTOLOGY_SQUAMOUS'].fillna('')
    ann['NSCLC_HISTOLOGY_ADENOCARCINOMA'] = ann['NSCLC_HISTOLOGY_ADENOCARCINOMA'].fillna('')
    ann['IO_IOCT'] = ann['IO_IOCT'].fillna('')
    
    # COHORT_2 flag: IO LINE == 1
    ann['COHORT_2'] = (ann['IO_LINE'] == 1).astype(int)
    ann.drop(columns=['IO_LINE'], inplace=True)
    
    # HAS_ALL_MODALITIES flag
    features['HAS_ALL_MODALITIES'] = (
        features['mod1'].notna() & 
        features['mod2'].notna() & 
        features['mod3'].notna() & 
        features['mod4'].notna()
    ).astype(int)
    
    all_mods_dict = dict(zip(features['Subject'], features['HAS_ALL_MODALITIES']))
    ann['HAS_ALL_MODALITIES'] = ann['slide'].map(all_mods_dict).fillna(0).astype(int)
    
    print(f"Patients with all modalities: {ann['HAS_ALL_MODALITIES'].sum()}")
    
    # Add early stopping
    np.random.seed(seed)
    train_df = ann[ann['dataset'] == 'train']
    
    total_n = int(len(train_df) * val_split)
    early_stop_indices = train_df.sample(n=total_n, random_state=seed).index.tolist()
    print(f"Generated {total_n} early stopping subjects randomly")

    allowed_early_outcomes = [
        'ORR', 'CBR', 'PFS_MONTHS', 'OS_MONTHS', 'TTF_MONTHS',
        'DEATH_EVENT_OC', 'PROGRESSION_EVENT_OC', 'THERAPY_END_EVENT_OC',
        'BEST_RESPONSE', 'DCR', 'os_months_6', 'os_months_24'
    ]

    for outcome in outcome_cols:
        if outcome not in allowed_early_outcomes:
            continue
            
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
        
    # Reorder columns
    base_cols = ['slide', 'FOLD', 'dataset'] + outcome_cols + ['patient']
    
    split_cols = []
    for outcome in outcome_cols:
        split_cols.extend([f'dataset_{outcome}', f'fold_{outcome}'])
    
    flag_cols = ['PDL1_GROUP', 'NSCLC_HISTOLOGY_ADENOCARCINOMA', 'NSCLC_HISTOLOGY_SQUAMOUS', 'IO_IOCT', 'COHORT_2', 'HAS_ALL_MODALITIES']
    
    early_cols = [f'early_stopping_{o}' for o in allowed_early_outcomes if o in outcome_cols]
    
    final_cols = base_cols + split_cols + flag_cols + early_cols
    
    ann = ann[final_cols]
    
    # Save
    ann.to_csv(output_path, index=False)
    print(f"Saved annotations to {output_path}")
    print(f"Shape: {ann.shape}")
    
    return ann

# Run
ann = create_annotations(use_subfolds=False)