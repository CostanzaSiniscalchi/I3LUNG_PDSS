import pandas as pd
import json
import os
from typing import Tuple

os.makedirs('Mil2/data/data/split', exist_ok=True)

with open('Mil2/data/data/split.json', 'r') as f:
    split = json.load(f)

train_subjects = split['train']
test_subjects = split['test']
uoc_subjects = split['ext_val_set']

rwd = pd.read_csv('Mil2/data/data/rwd.csv')
genomics = pd.read_csv('Mil2/data/data/genomics.csv')
digital_pathology = pd.read_csv('Mil2/data/data/digital_pathology.csv')
pyradiomics = pd.read_csv('Mil2/data/data/pyradiomics.csv')
fmrad = pd.read_csv('Mil2/data/data/fmrad.csv')

with open('Mil2/data/data/features.json', 'r') as f:
    features = json.load(f)

rwd_cols = ['Subject'] + [f for f in features['RWD'] if f in rwd.columns]
gen_cols = ['Subject'] + [f for f in features['GEN'] if f in genomics.columns]

rwd = rwd[rwd_cols]
genomics = genomics[gen_cols]

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df['Subject'].isin(train_subjects)].set_index('Subject')
    test = df[df['Subject'].isin(test_subjects)].set_index('Subject')
    uoc = df[df['Subject'].isin(uoc_subjects)].set_index('Subject')
    return train, test, uoc

modalities = {
    'rwd': rwd,
    'genomics': genomics,
    'digital_pathology': digital_pathology,
    'pyradiomics': pyradiomics,
    'fmrad': fmrad
}

for name, df in modalities.items():
    train, test, uoc = split_data(df)
    
    train.to_csv(f'Mil2/data/data/split/{name}_train.csv')
    test.to_csv(f'Mil2/data/data/split/{name}_test.csv')
    uoc.to_csv(f'Mil2/data/data/split/{name}_ext_val_set.csv')
    
    print(f"{name} - Train: {train.shape}, Test: {test.shape}, ext_val: {uoc.shape}")