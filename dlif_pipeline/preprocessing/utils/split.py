import pandas as pd
import json
import os
from pathlib import Path
from typing import Tuple

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent.parent / 'data'

os.makedirs(DATA_DIR / 'split', exist_ok=True)

rwd = pd.read_csv(DATA_DIR / 'rwd.csv')
genomics = pd.read_csv(DATA_DIR / 'genomics.csv')
digital_pathology = pd.read_csv(DATA_DIR / 'digital_pathology.csv')
pyradiomics = pd.read_csv(DATA_DIR / 'pyradiomics.csv')
fmrad = pd.read_csv(DATA_DIR / 'fmrad.csv')

with open(DATA_DIR / 'features.json', 'r') as f:
    features = json.load(f)

rwd_cols = ['Subject', 'SET', 'CENTER'] + [f for f in features['RWD'] if f in rwd.columns]
gen_cols = ['Subject', 'SET', 'CENTER'] + [f for f in features['GEN'] if f in genomics.columns]

rwd = rwd[rwd_cols]
genomics = genomics[gen_cols]

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df['SET'] == 'TRAIN'].set_index('Subject')
    test = df[df['SET'] == 'TEST'].set_index('Subject')
    ext_val = df[df['SET'] == 'EXVAL'].set_index('Subject')
    return train, test, ext_val

modalities = {
    'rwd': rwd,
    'genomics': genomics,
    'digital_pathology': digital_pathology,
    'pyradiomics': pyradiomics,
    'fmrad': fmrad
}

for name, df in modalities.items():
    train, test, ext_val = split_data(df)

    train.to_csv(DATA_DIR / 'split' / f'{name}_train.csv')
    test.to_csv(DATA_DIR / 'split' / f'{name}_test.csv')
    ext_val.to_csv(DATA_DIR / 'split' / f'{name}_ext_val.csv')

    print(f"{name} - Train: {train.shape}, Test: {test.shape}, ext_val: {ext_val.shape}")