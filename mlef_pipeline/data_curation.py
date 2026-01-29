import json
import math
from typing import Tuple

import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GroupKFold
from enums import Mode
import pandas as pd
from i3l_ml import ML
from enums import *


class DataLoader:

    data_path = {
        Mode.RWD: 'data/rwd.csv',
        Mode.DP: 'data/digital_pathology.csv',
        Mode.FMRAD: 'data/fmrad.csv',
        Mode.PYRAD: 'data/pyradiomics.csv',
        Mode.GEN: 'data/genomics.csv'
    }
    outcomes_path = 'data/outcomes.csv'


    def _get_data(self, modes: list[Mode]):
        mode_data = {}

        for mode in modes:
            if mode in self.data_path:
                path = self.data_path[mode]
                mode_data[mode] = pd.read_csv(path)
                if mode == Mode.RWD:
                    with open ('mlef_pipeline/features.json', 'r') as f:
                        selected_features = json.load(f)['RWD']
                    mode_data[mode] = mode_data[mode][['Subject', 'CENTER', 'SET'] + selected_features]
                elif mode == Mode.GEN:
                    with open ('mlef_pipeline/features.json', 'r') as f:
                        selected_features = json.load(f)['GEN']
                    mode_data[mode] = mode_data[mode][['Subject', 'CENTER', 'SET'] + selected_features]
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            
        return mode_data
    

    def _early_fusion(self, mode_data: dict[Mode, pd.DataFrame], outcome: str, is_survival: bool=False) -> pd.DataFrame:
        merged = pd.DataFrame()
        
        for data in mode_data.values():
            if merged.empty:
                merged = data
            else:
                merged = pd.merge(
                    left=merged, 
                    right=data, 
                    on='Subject', 
                    how='inner',
                    suffixes=('', '_DROP')
                )
                # Drop duplicate columns after merge
                merged = merged[[col for col in merged.columns if not col.endswith('_DROP')]]
            
        return merged
    

    def _get_subanalysis_data(self, df: pd.DataFrame, subanalysis: str, subanalysis_features: pd.DataFrame) -> pd.DataFrame:
        match subanalysis:
            case Subanalysis.C23:
                pass
            case Subanalysis.C2:
                subanalysis_features = subanalysis_features[subanalysis_features['IO LINE'] == 1]
            case Subanalysis.IO_ONLY:
                subanalysis_features = subanalysis_features[subanalysis_features['IO_CHT'] == 0]
            case Subanalysis.IO_CHT:
                subanalysis_features = subanalysis_features[subanalysis_features['IO_CHT'] == 1]
            case Subanalysis.LOW_PDL1:
                subanalysis_features = subanalysis_features[subanalysis_features['PDL1 CATEGORY'] == 0]
            case Subanalysis.MID_PDL1:
                subanalysis_features = subanalysis_features[subanalysis_features['PDL1 CATEGORY'] == 1]
            case Subanalysis.HIGH_PDL1:
                subanalysis_features = subanalysis_features[subanalysis_features['PDL1 CATEGORY'] == 2]
            case Subanalysis.SQUAMOUS:
                subanalysis_features = subanalysis_features[subanalysis_features['HISTOLOGY SQUAMOUS'] == 1]
            case Subanalysis.ADENOCARCINOMA:
                subanalysis_features = subanalysis_features[subanalysis_features['HISTOLOGY ADENOCARCINOMA'] == 1]
            case Subanalysis.INT:
                subanalysis_features = subanalysis_features[subanalysis_features['CENTER'] == 'INT']
            case _:
                raise ValueError(f"Unsupported subanalysis: {subanalysis}")
        
        return df[df['Subject'].isin(subanalysis_features['Subject'])]
    

    def create_dataset(self, modes: list[Mode], outcome: str, subanalysis: str=Subanalysis.C23, is_survival: bool=False) -> pd.DataFrame:
        mode_data = self._get_data(modes)
        dataset = self._early_fusion(mode_data, outcome, is_survival=is_survival)

        outcomes = pd.read_csv(self.outcomes_path) #add OS_6 and OS_24
        
        dataset = pd.merge(
            left=dataset,
            right=outcomes,
            on='Subject',
            how='inner'
        ).dropna(subset=[outcome])

        if Mode.RWD in modes:
            rwd = mode_data[Mode.RWD].copy()
        else: 
            rwd = pd.read_csv(self.data_path[Mode.RWD])
        
        subanalysis_features = pd.merge(rwd, outcomes, on='Subject', how='inner')
        
        dataset = self._get_subanalysis_data(dataset, subanalysis, subanalysis_features)
        
        return dataset
        
    
    def get_loco_folds(self, patient_ids: pd.Series) -> pd.Series:
        sites = ['INT', 'GHD', 'MH', 'SZMC', 'VHIO', 'UOC']

        return patient_ids.apply(lambda x: next((p for p in sites if x.startswith(p)), None))
    

    def impute_df(self, df: pd.DataFrame, imputer=None) -> Tuple[pd.DataFrame, IterativeImputer]:
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

    def normalize(self, df: pd.DataFrame, scaler: StandardScaler=None, to_standard_normalize: list[str]=None, to_log_normalize: list[str]=None) -> Tuple[pd.DataFrame, StandardScaler, list[str]]:
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
        df[to_log_normalize] = df[to_log_normalize].map(lambda x: math.log(x + 1))    
        
        if to_standard_normalize is None:
            to_standard_normalize = [col for col in features_names if col not in categorical_features]
        
        if scaler is None and len(to_standard_normalize) > 0:
            scaler = StandardScaler()
            scaler.fit(df[to_standard_normalize])

        if len(to_standard_normalize) > 0:
            df[to_standard_normalize] = scaler.transform(df[to_standard_normalize])
        df = df.rename(columns={col: f'log_{col}' for col in to_log_normalize})
        
        return df, scaler, to_standard_normalize, to_log_normalize
    

class SafeGroupKFold(GroupKFold):
    def split(self, X, y, groups):
        for train_idx, val_idx in super().split(X, y, groups):
            y_val = y.iloc[val_idx]
            if len(np.unique(y_val)) > 1:
                yield train_idx, val_idx
