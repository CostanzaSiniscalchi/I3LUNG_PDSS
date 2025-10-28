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
        Mode.RWD: '../data/rwd.csv',
        Mode.DP: '../data/digital_pathology.csv',
        Mode.FMRAD: '../data/fmrad.csv',
        Mode.PYRAD: '../data/pyradiomics.csv',
        Mode.GEN: '../data/genomics.csv'
    }
    outcomes_path = '../data/outcomes.csv'


    def _get_data(self, modes: list[Mode]):
        mode_data = {}

        for mode in modes:
            if mode in self.data_path:
                path = self.data_path[mode]
                mode_data[mode] = pd.read_csv(path)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            
        return mode_data
    

    def _early_fusion(self, mode_data: dict[Mode, pd.DataFrame], outcome: str) -> pd.DataFrame:
        merged = pd.DataFrame()

        outcomes = pd.read_csv(self.outcomes_path)

        if Mode.FMRAD in mode_data.keys():
            # reduce the number of fm_rad features with lasso
            print("Selecting FM-RAD features with LASSO...")
            fmrad = pd.merge(
                left=mode_data[Mode.FMRAD],
                right=outcomes[['Subject', outcome]],
                on='Subject',
                how='inner'
            ).dropna(subset=[outcome])

            selected_fmrad = ML.lasso_selection(
                X=fmrad.drop(columns=['Subject', outcome]),
                y=fmrad[outcome],
                target_features=100,
                tolerance=10,
            )
            mode_data[Mode.FMRAD] = mode_data[Mode.FMRAD][['Subject'] + selected_fmrad]

        for data in mode_data.values():
            if merged.empty:
                merged = data
            else:
                merged = pd.merge(
                    left=merged, 
                    right=data, 
                    on='Subject', 
                    how='inner'
                )
            
        return merged
    

    def _get_subanalysis_data(self, df: pd.DataFrame, cohort: int, subanalysis: str, subanalysis_features: pd.DataFrame) -> pd.DataFrame:
        match subanalysis:
            case Subanalysis.CLASSIC:
                pass
            case Subanalysis.IO_ONLY:
                subanalysis_features = subanalysis_features[subanalysis_features['IO IOCT'] == 0]
            case Subanalysis.IO_CT:
                subanalysis_features = subanalysis_features[subanalysis_features['IO IOCT'] == 1]
            case Subanalysis.LOW_PDL1:
                subanalysis_features = subanalysis_features[(subanalysis_features['PDL1 CATEGORY'] == 0) | (subanalysis_features['PDL1 CATEGORY'] == 1)]
            case Subanalysis.HIGH_PDL1:
                subanalysis_features = subanalysis_features[subanalysis_features['PDL1 CATEGORY'] == 2]
            case Subanalysis.SQUAMOUS:
                subanalysis_features = subanalysis_features[subanalysis_features['HISTOLOGY SQUAMOUS'] == 1]
            case Subanalysis.ADENOCARCINOMA:
                subanalysis_features = subanalysis_features[subanalysis_features['HISTOLOGY ADENOCARCINOMA'] == 1]
            case _:
                raise ValueError(f"Unsupported subanalysis: {subanalysis}")
        
        if cohort == 2:
            subanalysis_features = subanalysis_features[subanalysis_features['IO LINE'] == 2]
        
        return df[df['Subject'].isin(subanalysis_features['Subject'])]
    

    def create_dataset(self, modes: list[Mode], outcome: str, cohort: int=23, subanalysis: str=Subanalysis.CLASSIC) -> pd.DataFrame:
        mode_data = self._get_data(modes)
        dataset = self._early_fusion(mode_data, outcome)

        outcomes = pd.read_csv(self.outcomes_path)
        
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
        
        dataset = self._get_subanalysis_data(dataset, cohort, subanalysis, rwd)
        
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
        print(f'{len(to_log_normalize)} features log normalized')
        print(f'{len(to_standard_normalize)} features standardized')
        
        return df, scaler, to_standard_normalize, to_log_normalize
    

    def get_outcome(self, outcome: pd.Series, outcome_name: Outcome):
        match outcome_name:
            case Outcome.OS_6:
                outcome = (outcome >= 6).astype(int)
            case Outcome.OS_24:
                outcome = (outcome >= 24).astype(int)
            case Outcome.OS_SURV:
                outcome = outcome  # Keep as is for survival analysis
            case _:
                raise ValueError(f"Unsupported outcome: {outcome_name}")

        return outcome
    

class SafeGroupKFold(GroupKFold):
    def split(self, X, y, groups):
        for train_idx, val_idx in super().split(X, y, groups):
            y_val = y[val_idx]
            if len(np.unique(y_val)) > 1:
                yield train_idx, val_idx
