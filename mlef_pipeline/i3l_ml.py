import json
from lifelines import CoxPHFitter
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import BaseCrossValidator, KFold, GridSearchCV, cross_val_score
from sklearn.utils import compute_sample_weight
from skopt import BayesSearchCV
from i3l_statistics import Statistics
import pandas as pd
from enums import *
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='skopt')


class ML:

    @staticmethod
    def lasso_selection(X: pd.DataFrame, y: pd.Series, target_features: int=10, tolerance: int=5, max_iter=100):
        left = 1e-10
        right = 1
        stats = Statistics()
        iteration = 0
        
        # select more features than required
        while iteration < max_iter:
            iteration += 1
            
            current_alpha = (left + right) / 2
            
            selector = SelectFromModel(
                estimator=Lasso(
                    alpha=current_alpha,
                    random_state=10
                ),
                prefit=False
            )
            selector.fit(X, y)
            
            n_features = sum(selector.get_support())
            
            if target_features < n_features < target_features + tolerance:
                break
                
            if n_features > target_features:
                left = current_alpha
            else:
                right = current_alpha
            
            if right - left < 1e-10:
                break
        
        # filter with elbow method
        selected_features = X.columns[selector.get_support()].tolist()
        selected_coefficients = selector.estimator_.coef_[selector.get_support()]
        feature_importance = pd.DataFrame({'FEATURE': selected_features, 'COEFFICIENT': selected_coefficients})
        selected_features = stats.elbow_selection(feature_importance, target_features - tolerance, target_features + tolerance)
        
        selected_features = selected_features['FEATURE'].to_list()
        
        return selected_features
    

    @staticmethod
    def coxnet_selection(X_train: pd.DataFrame, y_train: pd.DataFrame, cv, folds=None, target_features=15, uncertainty=5):
        if X_train.shape[1] < target_features - uncertainty:
            return X_train.columns
        
        if isinstance(cv, int):
            kf = KFold(n_splits=cv, shuffle=True, random_state=10)
            cv = lambda: kf.split(X_train, y_train)
        
        statistics = Statistics()
        config = json.load(open('mlef_pipeline/survival_config.json'))
        
        l1_ratio = 0.1
        alpha_min = 1e-4
        alpha_max = 1e0
        max_iter = 20
        
        cox_params = config['param_grid_COXNET']
        
        for _ in range(max_iter):
            alpha = (alpha_min + alpha_max) / 2
            
            cox_params.update({
                'alphas': [[alpha]],
                'l1_ratio': [l1_ratio]
            })
            
            gcv = ML.grid_search(X_train, y_train, CoxnetSurvivalAnalysis(), cox_params, cv(), folds)
            best_model = gcv.best_estimator_
            
            coefficients = [el for coef_list in best_model.coef_ for el in coef_list]
            feature_df = pd.DataFrame({
                'FEATURE': X_train.columns,
                'COEFFICIENT': coefficients
            })
            feature_df = feature_df[feature_df['COEFFICIENT'] != 0]
            n_features = len(feature_df)
            
            if target_features < n_features < target_features + uncertainty:
                break
            
            if n_features < target_features:
                alpha_max = alpha
            
            if n_features > target_features + uncertainty:
                alpha_min = alpha
        
        selected_features = statistics.elbow_selection(feature_df, target_features - uncertainty, target_features + uncertainty)
        
        return list(selected_features['FEATURE'].values)
    

    def _get_model(self, model: str):
        match model:
            case Model.LR:
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(random_state=10, max_iter=1000, n_jobs=-1)
            case Model.RF:
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(random_state=10, n_jobs=-1)
            case _:
                raise ValueError(f"Unsupported model: {model}")
    

    def make_weighted_auc_scorer(self, N):
        def weighted_auc(estimator, X_val, y_val):
            y_proba = estimator.predict_proba(X_val)[:, 1]
            fold_auc = roc_auc_score(y_val, y_proba)
            return fold_auc * (len(y_val) / N)
        return weighted_auc
    

    @staticmethod
    def get_weighted_cv(model, X, y, cv_getter, scorer, sample_weight=None):
        """
        Compute weighted cross-validation score with proper handling of variable fold sizes.
        
        Args:
            model: The model to evaluate
            X: Features
            y: Target
            cv_getter: Function that returns CV splits
            scorer: Scoring function or string
            sample_weight: Optional sample weights
        
        Returns:
            score_mean: Weighted mean score
            score_std: Weighted standard deviation
        """
        fold_sizes = np.array([len(test_idx) for _, test_idx in cv_getter()])
        
        score = cross_val_score(model, X, y, scoring=scorer, cv=cv_getter(), params={'sample_weight': sample_weight})
        score = np.array(score)
        valid = ~np.isnan(score)
        score_valid = score[valid]
        weights_valid = fold_sizes[valid]
        score_mean = np.average(score_valid, weights=weights_valid)
        var = np.average((score_valid - score_mean)**2, weights=weights_valid)
        score_std = np.sqrt(var)
        
        return score_mean, score_std

    def hyperparameter_tuning(self, model, param_grid, cv, X_train, y_train, sample_weight):
        n_iter = 50
        opt = BayesSearchCV(
            estimator=model, 
            search_spaces=param_grid, 
            scoring=self.make_weighted_auc_scorer(len(y_train)), 
            n_iter=n_iter, 
            refit=True, 
            cv=cv(), 
            error_score=0.5, 
            random_state=10
        )
        np.int = int

        opt.fit(X_train, y_train, sample_weight=sample_weight)

        return opt.best_estimator_
            

    def train_classification_model(self, X: pd.DataFrame, y: pd.Series, model_name: Model, cv: BaseCrossValidator, select_features: bool=True):
        """
        Train a machine learning model with optional feature selection and hyperparameter tuning.

        Parameters:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
        model (str): Model type ('LR', 'RF').
        cv: Cross-validation strategy.
        select_features (bool): Whether to perform feature selection using Lasso.
        Returns:
        The trained model.
        """

        clf = self._get_model(model_name)
        sample_weights = pd.Series(
            compute_sample_weight(class_weight='balanced', y=y), 
            index=y.index
        )

        fm_rad_features = [col for col in X.columns if col.startswith('pred_') or col.startswith('log_pred_')]
        if len(fm_rad_features) > 0:
            selected_fm_rad_features = self.lasso_selection(
                X[fm_rad_features], 
                y,
                target_features=100,
                tolerance=10
            )
            X = X.drop(columns=[col for col in fm_rad_features if col not in selected_fm_rad_features])

        if select_features:
            selected_features = self.lasso_selection(X, y, target_features=15, tolerance=10)
            X = X[selected_features]
            print(selected_features)
        with open('mlef_pipeline/classification_config.json', 'r') as f:
            param_grids = json.load(f)
        
        best_clf = self.hyperparameter_tuning(
            model=clf,
            param_grid=param_grids[f'param_grid_{model_name.value}'],
            cv=cv,
            X_train=X,
            y_train=y,
            sample_weight=sample_weights
        )

        return best_clf
    

    def train_survival_model(self, dataset: pd.DataFrame, model_name: Model, cv: BaseCrossValidator, select_features: bool=True):
        """
        Train a survival analysis model with optional feature selection and hyperparameter tuning.

        Parameters:
        X (pd.DataFrame): Feature set.
        y (pd.DataFrame): Target variable with survival time and event indicator.
        model (str): Model type.
        cv: Cross-validation strategy.
        select_features (bool): Whether to perform feature selection using Lasso.
        Returns:
        The trained model.
        """

        match model_name:
            case Model.COX:
                cph = CoxPHFitter(penalizer=0.5)
            case _:
                raise ValueError(f"Unsupported survival model: {model_name}")

        if select_features:
            selected_features = self.lasso_selection(X, y['OS MONTHS'], target_features=15, tolerance=10)
            X = X[selected_features]

        cph.fit(dataset, duration_col='TIME', event_col='EVENT')

        return cph
    

    @staticmethod
    def make_weighted_cindex_scorer(N):
        
        def weighted_cindex(estimator, X_val, y_val):
            events = y_val["EVENT"]
            times = y_val["TIME"]
            n = len(events)
            
            if n < 2 or events.sum() == 0 or np.unique(times).size < 2:
                return 0.5
            
            y_pred = estimator.predict(X_val)
            fold_cindex = concordance_index_censored(y_val['EVENT'], y_val['TIME'], y_pred)[0]
            return fold_cindex * (len(y_val) / N)
        
        return weighted_cindex


    @staticmethod
    def grid_search(X, y, model, params, cv, folds=None):
        return GridSearchCV(
            model,
            param_grid=params,
            cv=cv,
            error_score=0.5,
            n_jobs=4,
            verbose=0,
            scoring=ML.make_weighted_cindex_scorer(X.shape[0])
        ).fit(X, y, groups=folds)
    