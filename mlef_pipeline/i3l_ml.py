import json
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import compute_sample_weight
from skopt import BayesSearchCV
from i3l_statistics import Statistics
import pandas as pd
from enums import *


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
    

    def _get_model(self, model: str):
        match model:
            case Model.LR:
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(random_state=10, max_iter=1000, n_jobs=-1)
            case Model.RF:
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(random_state=10, n_jobs=-1)
            case Model.XGB:
                from xgboost import XGBClassifier
                return XGBClassifier(random_state=10, n_jobs=-1)
            case _:
                raise ValueError(f"Unsupported model: {model}")
    

    def make_weighted_auc_scorer(self, N):
        def weighted_auc(estimator, X_val, y_val):
            y_proba = estimator.predict_proba(X_val)[:, 1]
            fold_auc = roc_auc_score(y_val, y_proba)
            return fold_auc * (len(y_val) / N)
        return weighted_auc
    

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
            

    def train_model(self, X: pd.DataFrame, y: pd.Series, model_name: Model, cv: BaseCrossValidator, select_features: bool=True):
        """
        Train a machine learning model with optional feature selection and hyperparameter tuning.

        Parameters:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
        model (str): Model type ('LR', 'RF', 'XGB').
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

        if select_features:
            selected_features = self.lasso_selection(X, y, target_features=15, tolerance=10)
            X = X[selected_features]
        
        with open('classification_config.json', 'r') as f:
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
    