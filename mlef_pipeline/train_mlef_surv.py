"""
train_mlef_surv.py

Script version of the survival analysis workflow from the notebook `survival.ipynb`.

This is a minimal direct port of the notebook logic so you can run the survival pipeline
from a Python script. It mirrors the order and operations used in the notebook:
 - load dataset and split using `split.json`
 - drop test samples with TIME > max train TIME (same heuristic as notebook)
 - impute and normalize
 - feature selection with `ml.coxnet_selection`
 - fit CoxPH model and compute c-index on CV, train, test, ext

Usage:
    python train_mlef_surv.py

If you need command-line options, extend the argparse part at the bottom.
"""

from __future__ import annotations

import json
import argparse
import numpy as np
import pandas as pd

from data_curation import DataLoader  # local module (same as notebook)
from i3l_statistics import Statistics
from i3l_ml import ML
from enums import Mode, Subanalysis
from lifelines import CoxPHFitter
from sklearn.model_selection import GroupKFold


def get_scores(values: dict) -> str:
    return f"{values['c_index']:.2f} ± {(values['ci'][1] - values['ci'][0]) / 2 :.2f}"


def coxph_cv(train_set: pd.DataFrame, cv_getter: callable) -> np.ndarray:
    """Run cross-validated CoxPH and return per-sample predicted risk (NaN where not predicted).

    Parameters
    ----------
    train_set : pd.DataFrame
        DataFrame containing features + 'TIME' and 'EVENT' columns.
    cv_getter : callable
        Callable that returns an iterable of (train_index, test_index) splits (indices into train_set).
    """
    pred_risk = np.empty(train_set.shape[0])
    pred_risk[:] = np.nan

    for index_train, index_test in cv_getter():
        # skip tiny folds
        if len(index_test) < 3:
            continue

        X_train_fold = train_set.iloc[index_train]
        X_val_fold = train_set.iloc[index_test]

        cph = CoxPHFitter(penalizer=0.5)
        cph.fit(X_train_fold, duration_col='TIME', event_col='EVENT')

        # lifelines returns a pandas Series; assign into numpy array
        pred_risk[index_test] = cph.predict_partial_hazard(X_val_fold)

    return pred_risk


def main(args: argparse.Namespace) -> None:
    dl = DataLoader()
    ml = ML()
    stats = Statistics()

    # create dataset (mirrors notebook call)
    dataset = dl.create_dataset(
        modes=args.modes,
        outcome=args.outcome,
        subanalysis=args.subanalysis,
        is_survival=True,
    )

    # load split
    with open('mlef_pipeline/split.json', 'r') as f:
        split = json.load(f)

    train_set = dataset[dataset['Subject'].isin(split['TRAIN_SET'])].set_index('Subject')
    test_set = dataset[dataset['Subject'].isin(split['TEST_SET'])].set_index('Subject')
    ext_set = dataset[dataset['Subject'].str.startswith('UOC')].set_index('Subject')

    # ensure test max TIME <= train max TIME by dropping worst test samples
    removed = 0
    while True:
        max_train_time = train_set[args.outcome].max()
        max_test_time = test_set[args.outcome].max()
        if max_test_time <= max_train_time:
            break
        removed += 1
        max_test_index = test_set[args.outcome].idxmax()
        test_set = test_set.drop(max_test_index)

    # prepare X/y and rename columns to TIME/EVENT like in notebook
    X_train = train_set.drop(columns=[args.outcome, 'DEATH EVENT'])
    y_train = train_set[[args.outcome, 'DEATH EVENT']]
    X_test = test_set.drop(columns=[args.outcome, 'DEATH EVENT'])
    y_test = test_set[[args.outcome, 'DEATH EVENT']]
    X_ext = ext_set.drop(columns=[args.outcome, 'DEATH EVENT'])
    y_ext = ext_set[[args.outcome, 'DEATH EVENT']]

    y_train = y_train.rename(columns={args.outcome: 'TIME', 'DEATH EVENT': 'EVENT'})
    y_test = y_test.rename(columns={args.outcome: 'TIME', 'DEATH EVENT': 'EVENT'})
    y_ext = y_ext.rename(columns={args.outcome: 'TIME', 'DEATH EVENT': 'EVENT'})

    # drop submodel features if present (same as notebook)
    with open('mlef_pipeline/submodel_features.json', 'r') as f:
        submodel_features = json.load(f)
        submodel_features = [f for f in submodel_features if f in X_train.columns]

    if len(submodel_features) > 0:
        X_train = X_train.drop(columns=submodel_features)
        X_test = X_test.drop(columns=submodel_features)
        X_ext = X_ext.drop(columns=submodel_features)

    # impute
    X_train_imputed, imputer = dl.impute_df(X_train)
    X_test_imputed, _ = dl.impute_df(X_test, imputer=imputer)
    X_ext_imputed, _ = dl.impute_df(X_ext, imputer=imputer)

    # normalize
    X_train_scaled, scaler, to_standard_normalize, to_log_normalize = dl.normalize(X_train_imputed)
    X_test_scaled, _, _, _ = dl.normalize(X_test_imputed, scaler=scaler, to_standard_normalize=to_standard_normalize, to_log_normalize=to_log_normalize)
    X_ext_scaled, _, _, _ = dl.normalize(X_ext_imputed, scaler=scaler, to_standard_normalize=to_standard_normalize, to_log_normalize=to_log_normalize)

    # train folds using same loco strategy as notebook
    train_folds = dl.get_loco_folds(pd.Series(train_set.index))
    cv = GroupKFold(n_splits=len(train_folds.unique()))

    def cv_getter():
        return cv.split(X_train_scaled, y_train, groups=train_folds)

    # ensure EVENT dtype
    y_train['EVENT'] = y_train['EVENT'].astype(bool)
    y_train = y_train[['EVENT', 'TIME']]

    # feature selection (coxnet)
    features = ml.coxnet_selection(
        X_train=X_train_scaled,
        y_train=y_train.to_records(index=False),
        cv=cv_getter,
        folds=train_folds,
        target_features=args.target_features,
        uncertainty=args.uncertainty,
    )
    print(f'Selected {len(features)} features: {features}')

    # select features
    X_train_sel = X_train_scaled[features]
    X_test_sel = X_test_scaled[features]
    X_ext_sel = X_ext_scaled[features]

    # build combined tables for lifelines
    train_tbl = pd.concat([X_train_sel, pd.DataFrame(y_train, index=X_train_sel.index)], axis=1)
    test_tbl = pd.concat([X_test_sel, pd.DataFrame(y_test, index=X_test_sel.index)], axis=1)
    ext_tbl = pd.concat([X_ext_sel, pd.DataFrame(y_ext, index=X_ext_sel.index)], axis=1)

    # fit CoxPH
    cph = CoxPHFitter(penalizer=0.5)
    cph.fit(train_tbl, duration_col='TIME', event_col='EVENT')

    # risk scores and c-index
    risk_scores_train = cph.predict_log_partial_hazard(train_tbl).values
    cindex_train = stats.compute_c_index_and_ci(
        time=train_tbl['TIME'],
        event=train_tbl['EVENT'],
        risk_score=risk_scores_train,
    )

    risk_scores_test = cph.predict_log_partial_hazard(test_tbl).values
    cindex_test = stats.compute_c_index_and_ci(
        time=test_tbl['TIME'],
        event=test_tbl['EVENT'],
        risk_score=risk_scores_test,
    )

    risk_scores_ext = cph.predict_log_partial_hazard(ext_tbl).values
    cindex_ext = stats.compute_c_index_and_ci(
        time=ext_tbl['TIME'],
        event=ext_tbl['EVENT'],
        risk_score=risk_scores_ext,
    )

    # cross-validated risks
    cv_risks = coxph_cv(
        train_set=pd.concat([X_train_sel, pd.DataFrame(y_train, index=X_train_sel.index)], axis=1),
        cv_getter=cv_getter,
    )

    cindex_cv = stats.compute_c_index_and_ci(
        time=train_tbl['TIME'],
        event=train_tbl['EVENT'],
        risk_score=cv_risks,
    )

    # print results similar to notebook
    print(
        'Train shape', train_tbl.shape[0], '\n',
        'Test shape', test_tbl.shape[0], '\n',
        'Ext shape', ext_tbl.shape[0], '\n',
        'CV', get_scores(cindex_cv), '\n',
        'Train', get_scores(cindex_train), '\n',
        'Test', get_scores(cindex_test), '\n',
        'Ext', get_scores(cindex_ext), '\n',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train survival CoxPH pipeline (script port of survival.ipynb)')
    parser.add_argument('--modes', nargs='+', default=['RWD'], choices=[m.value for m in Mode], help='Modes to pass to DataLoader.create_dataset (default: RWD). Can specify multiple modes.')
    parser.add_argument('--outcome', default='OS MONTHS', help="Outcome column name (default: 'OS MONTHS').")
    parser.add_argument('--subanalysis', default='C23', choices=[s.value for s in Subanalysis], help='Subanalysis value to pass to DataLoader.create_dataset (default: C23).')
    parser.add_argument('--target-features', type=int, default=15, dest='target_features', help='Number of target features for selection')
    parser.add_argument('--uncertainty', type=int, default=5, help='Uncertainty parameter for coxnet_selection')
    args = parser.parse_args()

    # Convert string arguments to enum objects
    args.modes = [Mode(m) for m in args.modes]
    args.subanalysis = Subanalysis(args.subanalysis)

    main(args)
