import math
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

# Metadata columns held out of imputation/normalization when present, and
# re-attached to the result unchanged. Callers that don't carry these
# columns (e.g. mlef_pipeline, which slices modality-specific feature
# frames) are unaffected since the columns simply aren't found.
METADATA_COLUMNS = ['CENTER', 'SET']

CB_CATEGORICAL_FEATURES = [
    "ECOG PS",
    "SEX",
    "SMOKING CURRENT",
    "SMOKING FORMER",
    "SMOKING NEVER",
    "PDL1 CATEGORY",
    "PARENCHYMAL BRAIN METS AT IO START",
    "LIVER METS AT IO START",
    "BONE METS AT IO START",
]

CB_CATEGORICAL_BOUNDS: dict = {
    "ECOG PS":                            (0, 4),
    "SEX":                                (0, 1),
    "SMOKING CURRENT":                    (0, 1),
    "SMOKING FORMER":                     (0, 1),
    "SMOKING NEVER":                      (0, 1),
    "PDL1 CATEGORY":                      (0, 2),
    "PARENCHYMAL BRAIN METS AT IO START": (0, 1),
    "LIVER METS AT IO START":             (0, 1),
    "BONE METS AT IO START":              (0, 1),
}

GEN_CATEGORICAL_FEATURES = [
    "DRIVER",
    "KRAS",
    "P53",
    "STK11",
]

GEN_CATEGORICAL_BOUNDS: dict = {
    "DRIVER": (0, 1),
    "KRAS":   (0, 1),
    "P53":    (0, 1),
    "STK11":  (0, 1),
}


def impute_df(df: pd.DataFrame, imputer=None, categorical_features: list = None, categorical_bounds: dict = None) -> Tuple[pd.DataFrame, IterativeImputer]:
    """Impute missing values in a feature DataFrame via MICE-style iterative imputation.

    Any `CENTER` / `SET` columns present are held out as metadata and
    re-attached after imputation. All remaining columns are coerced to
    numeric, so non-numeric strings become NaN and are imputed alongside any
    pre-existing missing values.

    Pass `imputer=None` on the training split to fit a fresh imputer; reuse the
    returned imputer on test / external-validation splits so the same fitted
    model is applied (preventing leakage).

    Categorical columns named in `categorical_features` are post-processed:
    imputed values are clipped to the supplied `categorical_bounds[col]` (or the
    observed [min, max] of the split if absent) and rounded to the nearest int,
    so they land back inside a valid level set.

    Note on the sampling toggle: the imputer is *fit* with `sample_posterior=True`
    (MICE-like Bayesian iterations) but transforms — on TRAIN as well as
    TEST/EXVAL — are done with `sample_posterior=False` so the output is
    deterministic. The original setting is restored at the end so the
    returned/persisted imputer is unchanged from how it was fit.

    Parameters
    ----------
    df : pd.DataFrame
        The features to impute, optionally with `CENTER` / `SET` columns.
    imputer : IterativeImputer, optional
        Pre-fitted imputer to reuse. If None, a new one is fit on `df`.
    categorical_features : list, optional
        Columns to clip-and-round after imputation. Empty list by default.
    categorical_bounds : dict, optional
        `{col: (lo, hi)}` clipping bounds for categorical columns. Columns not
        present fall back to the observed min/max of the input split.

    Returns
    -------
    imputed_df : pd.DataFrame
        Same shape and index as `df`, with NaNs filled.
    imputer : IterativeImputer
        The fitted imputer (newly trained or the one passed in).
    """
    metadata_cols = [col for col in METADATA_COLUMNS if col in df.columns]
    if imputer is not None:
        # Transform-only: restrict to exactly the columns (and order) the
        # imputer was fit on, so extra columns in `df` (e.g. a new cohort's
        # raw CSV) don't trip sklearn's fit/transform feature-name check.
        cols_to_impute = list(imputer.feature_names_in_)
    else:
        cols_to_impute = [col for col in df.columns if col not in metadata_cols]
    metadata = df[metadata_cols].copy()
    df_to_impute = df[cols_to_impute].copy()

    # Coerce so stray strings become NaN and get imputed.
    for col in df_to_impute.columns:
        df_to_impute[col] = pd.to_numeric(df_to_impute[col], errors='coerce')

    if categorical_features is None:
        categorical_features = []
    if categorical_bounds is None:
        categorical_bounds = {}

    if imputer is None:
        # sample_posterior=True draws from the per-iteration posterior during
        # fit — MICE-like behavior, not point estimates.
        imputer = IterativeImputer(
            missing_values=np.nan,
            random_state=10,
            n_nearest_features=3,
            initial_strategy='median',
            max_iter=10,
            sample_posterior=True
        )
        imputer = imputer.fit(df_to_impute)

    # Force a deterministic (mean) transform regardless of how the imputer was
    # fit, then restore the original flag so the persisted object is unchanged.
    was_sampling = imputer.sample_posterior
    imputer.sample_posterior = False
    try:
        imputed_df = pd.DataFrame(imputer.transform(df_to_impute), columns=df_to_impute.columns, index=df_to_impute.index)
    finally:
        imputer.sample_posterior = was_sampling

    # Snap categoricals back to integer levels in [lo, hi]. Falling back to the
    # split's observed min/max is brittle on TEST/EXVAL
    for col in categorical_features:
        if col not in imputed_df.columns:
            continue
        lo, hi = categorical_bounds.get(col, (df_to_impute[col].min(), df_to_impute[col].max()))
        imputed_df[col] = imputed_df[col].clip(lower=lo, upper=hi).round(0).astype(int)

    imputed_df = pd.concat([metadata, imputed_df], axis=1)
    return imputed_df, imputer


def normalize(df: pd.DataFrame, scaler: StandardScaler = None, to_standard_normalize: list = None, to_log_normalize: list = None, min_shifts: dict = None, categorical_features: list = None) -> Tuple[pd.DataFrame, StandardScaler, list, list, dict]:
    """Log-transform skewed features and standard-scale continuous features.

    Any `CENTER` / `SET` columns present are held out as metadata. The
    remaining columns are coerced to numeric and then normalized in two
    passes:

    1. Log step. For every column in `to_log_normalize`, the value is shifted
       by `min_shifts[col]` (so the column is non-negative), clipped at 0 to
       handle TEST/EXVAL values that fall below the TRAIN minimum, and then
       `log(x + 1)` is applied. These columns are renamed with a `log_` prefix
       in the output.
    2. Standard-scaling step. Every column in `to_standard_normalize` is
       centered and scaled with a fitted `StandardScaler`.

    On the training split (with all three optional arguments left at `None`),
    the function auto-selects:
    - log targets: columns with `|skew| > 0.9` that are not in `categorical_features`;
    - standard-scale targets: all columns not in `categorical_features`;
    - min shifts: `-min(col)` for any auto-selected log column whose train
      minimum is negative.
    The caller MUST pass `categorical_features` explicitly; there is no
    automatic detection. The auto-selected lists are then threaded through
    to the test / external-validation calls.

    For TEST / EXVAL the caller MUST pass back the `scaler`, both column
    lists, and `min_shifts` returned from the training call so the same
    transform is applied. Test values whose original signed value would have
    been below the train minimum are clamped to 0 before the log, which means
    out-of-distribution low values collapse to a single point in log space.

    Parameters
    ----------
    df : pd.DataFrame
        The features to normalize, optionally with `CENTER` / `SET` columns.
    scaler : StandardScaler, optional
        Pre-fitted scaler to reuse. If None, a new one is fit on the
        non-categorical columns of `df`.
    to_standard_normalize : list, optional
        Columns to standard-scale. Auto-selected when None.
    to_log_normalize : list, optional
        Columns to log-transform. Auto-selected when None.
    min_shifts : dict, optional
        `{col: shift}` map of per-column shifts to apply before `log(x + 1)`.
        Computed from `df` when None.
    categorical_features : list, optional
        Columns to exclude from log-transform and standard-scaling auto-selection.
        Only used on the training call (when `to_log_normalize` / `to_standard_normalize`
        are None). Pass the same list used for `impute_df`.

    Returns
    -------
    result : pd.DataFrame
        Metadata + normalized features (log-transformed columns are prefixed `log_`).
    scaler : StandardScaler
        Fitted scaler (None if no columns required standard-scaling).
    to_standard_normalize : list
        The standard-scaling column list actually used (original column names,
        not the `log_`-prefixed output names).
    to_log_normalize : list
        The log-transform column list actually used (original names).
    min_shifts : dict
        The shift map actually used (empty if no shifts were needed).
    """
    metadata_cols = [col for col in METADATA_COLUMNS if col in df.columns]
    metadata = df[metadata_cols].copy()
    df_to_norm = df.drop(columns=metadata_cols).copy()

    # Coerce to numeric; any non-parseable cells become NaN.
    for col in df_to_norm.columns:
        df_to_norm[col] = pd.to_numeric(df_to_norm[col], errors='coerce')

    features_names = list(df_to_norm.columns)
    if to_log_normalize is None:
        # Auto-select on the training split: skewed continuous columns get the
        # log transform; categoricals are skipped regardless of skew.
        _cats = categorical_features if categorical_features is not None else []
        print(f"Found categorical features: {_cats}")
        to_log_normalize = [
            column for column in df_to_norm.columns
            if abs(df_to_norm[column].skew()) > 0.9 and column not in _cats
        ]

    # When called on train (min_shifts=None), compute shifts from train data.
    # When called on test/exval, use the train-derived shifts and clip any remaining negatives to 0.
    if min_shifts is None:
        min_shifts = {}
        for col in to_log_normalize:
            col_min = df_to_norm[col].min()
            if col_min < 0:
                min_shifts[col] = float(-col_min)
    for col in to_log_normalize:
        shift = min_shifts.get(col, 0.0)
        df_to_norm[col] = (df_to_norm[col] + shift).clip(lower=0)
    df_to_norm[to_log_normalize] = df_to_norm[to_log_normalize].map(lambda x: math.log(x + 1))

    if to_standard_normalize is None:
        _cats = categorical_features if categorical_features is not None else []
        to_standard_normalize = [col for col in features_names if col not in _cats]

    if scaler is None and len(to_standard_normalize) > 0:
        scaler = StandardScaler()
        scaler.fit(df_to_norm[to_standard_normalize])

    if len(to_standard_normalize) > 0:
        df_to_norm[to_standard_normalize] = scaler.transform(df_to_norm[to_standard_normalize])
    # Rename log-transformed columns with a prefix so downstream code can
    # tell them apart from raw features.
    df_to_norm = df_to_norm.rename(columns={col: f'log_{col}' for col in to_log_normalize})

    result = pd.concat([metadata, df_to_norm], axis=1)
    return result, scaler, to_standard_normalize, to_log_normalize, min_shifts
