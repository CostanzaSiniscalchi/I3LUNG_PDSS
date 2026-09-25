"""Transform-only preprocessing for a new cohort (e.g. prospective), reusing
the imputers/scalers fit on the retrospective TRAIN split.

Unlike impute_and_normalize.py, this never fits anything and has no
TRAIN/TEST/EXVAL split to apply — every row in the new cohort's raw CSV is
transformed the same way.
"""
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# This file's own directory is added (not its parent) so the plain "artifacts"
# import below stays flat and never shadows the *repo-root* "utils" package
# that "from utils.preprocessing import ..." needs.
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from artifacts import ModalityArtifacts, load_modality_artifacts  # noqa: E402
from utils.preprocessing import (  # noqa: E402
    impute_df,
    normalize,
    CB_CATEGORICAL_FEATURES,
    CB_CATEGORICAL_BOUNDS,
    GEN_CATEGORICAL_FEATURES,
    GEN_CATEGORICAL_BOUNDS,
    NON_FEATURE_COLUMNS,
)

# Base (non-dp) modalities this script knows about by name, used to build
# the preprocess_new_cohort.py CLI's fixed --<modality>-path override flags.
# Digital pathology is handled separately via `dp_types` below since it's an
# open-ended set of source variants (digital_pathology, digital_pathology_titan,
# digital_pathology_titan_coral, ...), not a fixed list.
DEFAULT_FILENAMES = {
    "cb": "cb.csv",
    "genomics": "genomics.csv",
    "pyradiomics": "pyradiomics.csv",
    "fmrad": "fmrad.csv",
}

_CATEGORICAL = {
    "cb": (CB_CATEGORICAL_FEATURES, CB_CATEGORICAL_BOUNDS),
    "genomics": (GEN_CATEGORICAL_FEATURES, GEN_CATEGORICAL_BOUNDS),
}


def resolve_input_path(modality: str, raw_data_dir: Path, input_paths: Dict[str, str]) -> Optional[Path]:
    """Return the path to use for `modality`'s raw CSV: the override in
    `input_paths` if given, otherwise `<raw_data_dir>/<modality>.csv` if
    that file exists, otherwise None (modality is skipped).

    Every modality's default filename is just '<modality>.csv' — this also
    covers any digital_pathology_<token> variant without needing an entry in
    DEFAULT_FILENAMES.
    """
    override = input_paths.get(modality)
    if override:
        return Path(override)
    default_path = Path(raw_data_dir) / f"{modality}.csv"
    return default_path if default_path.exists() else None


def transform_modality(df: pd.DataFrame, modality: str, artifacts: ModalityArtifacts) -> pd.DataFrame:
    """Transform-only impute + normalize a new cohort's raw modality dataframe,
    using artifacts fit on the retrospective TRAIN split. Never fits anything.
    """
    reference = artifacts.imputer if artifacts.imputer is not None else artifacts.scaler
    expected_cols = list(reference.feature_names_in_)
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Modality {modality!r}: input is missing columns the fitted artifacts expect: {missing}"
        )

    if artifacts.imputer is not None:
        categorical_features, categorical_bounds = _CATEGORICAL[modality]
        df, _ = impute_df(
            df,
            imputer=artifacts.imputer,
            categorical_features=categorical_features,
            categorical_bounds=categorical_bounds,
        )

    result, _, _, _, _ = normalize(
        df,
        scaler=artifacts.scaler,
        to_standard_normalize=artifacts.to_standard_normalize,
        to_log_normalize=artifacts.to_log_normalize,
        min_shifts=artifacts.min_shifts,
    )
    return result


def preprocess_new_cohort(
    raw_data_dir: Path,
    artifacts_dir: Path,
    output_dir: Path,
    modalities=("cb", "genomics", "pyradiomics", "fmrad"),
    dp_types=("digital_pathology",),
    input_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Transform each modality present for the new cohort and write
    <output_dir>/<modality>_processed.csv in the same schema
    impute_and_normalize.py produces for the retrospective cohort, so
    create_parquet.py --data-dir <output_dir> can consume it unchanged.

    `dp_types` is the open-ended set of digital pathology source variants to
    process (e.g. 'digital_pathology', 'digital_pathology_titan',
    'digital_pathology_titan_coral', ...), matching create_parquet.py's and
    create_annotations.py's --dp-types. Each is treated exactly like any
    other modality: read from '<raw_data_dir>/<dp_type>.csv' (or an override
    path), transformed with the artifacts saved under that same name by
    impute_and_normalize.py, and written to '<output_dir>/<dp_type>_processed.csv'.

    A modality is skipped (with a printed warning) if neither an override
    path nor the default file exists in raw_data_dir — mirrors the HAS_{mod}
    pattern in create_annotations.py. This means radpy/radfm/any dp variant
    you don't have data for yet can simply be left out of `raw_data_dir` and
    will be skipped rather than erroring.
    """
    raw_data_dir = Path(raw_data_dir)
    artifacts_dir = Path(artifacts_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = input_paths or {}

    results = {}
    for modality in list(modalities) + list(dp_types):
        input_path = resolve_input_path(modality, raw_data_dir, input_paths)
        if input_path is None:
            print(f"[SKIP] {modality}: no input file found (looked for override, then {raw_data_dir / f'{modality}.csv'})")
            continue

        print(f"Processing {modality} from {input_path}...")
        df = pd.read_csv(input_path, index_col="Subject")
        df = df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns])
        artifacts = load_modality_artifacts(modality, artifacts_dir)
        result = transform_modality(df, modality, artifacts)

        output_path = output_dir / f"{modality}_processed.csv"
        result.to_csv(output_path)
        print(f"  Saved -> {output_path}")
        results[modality] = result

    return results
