"""Load the imputer/scaler/norm_config artifacts that impute_and_normalize.py
saves when fitting on the retrospective cohort, so a new cohort (e.g.
prospective) can be transformed with the exact same fitted objects.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

# Modalities that go through imputation before normalization. All other
# modalities (digital_pathology, any digital_pathology_<token> variant,
# pyradiomics, fmrad) are assumed complete and only get normalized — mirrors
# the branches in impute_and_normalize.py.
MODALITIES_WITH_IMPUTER = {"cb", "genomics"}

# impute_and_normalize.py saves genomics artifacts under the "gen" prefix,
# not "genomics" — kept as-is here for consistency with existing files on
# disk. Every other modality (including any digital_pathology_<token>
# variant, e.g. 'digital_pathology_titan_coral') is saved under its own name
# as-is, so no entry is needed here for those.
ARTIFACT_PREFIX_OVERRIDES = {
    "genomics": "gen",
}


def artifact_prefix(modality: str) -> str:
    return ARTIFACT_PREFIX_OVERRIDES.get(modality, modality)


@dataclass
class ModalityArtifacts:
    imputer: Optional[IterativeImputer]
    scaler: Optional[StandardScaler]
    to_standard_normalize: list
    to_log_normalize: list
    min_shifts: dict


def load_modality_artifacts(modality: str, artifacts_dir: Path) -> ModalityArtifacts:
    """Load the persisted imputer/scaler/norm_config for `modality` from `artifacts_dir`
    (the retrospective data directory that impute_and_normalize.py wrote into).

    Raises FileNotFoundError naming the exact missing path if an artifact isn't
    there — never falls back to fitting a new one.
    """
    artifacts_dir = Path(artifacts_dir)
    prefix = artifact_prefix(modality)

    scaler_path = artifacts_dir / f"{prefix}_scaler.pkl"
    config_path = artifacts_dir / f"{prefix}_norm_config.json"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found for modality {modality!r}: {scaler_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Norm config not found for modality {modality!r}: {config_path}")

    scaler = joblib.load(scaler_path)
    with open(config_path) as f:
        norm_config = json.load(f)

    imputer = None
    if modality in MODALITIES_WITH_IMPUTER:
        imputer_path = artifacts_dir / f"{prefix}_imputer.pkl"
        if not imputer_path.exists():
            raise FileNotFoundError(f"Imputer not found for modality {modality!r}: {imputer_path}")
        imputer = joblib.load(imputer_path)

    return ModalityArtifacts(
        imputer=imputer,
        scaler=scaler,
        to_standard_normalize=norm_config["to_standard_normalize"],
        to_log_normalize=norm_config["to_log_normalize"],
        min_shifts=norm_config.get("min_shifts", {}),
    )
