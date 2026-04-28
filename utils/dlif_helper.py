"""Shared DLIF / MLEF mapping utilities used by both classification and survival helpers."""
import pandas as pd
import numpy as np
from pathlib import Path

def outcome_to_dlif(outcome_str: str) -> str:
    """Convert MLEF outcome names to DLIF format."""
    mapping = {
        'OS_24': 'os_months_24',
        'OS_6': 'os_months_6',
        'DCR': 'DCR'
    }
    return mapping.get(outcome_str, outcome_str)


def map_dlif_to_mlef_modality(dlif_name: str) -> str:
    """
    Map DLIF modality names to MLEF-style names.
    DLIF: cb, cb_dp, cb_radfm, cb_radpy, cb_radfm_dp, cb_radpy_dp
    MLEF: CB, CB_DP, CB_FMRAD, CB_PYRAD, CB_DP_FMRAD, CB_DP_PYRAD
    """
    mapping = {
        'cb': 'CB',
        'cb_dp': 'CB_DP',
        'cb_radfm': 'CB_FMRAD',
        'cb_radpy': 'CB_PYRAD',
        'cb_radfm_dp': 'CB_DP_FMRAD',
        'cb_radpy_dp': 'CB_DP_PYRAD',
        'cb_radfm_dp_genomics': 'CB_DP_FMRAD_G',
        'cb_radpy_dp_genomics': 'CB_DP_PYRAD_G',
        'cb': 'CB',
        'cb_dp': 'CB_DP',
        'cb_radfm': 'CB_FMRAD',
        'cb_radpy': 'CB_PYRAD',
        'cb_radfm_dp': 'CB_DP_FMRAD',
        'cb_radpy_dp': 'CB_DP_PYRAD',
        'cb_radfm_dp_genomics': 'CB_DP_FMRAD_G',
        'cb_radpy_dp_genomics': 'CB_DP_PYRAD_G',
    }
    return mapping.get(dlif_name.lower(), dlif_name.upper())


def map_mlef_to_dlif_modality(mlef_name: str) -> str:
    """
    Map display/MLEF modality names to DLIF-style directory names.
    """
    mapping = {
        'CB': 'cb',
        'CB_DP': 'cb_dp',
        'CB_FMRAD': 'cb_radfm',
        'CB_PYRAD': 'cb_radpy',
        'CB_DP_FMRAD': 'cb_radfm_dp',
        'CB_DP_PYRAD': 'cb_radpy_dp',
        'CB_DP_FMRAD_G': 'cb_radfm_dp_genomics',
        'CB_DP_PYRAD_G': 'cb_radpy_dp_genomics',
        'CB': 'cb',
        'CB_DP': 'cb_dp',
        'CB_FMRAD': 'cb_radfm',
        'CB_PYRAD': 'cb_radpy',
        'CB_DP_FMRAD': 'cb_radfm_dp',
        'CB_DP_PYRAD': 'cb_radpy_dp',
        'CB_DP_FMRAD_G': 'cb_radfm_dp_genomics',
        'CB_DP_PYRAD_G': 'cb_radpy_dp_genomics',
    }
    return mapping.get(mlef_name.upper(), mlef_name.lower())

def read_n_train_dlif(predictions_path) -> float:
    """Read number of samples from DLIF prediction files.

    Args:
        predictions_path: Single Path (standard/evaluation) or list of Paths
            (cross-validation folds). For CV, concatenates all fold predictions
            to get total dataset size.
    """
    if predictions_path is None:
        return np.nan
    try:
        if isinstance(predictions_path, list):
            # CV: concatenate all fold predictions
            dfs = [pd.read_parquet(p) for p in predictions_path if p.exists()]
            if not dfs:
                return np.nan
            return int(len(pd.concat(dfs, ignore_index=True)))
        else:
            if not predictions_path.exists():
                return np.nan
            df = pd.read_parquet(predictions_path)
            return int(len(df))
    except Exception:
        return np.nan

def pair_paths_dlif(base_dir: Path, modality: str, dlif_eval_type: str, task: str, dlif_seed: int, use_preds=False) -> dict:
    """
    Path resolver for DLIF architecture.
    Returns dict with paths to DLIF files.
    """
    # DLIF modality directories use lowercase with underscores
    dlif_modality = map_mlef_to_dlif_modality(modality)

    # Build path based on whether we're using preds or results directory
    if use_preds:
        # Simpler structure: base_dir / modality /
        mod_dir = base_dir / dlif_modality
    else:
        # Full structure: base_dir / modality / seed_X /
        mod_dir = base_dir / dlif_modality / f"seed_{dlif_seed}"

    if not mod_dir.exists():
        return {
            "mod": {
                "results": None,
                "pred": None,
                "model": None,
                "train": None,
            },
            "cb_only": None
            "cb_only": None
        }

    # Find eval predictions based on evaluation type
    if dlif_eval_type == "cross_validation":
        # CV: predictions are split across fold_*/eval/predictions.parquet
        pred_fold_paths = sorted(mod_dir.glob("fold_*/eval/00000-mb_attention_mil/predictions.parquet"))
        pred_path = pred_fold_paths if pred_fold_paths else None
    else:
        # Standard / evaluation: single predictions file
        pred_path = mod_dir / "eval/00000-mb_attention_mil/predictions.parquet"
        if not pred_path.exists():
            pred_path = next(mod_dir.glob("eval/*/predictions.parquet"), None)

    if task =='classification':
        paths = {
            "mod": {
                "results": mod_dir / "eval_auc_ci.csv" if (mod_dir / "eval_auc_ci.csv").exists() else None,
                "pred": pred_path,
                "model": None,  # DLIF stores models differently
                "train": None,
            },
            "cb_only": None  # DLIF doesn't have CB_ONLY subdirectories
            "cb_only": None  # DLIF doesn't have CB_ONLY subdirectories
        }
    elif task == 'survival':
        paths = {
            "mod": {
                "results": mod_dir / "eval_cindex_ci.csv" if (mod_dir / "eval_cindex_ci.csv").exists() else None,
                "pred": pred_path,
                "model": None,
                "train": None,
            },
            "cb_only": None  # DLIF doesn't have CB_ONLY subdirectories
            "cb_only": None  # DLIF doesn't have CB_ONLY subdirectories
        }
    else:
        paths = {
            "mod": {
                "results": None,
                "pred": None,
                "model": None,
                "train": None,
            },
            "cb_only": None
            "cb_only": None
        }
    return paths