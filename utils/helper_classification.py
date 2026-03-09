import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from .DeLong_test import auc_roc_ci, delong_roc_variance, fastDeLong_no_weights, compute_ground_truth_statistics
from .dlif_helper import *
from scipy import stats
import shap
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from typing import Iterable, Optional, Union, List, Tuple, Literal, Dict
from sklearn.metrics import confusion_matrix
from itertools import combinations
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines.plotting import add_at_risk_counts

def delong_test_comparison(y_true, y_pred1, y_pred2, alpha=0.05):
    """
    Compare two ROC curves using DeLong's test for statistical significance.
    Compute p-value for the difference between two AUCs.
    
    Args:
        y_true: True binary labels
        y_pred1: Predictions from first model (e.g., biomarker)
        y_pred2: Predictions from second model (e.g., MLEF model)
        alpha: Significance level (default 0.05)
    
    Returns:
        dict: Contains AUCs, p-value, and significance test results
    """
    # Compute AUC and variance for each model
    auc1, var1 = delong_roc_variance(y_true, y_pred1)
    auc2, var2 = delong_roc_variance(y_true, y_pred2)
    
    # Compute covariance between the two AUCs
    # For this we need to combine predictions and compute joint variance
    predictions_combined = np.array([y_pred1, y_pred2])
    order, label_1_count = compute_ground_truth_statistics(y_true)
    predictions_sorted_transposed = predictions_combined[:, order]
    aucs, cov_matrix = fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    
    # Extract covariance
    cov_12 = cov_matrix[0, 1] if cov_matrix.ndim > 1 else 0
    
    # Compute test statistic
    auc_diff = auc1 - auc2
    var_diff = var1 + var2 - 2 * cov_12
    
    if var_diff <= 0:
        var_diff = 1e-10  # Avoid division by zero
    
    z_score = auc_diff / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
    
    # Get confidence intervals
    auc1_ci = auc_roc_ci(y_true, y_pred1, 0.95)[1]
    auc2_ci = auc_roc_ci(y_true, y_pred2, 0.95)[1]
    
    return {
        'auc1': auc1,
        'auc2': auc2,
        'auc1_ci': auc1_ci,
        'auc2_ci': auc2_ci,
        'auc_difference': auc_diff,
        'p_value': p_value,
        'z_score': z_score,
        'significant': p_value < alpha,
        'alpha': alpha
    }


def p_to_stars(p: float) -> str:
    """Converts a p-value to a significance star string."""
    if p <= 0.0001: return '****'
    if p <= 0.001: return '***'
    if p <= 0.01: return '**'
    if p <= 0.05: return '*'
    return 'ns'

def plot_auc_results(
    architecture: Union[Literal["MLEF", "DLIF"], Path],
    outcome: str,                                              # e.g. OS_24, OS_6, DCR
    analyses: Union[str, Iterable[str]],                       # e.g. "C23" or ["C2", "ADENO", ...]
    *,
    modality_order: Optional[List[str]] = None,                # enforce x-axis order
    exclude_modalities: Iterable[str] = ("DP", "FMRAD", "PYRAD"),
    title_prefix: Optional[str] = None,
    min_bubble: float = 30,
    max_bubble: float = 250,
    show: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
    dlif_base_path: Optional[Union[str, Path]] = None,         # Base path for DLIF pipeline
    dlif_eval_type: str = "standard",                          # standard, cross_validation, or evaluation
    dlif_feature_type: str = "hypothesis_driven",              # hypothesis_driven or data_driven
    dlif_extraction: str = "pyrad-noimp",                      # feature extraction method
    dlif_seed: int = 0,                                         # seed number
    use_preds: bool = True,                                     # whether to use prediction files for p-value computation
    save_name: Optional[str] = None                             # custom filename (without extension) for saved plot
):
    """
    Plot AUC results comparing different modalities for MLEF or DLIF architectures.

    Args:
        architecture: Either "MLEF" or "DLIF" (or a Path for custom locations)
        outcome: Outcome to analyze (e.g., 'OS_24', 'OS_6', 'DCR')
        analyses: Analysis name(s) to process (e.g., "C23" or ["C2", "C23"])
        modality_order: Optional list to enforce specific order of modalities on x-axis
        exclude_modalities: Modalities to exclude from the plot
        title_prefix: Optional prefix for the plot title
        min_bubble: Minimum bubble size for scatter plot
        max_bubble: Maximum bubble size for scatter plot
        show: Whether to display the plot
        save_dir: Optional directory to save the plot
        dlif_base_path: Base path for DLIF pipeline results (only used for DLIF)
        dlif_eval_type: DLIF evaluation type ('standard', 'cross_validation', or 'evaluation')
        dlif_feature_type: DLIF feature type ('hypothesis_driven' or 'data_driven')
        dlif_extraction: DLIF feature extraction method (e.g., 'pyrad-noimp')
        dlif_seed: DLIF random seed number
        use_preds: If True, compute p-values by reading prediction files and running DeLong test.
                   If False, skip p-value computation and stars will not be shown on the plot.
                   This is useful when prediction files are not available or when you only
                   want to visualize AUC metrics without statistical comparisons.

    Expected per-analysis layout:

    MLEF:
        mlef_pipeline/results/
         OS_24/ (or DCR/ OR OS_6/)
          C23/
            RWD/
              model_XX.pkl
              train_set.xlsx/.csv
              test_set.xlsx/.csv
              exval_set.csv
              prediction_CV.xlsx/.csv   (columns: Subject, y_pred, y_true)
              prediction_EXVAL.csv
              prediction_TEST.csv
              results.xlsx/.csv      (metrics incl. CV AUC)
            RWD_DP/
               RWD_ONLY/         (MLEF only)
            RWD_PYRAD/
            RWD_FMRAD/
            RWD_DP_PYRAD/
            RWD_DP_FMRAD/

    DLIF (when use_preds=False):
        dlif_pipeline/results/
         C23/
          os_months_24/ (or DCR/)
           classification/
            standard/ (or cross_validation/, evaluation/)
             hypothesis_driven/ (or data_driven/)
              pyrad-noimp/ (or other extraction methods)
               rwd/
                seed_0/
                 predictions.parquet
                 predictions_train.parquet
                 eval_auc_ci.csv
                 eval_classification_metrics.csv
               rwd_dp/
               rwd_radfm/
               rwd_radpy/
               rwd_radfm_dp/
               rwd_radpy_dp/

    DLIF (when use_preds=True):
        dlif_pipeline/preds/
         os_months_24/ (or DCR/)
          classification/
           standard/ (or cross_validation/)
            rwd/
             predictions.parquet
             predictions_train.parquet
             eval_auc_ci.csv
            rwd_dp/
            rwd_radfm/
            rwd_radpy/
            rwd_radfm_dp/
            rwd_radpy_dp/
    """

    # ------------------------- utilities -------------------------
    def _parse_mean_std(val) -> Tuple[float, float]:
        if pd.isna(val):
            return (np.nan, np.nan)
        if isinstance(val, (int, float, np.floating)):
            return (float(val), 0.0)
        s = str(val).replace("+/-", "±")
        parts = [p.strip() for p in s.split("±")]
        try:
            if len(parts) == 2:
                return (float(parts[0]), float(parts[1]))
            return (float(parts[0]), 0.0)
        except Exception:
            return (np.nan, np.nan)

    def _read_auc_cv(results_path: Path) -> Tuple[float, float]:
        if results_path is None or not results_path.exists():
            return (np.nan, np.nan)
        # Try to read as Excel first, then CSV
        try:
            if results_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(results_path)
            else:
                df = pd.read_csv(results_path)
        except Exception:
            return (np.nan, np.nan)
        # 1) direct 'CV AUC'
        for col in df.columns:
            if str(col).strip().upper() in ("CV AUC", "CV_AUC", "AUC_CV", "AUC_cv"):
                return _parse_mean_std(df[col].iloc[0])
        # 2) SET + AUC with a 'CV' row
        set_cols = [c for c in df.columns if str(c).strip().upper() in ("SET", "SPLIT", "PHASE")]
        auc_cols = [c for c in df.columns if str(c).strip().upper() == "AUC"]
        if set_cols and auc_cols:
            sub = df.copy()
            mask = False
            for sc in set_cols:
                mask = mask | (sub[sc].astype(str).str.upper().str.contains("CV"))
            sub = sub[mask]
            if not sub.empty:
                return _parse_mean_std(sub[auc_cols[0]].iloc[0])
        # 3) any 'AUC' column at first row
        if auc_cols:
            return _parse_mean_std(df[auc_cols[0]].iloc[0])
        return (np.nan, np.nan)

    def _read_auc_dlif(auc_csv_path: Path) -> Tuple[float, float]:
        """Read AUC from DLIF's eval_auc_ci.csv file."""
        if auc_csv_path is None or not auc_csv_path.exists():
            return (np.nan, np.nan)
        try:
            df = pd.read_csv(auc_csv_path)
            # Expected columns: auc, ci_lower, ci_upper
            if 'auc' in df.columns:
                auc = float(df['auc'].iloc[0])
                # Calculate std from CI if available
                if 'ci_lower' in df.columns and 'ci_upper' in df.columns:
                    ci_lower = float(df['ci_lower'].iloc[0])
                    ci_upper = float(df['ci_upper'].iloc[0])
                    std = (ci_upper - ci_lower) / (2)
                    return (auc, std)
                return (auc, 0.0)
        except Exception:
            pass
        return (np.nan, np.nan)

    def _read_model_name(model_path: Path) -> str:
        """
        Model files are named like 'model_LR' or 'model_RF'.
        Return the substring after the first underscore.
        """
        if model_path is None or not model_path.exists():
            return "UnknownModel"
        stem = model_path.name  # allow filenames without extensions
        # If it has an extension, strip it
        if "." in stem:
            stem = stem.split(".", 1)[0]
        if "_" in stem:
            return stem.split("_", 1)[1] or "UnknownModel"
        return "UnknownModel"

    def _read_n_samples(pred_path: Path) -> float:
        if pred_path is None or not pred_path.exists():
            return np.nan
        try:
            if pred_path.suffix.lower() in ['.xlsx', '.xls']:
                return int(pd.read_excel(pred_path).shape[0])
            else:
                return int(pd.read_csv(pred_path).shape[0])
        except Exception:
            return np.nan

    def _scale_sizes(raw_sizes: Iterable[Union[int, float]]) -> np.ndarray:
        arr = np.array(list(raw_sizes), dtype=float)
        if arr.size == 0:
            return np.array([])
        # If all values are NaN, return default size for all
        if np.all(np.isnan(arr)):
            return np.full(arr.shape, (min_bubble + max_bubble) / 2.0)
        amin, amax = np.nanmin(arr), np.nanmax(arr)
        # If min/max are not finite or range is too small, use default size
        if not np.isfinite(amin) or not np.isfinite(amax) or (amax - amin < 1e-9):
            return np.full(arr.shape, (min_bubble + max_bubble) / 2.0)
        # Normalize and scale, handling NaN values
        norm = (arr - amin) / (amax - amin + 1e-6)
        scaled = norm * (max_bubble - min_bubble) + min_bubble
        # Replace NaN values with default size
        scaled[np.isnan(scaled)] = (min_bubble + max_bubble) / 2.0
        return scaled

    def _ordered_modalities(mods: List[str]) -> List[str]:
        filtered = [m for m in mods if m not in exclude_modalities]
        if modality_order:
            # Build a lookup that maps both "CB…" and "RWD…" forms to the actual name in filtered
            def _normalise(name: str) -> str:
                return name.upper().replace("CB", "RWD")
            norm_to_actual = {_normalise(m): m for m in filtered}
            in_order = []
            for req in modality_order:
                actual = norm_to_actual.get(_normalise(req))
                if actual is not None:
                    in_order.append(actual)
            leftovers = [m for m in filtered if m not in in_order]
            return in_order + leftovers
        return filtered

    def _collect_modalities(analysis_dir: Path) -> List[str]:
        return _ordered_modalities([p.name for p in analysis_dir.iterdir()
                                    if p.is_dir() and p.name.upper().startswith("RWD")])

    def _pair_paths(analysis_dir: Path, modality: str):
        """
        Robust path resolver for MLEF:
        - accepts RWD_ONLY or rwd-only (any case)
        - accepts files named like results(.xlsx/.xls/.csv), prediction(_CV)(.xlsx/.csv), train_set(.xlsx/.csv), etc.
        - accepts files with different case
        """
        def find_first(dir_: Path, stems: List[str]) -> Optional[Path]:
            if not dir_.exists():
                return None
            # try exact matches first (check both xlsx and csv extensions)
            for st in stems:
                for ext in ("", ".xlsx", ".xls", ".csv"):
                    p = dir_ / f"{st}{ext}"
                    if p.exists():
                        return p
            # then glob by prefix (case-insensitive)
            cand: List[Path] = []
            for st in stems:
                cand += list(dir_.glob(f"{st}*"))
                cand += list(dir_.glob(f"{st.upper()}*"))
                cand += list(dir_.glob(f"{st.lower()}*"))
            # prefer xlsx/xls/csv if multiple
            cand = sorted(cand, key=lambda x: (x.suffix.lower() not in {".xlsx", ".xls", ".csv"}, len(x.name)))
            return cand[0] if cand else None

        def find_subdir_any(dir_: Path, names: List[str]) -> Optional[Path]:
            if not dir_.exists():
                return None
            # exact
            for n in names:
                p = dir_ / n
                if p.exists() and p.is_dir():
                    return p
            # case-insensitive scan
            low_targets = {n.lower(): n for n in names}
            for p in dir_.iterdir():
                if p.is_dir() and p.name.lower() in low_targets:
                    return p
            return None

        mod_dir = analysis_dir / modality

        paths = {
            "mod": {
                "results": find_first(mod_dir, ["results", "Results"]),
                "pred":    find_first(mod_dir, ["prediction_CV", "prediction", "Prediction"]),
                "model":   find_first(mod_dir, ["model_", "model"]),   # picks model_LR / model_RF
                "train":   find_first(mod_dir, ["train_set", "Train_set", "train"]),
            },
            "rwd_only": None
        }

        if arch_name == "MLEF":
            ro_dir = find_subdir_any(
                mod_dir,
                ["RWD_ONLY", "rwd-only", "Rwd_only", "RWD-ONLY"]
            )
            if ro_dir:
                paths["rwd_only"] = {
                    "results": find_first(ro_dir, ["results", "Results"]),
                    "pred":    find_first(ro_dir, ["prediction_CV", "prediction", "Prediction"]),
                    "model":   find_first(ro_dir, ["model_", "model"]),
                    "train":   find_first(ro_dir, ["train_set", "Train_set", "train"]),
                }
        return paths


    def _compute_pvalue(pred_mod_path: Path, pred_ro_path: Path) -> Optional[float]:
        """
        Read predictions directly from prediction files (Subject, y_pred, y_true),
        align on Subject, then run DeLong.
        Supports both .xlsx, .csv, and .parquet files.
        """
        if not use_preds:
            return None

        if not (pred_mod_path and pred_ro_path and pred_mod_path.exists() and pred_ro_path.exists()):
            return None
        try:
            # Read modality predictions
            if pred_mod_path.suffix.lower() == '.parquet':
                dm = pd.read_parquet(pred_mod_path)
            elif pred_mod_path.suffix.lower() in ['.xlsx', '.xls']:
                dm = pd.read_excel(pred_mod_path)
            else:
                dm = pd.read_csv(pred_mod_path)

            # Read RWD-only predictions
            if pred_ro_path.suffix.lower() == '.parquet':
                dr = pd.read_parquet(pred_ro_path)
            elif pred_ro_path.suffix.lower() in ['.xlsx', '.xls']:
                dr = pd.read_excel(pred_ro_path)
            else:
                dr = pd.read_csv(pred_ro_path)
        except Exception:
            return None

        # For parquet files (DLIF format), need to handle different column structure
        if pred_mod_path.suffix.lower() == '.parquet':
            # DLIF uses 'slide' instead of 'Subject' and needs softmax conversion
            if 'y_pred0' in dm.columns and 'y_pred1' in dm.columns:
                exp_pred0 = np.exp(dm['y_pred0'])
                exp_pred1 = np.exp(dm['y_pred1'])
                dm['y_pred'] = exp_pred1 / (exp_pred0 + exp_pred1)
            elif 'pred' in dm.columns:
                dm['y_pred'] = dm['pred']

            if 'slide' in dm.columns:
                dm = dm.rename(columns={'slide': 'Subject'})

        if pred_ro_path.suffix.lower() == '.parquet':
            if 'y_pred0' in dr.columns and 'y_pred1' in dr.columns:
                exp_pred0 = np.exp(dr['y_pred0'])
                exp_pred1 = np.exp(dr['y_pred1'])
                dr['y_pred'] = exp_pred1 / (exp_pred0 + exp_pred1)
            elif 'pred' in dr.columns:
                dr['y_pred'] = dr['pred']

            if 'slide' in dr.columns:
                dr = dr.rename(columns={'slide': 'Subject'})

        # minimal schema check
        for col in ("Subject", "y_proba", "y_true"):
            if col not in dm.columns:
                return None
        if "Subject" not in dr.columns or "y_proba" not in dr.columns:
            return None

        m = dm.rename(columns={"y_proba": "y_proba_mod"})
        r = dr.rename(columns={"y_proba": "y_proba_ro"})
        merged = pd.merge(m[["Subject", "y_true", "y_proba_mod"]],
                          r[["Subject", "y_proba_ro"]],
                          on="Subject", how="inner")
        if merged.empty:
            return None
        return float(delong_test_comparison(
            merged["y_true"].to_numpy(),
            merged["y_proba_mod"].to_numpy(),
            merged["y_proba_ro"].to_numpy()
        )['p_value'])

    def _plot_one(analysis: str, rows: List[dict]) -> plt.Figure:
        modalities = [r["modality"] for r in rows]
        if arch_name == "MLEF":
            X = np.arange(len(modalities)) * 1.6
        else:
            X = np.arange(len(modalities))

        auc_mod_mean = np.array([r["auc_mod_mean"] for r in rows], dtype=float)
        auc_mod_std  = np.array([r["auc_mod_std"]  for r in rows], dtype=float)
        n_train_mod  = np.array([r["n_train_mod"]  for r in rows], dtype=float)
        sizes        = _scale_sizes(n_train_mod)
        model_names  = [r["model_mod"] for r in rows]

        fig = plt.figure(figsize=(12, 6))
        # BLUE: modality
        if arch_name == "MLEF":
            label = "CV AUC"
        elif dlif_eval_type == "cross_validation":
            label = "CV AUC"
        elif dlif_eval_type == "evaluation":
            label = "Ext-Val AUC"
        else:
            label = "Test AUC"
        plt.plot(X, auc_mod_mean, linestyle="-", marker="o", label=label, color="#1a80bb")
        if len(sizes) > 0:
            plt.scatter(X, auc_mod_mean, s=sizes, color="#1a80bb", zorder=3)
        else:
            plt.scatter(X, auc_mod_mean, s=100, color="#1a80bb", zorder=3)
        plt.fill_between(X, auc_mod_mean - auc_mod_std, auc_mod_mean + auc_mod_std,
                         alpha=0.3, color="#8cc5e3", label="Confidence interval")

        multimodal_better = None

        if arch_name == "MLEF":
            auc_ro_mean = np.array([r["auc_ro_mean"] for r in rows], dtype=float)
            auc_ro_std  = np.array([r["auc_ro_std"]  for r in rows], dtype=float)
            plt.plot(X, auc_ro_mean, linestyle="-", marker="o", label="CV AUC - CB-only matched", color="#a00000")
            if len(sizes) > 0:
                plt.scatter(X, auc_ro_mean, s=sizes, color="#a00000", zorder=3)
            else:
                plt.scatter(X, auc_ro_mean, s=100, color="#a00000", zorder=3)
            plt.fill_between(X, auc_ro_mean - auc_ro_std, auc_ro_mean + auc_ro_std,
                             alpha=0.3, color="#d8a6a6", label="Confidence interval - CB-only matched")
            multimodal_better = (auc_mod_mean > auc_ro_mean)

        if arch_name == "DLIF":
            xticks = []
            for m, n in zip(modalities, n_train_mod):
                parts = m.split('_')
                parts = ['CB' if p == 'rwd' else p for p in parts]
                # Join with newlines
                formatted_name = '\n'.join(parts)
                xticks.append(f"{formatted_name}")
            plt.xticks(X, xticks, rotation=0, fontsize=12)
        else:
            xticks = []
            for m, n in zip(modalities, n_train_mod):
                parts = m.replace('RWD', 'CB').split('_')
                label_text = '\n'.join(parts)
                if np.isfinite(n):
                    label_text += f"\n(n={int(n)})"
                xticks.append(label_text)
            plt.xticks(X, xticks, rotation=0, fontsize=12)
        

        for i, (mval, mstd, mname) in enumerate(zip(auc_mod_mean, auc_mod_std, model_names)):
            base_y = 0.06 if (multimodal_better is not None and multimodal_better[i]) else 0.02
            if arch_name == "DLIF":
                plt.text(X[i], base_y, f"{mval:.2f} ± {mstd:.2f}",
                        fontsize=12, ha="center", color="#1a80bb")
            else:
                plt.text(X[i], base_y, f"{mval:.2f} ± {mstd:.2f} ({mname})",
                        fontsize=11, ha="center", color="#1a80bb")
            star = rows[i].get("stars", "")
            if star:
                plt.text(X[i] + 0.63, base_y + 0.006, star, fontsize=11, ha="left", va="center", color="black")

        # --- RED annotations (keep values but REMOVE stars here) ---
        if arch_name == "MLEF":
            for i, rrow in enumerate(rows):
                rv, rs, rname = rrow["auc_ro_mean"], rrow["auc_ro_std"], rrow["model_ro"]
                base_y = 0.02 if multimodal_better[i] else 0.06
                if np.isfinite(rv) and np.isfinite(rs):
                    plt.text(X[i], base_y, f"{rv:.2f} ± {rs:.2f} ({rname})",
                            fontsize=11, ha="center", color="#a00000")

        if arch_name == "MLEF":
            _auc_prefix = "CV AUC"
        elif dlif_eval_type == "cross_validation":
            _auc_prefix = "CV AUC"
        elif dlif_eval_type == "evaluation":
            _auc_prefix = "Ext-Val AUC"
        else:
            _auc_prefix = "Test AUC"
        ttl = title_prefix or f"{_auc_prefix} - {arch_name}"
        plt.title(f"{ttl} - {outcome} {analysis}", pad=18)
        plt.ylabel("AUC", fontsize = 14)
        plt.tick_params(axis='y', labelsize=13)
        plt.ylim(0, 1)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=12)
        plt.xlim(X[0] - 0.7, X[-1] + 0.85) if len(X) > 0 else None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            if save_name:
                out = Path(save_dir) / f"{save_name}.png"
            else:
                out = Path(save_dir) / f"{ttl.replace(' ', '_')}_{outcome}_{analysis}.png"
            plt.savefig(out, dpi=600, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ------------------------- main logic -------------------------
    if isinstance(analyses, str):
        analyses = [analyses]

    # Normalize architecture to both string name and Path
    if isinstance(architecture, Path):
        arch_name = architecture.name  # Get the last part of the path (e.g., "MLEF" or "DLIF")
    else:
        arch_name = architecture

    for analysis in analyses:
        # Build the correct path based on architecture
        if architecture == "MLEF":
            # MLEF: mlef_pipeline/results/outcome/analysis/ 
            base_path = Path("mlef_pipeline/results") / outcome / analysis
            # base_path = Path("new/results") / outcome / analysis
            analysis_dir = base_path
        else:  # DLIF
            if use_preds:
                # When use_preds=True, use simpler preds directory structure
                # dlif_pipeline/preds/{outcome}/classification/standard/
                if dlif_base_path is None:
                    dlif_base_path = Path("dlif_pipeline/preds")
                else:
                    dlif_base_path = Path(dlif_base_path)

                dlif_outcome = outcome_to_dlif(outcome)
                analysis_dir = dlif_base_path / dlif_outcome / "classification" / dlif_eval_type
            else:
                # When use_preds=False, use full results directory structure
                # dlif_pipeline/results/analysis/outcome/classification/eval_type/feature_type/extraction/
                if dlif_base_path is None:
                    dlif_base_path = Path("dlif_pipeline/results")
                else:
                    dlif_base_path = Path(dlif_base_path)

                dlif_outcome = outcome_to_dlif(outcome)
                analysis_dir = (dlif_base_path / analysis / dlif_outcome / "classification" /
                              dlif_eval_type / dlif_feature_type / dlif_extraction)

        if not analysis_dir.exists():
            continue

        # Collect modalities
        if architecture == "DLIF":
            # For DLIF, list directories and map them to MLEF names
            if not analysis_dir.exists():
                continue
            dlif_modalities = [p.name for p in analysis_dir.iterdir()
                             if p.is_dir() and p.name.lower().startswith("rwd")]
            modalities = [map_dlif_to_mlef_modality(m) for m in dlif_modalities]
            modalities = _ordered_modalities(modalities)
        else:
            modalities = _collect_modalities(analysis_dir)

        if not modalities:
            continue

        rows: List[dict] = []
        for mod in modalities:
            # Get paths based on architecture
            if architecture == "DLIF":
                paths = pair_paths_dlif(analysis_dir, mod, dlif_eval_type, 'classification', dlif_seed, use_preds = use_preds)
                # Read DLIF-specific files
                auc_m, std_m = _read_auc_dlif(paths["mod"]["results"])
                model_m = "DLIF"  # DLIF uses MIL models
                ntrain_m = read_n_train_dlif(paths["mod"]["pred"])
            else:
                paths = _pair_paths(analysis_dir, mod)
                # modality side
                auc_m, std_m = _read_auc_cv(paths["mod"]["results"])
                model_m = _read_model_name(paths["mod"]["model"])
                ntrain_m = _read_n_samples(paths["mod"]["pred"])

            row = {
                "modality": mod,
                "auc_mod_mean": float(auc_m),
                "auc_mod_std": float(std_m) if np.isfinite(std_m) else 0.0,
                "n_train_mod": ntrain_m,
                "model_mod": model_m,
            }

            # paired RWD_ONLY (MLEF only) with RWD red==blue behavior
            if arch_name == "MLEF":
                if mod == "RWD":
                    # enforce coincidence for RWD: red == blue; no p-value
                    row.update({
                        "auc_ro_mean": row["auc_mod_mean"],
                        "auc_ro_std":  row["auc_mod_std"],
                        "n_train_ro":  row["n_train_mod"],
                        "model_ro":    row["model_mod"],
                        "pvalue":      None,
                        "stars":       ""
                    })
                else:
                    ro = paths["rwd_only"]
                    if ro is not None:
                        auc_r, std_r = _read_auc_cv(ro["results"])
                        pval = _compute_pvalue(paths["mod"]["pred"], ro["pred"])
                        row.update({
                            "auc_ro_mean": float(auc_r),
                            "auc_ro_std":  float(std_r) if np.isfinite(std_r) else 0.0,
                            "n_train_ro":  _read_n_samples(ro["pred"]),
                            "model_ro":    _read_model_name(ro["model"]),
                            "pvalue":      pval,
                            "stars":       p_to_stars(pval) if (pval is not None and np.isfinite(pval)) else ""
                        })
                    else:
                        row.update({
                            "auc_ro_mean": np.nan,
                            "auc_ro_std":  np.nan,
                            "n_train_ro":  np.nan,
                            "model_ro":    "NA",
                            "pvalue":      None,
                            "stars":       ""
                        })

            rows.append(row)

        rows = [r for r in rows if np.isfinite(r["auc_mod_mean"])]
        if not rows:
            continue

        _ = _plot_one(analysis, rows)


def add_single_biomarkers(outcome: str, df: pd.DataFrame, to_show: list[str]) -> tuple[pd.DataFrame, list[str]]: 
        """
        Create the dataframe with the results of the biomarkers.
        
        Args:
            outcome (str): The outcome to use for the prediction.
            df (pd.DataFrame): The dataframe with biomarkers data.
            to_show (list[str]): The list of biomarkers to show.
        
        Returns:
            tuple[pd.DataFrame, list[str]]: The dataframe with the scores of the desired biomarkers.
        """       
        
        biomarkers = pd.DataFrame(index=df.index)
        if 'BIO_PDL1_50' in to_show:
            biomarkers['BIO_PDL1_50'] = df.apply(lambda x: 
                'TP' if (x[outcome] == 1 and x['PDL1 CATEGORY'] >= 2) else
                'FP' if (x[outcome] == 0 and x['PDL1 CATEGORY'] >= 2) else
                'FN' if (x[outcome] == 1 and x['PDL1 CATEGORY'] < 2) else
                'TN' if (x[outcome] == 0 and x['PDL1 CATEGORY'] < 2) else
                np.nan, axis=1
            )
        if 'BIO_PDL1_1' in to_show:
            biomarkers['BIO_PDL1_1'] = df.apply(lambda x: 
                'TP' if (x[outcome] == 1 and x['PDL1 CATEGORY'] >= 1) else 
                'FP' if (x[outcome] == 0 and x['PDL1 CATEGORY'] >= 1) else
                'FN' if (x[outcome] == 1 and x['PDL1 CATEGORY'] < 1) else
                'TN' if (x[outcome] == 0 and x['PDL1 CATEGORY'] < 1) else
                np.nan, axis=1
            )
        if 'BIO_NLR' in to_show:
            biomarkers['BIO_NLR'] = df.apply(lambda x: 
                'TP' if (x[outcome] == 1 and x['NLR'] <= 4) else 
                'FP' if (x[outcome] == 0 and x['NLR'] <= 4) else
                'FN' if (x[outcome] == 1 and x['NLR'] > 4) else
                'TN' if (x[outcome] == 0 and x['NLR'] > 4) else
                np.nan, axis=1
            )
        if 'BIO_LDH' in to_show:
            biomarkers['BIO_LDH'] = df.apply(lambda x:
                'TP' if (x[outcome] == 1 and x['LDH'] <= 400) else
                'FP' if (x[outcome] == 0 and x['LDH'] <= 400) else
                'FN' if (x[outcome] == 1 and x['LDH'] > 400) else
                'TN' if (x[outcome] == 0 and x['LDH'] > 400) else
                np.nan, axis=1
            )
        if 'BIO_ECOG' in to_show:
            biomarkers['BIO_ECOG'] = df.apply(lambda x:
                'TP' if (x[outcome] == 1 and x['ECOG PS'] in [0, 1]) else
                'FP' if (x[outcome] == 0 and x['ECOG PS'] in [0, 1]) else
                'FN' if (x[outcome] == 1 and x['ECOG PS'] in [2, 3, 4]) else
                'TN' if (x[outcome] == 0 and x['ECOG PS'] in [2, 3, 4]) else
                np.nan, axis=1
            )
        
        
        df = pd.concat([df, biomarkers], axis=1)
        
        return df, biomarkers.columns.tolist()
def convert_to_binary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Convert the specified columns in the dataframe to binary values.
    
    Args:
        df (pd.DataFrame): The dataframe with the biomarkers data.
        columns (list[str]): The list of columns to convert.
    
    Returns:
        pd.DataFrame: The dataframe with the specified columns converted to binary values.
    """
    for col in columns:
        df[f'binary_{col}'] = df[col].map({'TP': 1, 'FP': 1, 'FN': 0, 'TN': 0})
    return df
#compute AUC, accuracy, sensitivity, specificity for each biomarker
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def compute_metrics_biomarker(y_true, y_pred):
    """
    Compute AUC, accuracy, sensitivity, and specificity.
    
    Args:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
    
    Returns:
        dict: Dictionary with AUC, accuracy, sensitivity, and specificity.
    """
    auc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'AUC': auc,
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    }

def compute_metrics(y_true, y_pred, y_proba):
    """
    Compute AUC, accuracy, sensitivity, and specificity.
    
    Args:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
        y_proba (pd.Series): Predicted probabilities.
    
    Returns:
        dict: Dictionary with AUC, accuracy, sensitivity, and specificity.
    """
    auc = roc_auc_score(y_true, y_proba)
    accuracy = accuracy_score(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    tn, fp, fn, tp = map(int, [tn, fp, fn, tp])
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'AUC': auc,
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    }

def plot_radar_charts(
    biomarker_scores: Dict,
    model_scores: Dict,
    dl_scores: Dict,
    p_values_ml:  List[float],
    p_values_dl:  List[float],
    size: List[int],
    biomarkers: List[str],
    metrics: List[str],
    title_prefix: str = 'Comparison'
):
    """
    Generates and displays a series of radar charts comparing biomarker and model performance.

    Args:
        biomarker_scores (Dict): A dictionary containing metric scores for each biomarker.
        model_scores (Dict): A dictionary containing MLEF metric's scores.
        dl_scores (Dict, optional): A dictionary containing DLIF metric's scores.
        p_values_ml (List[float]): A list of MLEF p-values for each biomarker, used for significance annotations.
        p_values_dl (List[float], optional): A list of DLIF p-values for each biomarker, used for significance annotations.
        size (List[int]): A list of sample sizes corresponding to each biomarker.
        biomarkers (List[str]): A list of biomarker names for the chart axes (e.g., ['PDL1', 'PDL1_50']).
        metrics (List[str]): A list of performance metrics to plot (e.g., ['AUC', 'Accuracy']).
        title_prefix (str, optional): A prefix for the chart title. Defaults to 'Comparison'.
    """
    


    N = len(biomarkers)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1] # Close the loop
    fontsize = 15
    # --- Plotting Loop ---
    for metric in metrics:
        plt.figure(figsize=(9, 9))
        ax = plt.subplot(111, projection='polar')
        
        vals_bio = [biomarker_scores[bio][metric] for bio in biomarkers] + [biomarker_scores[biomarkers[0]][metric]]
        vals_model = [model_scores[bio][metric] for bio in biomarkers] + [model_scores[biomarkers[0]][metric]]
        vals_dl = [dl_scores[bio][metric] for bio in biomarkers] + [dl_scores[biomarkers[0]][metric]] if dl_scores else None

        # Plotting the data
        ax.plot(angles, vals_bio, color="#811850", linewidth=2, label='Biomarkers')
        ax.fill(angles, vals_bio, color='#811850', alpha=0.5)
        ax.plot(angles, vals_model, color="#156ba9", linewidth=2, label='MLEF')
        ax.fill(angles, vals_model, color='#156ba9', alpha=0.5)
        if vals_dl is not None:
            ax.plot(angles, vals_dl, color="#477439", linewidth=2, label='DLIF')
            ax.fill(angles, vals_dl, color="#477439", alpha=0.5)

        
        # --- Labels and Ticks ---
        # if metric == 'AUC':
        #     tick_labels = [f"{b} {p_to_stars(p_values[i])}\nn={size[i]}" for i, b in enumerate(biomarkers)]
        # else:
        tick_labels = [f"{b}\nn={size[i]}" for i, b in enumerate(biomarkers)]
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tick_labels, fontsize=fontsize)
        ax.tick_params(axis='x', pad=46)
        
        ax.set_ylim(0, 1)
        ax.set_title(f'{title_prefix} - {metric} Comparison', pad=84, fontsize=14)
        
        # --- Annotations ---
        colors = ['#811850', '#156ba9', "#477439"]
        for i, angle in enumerate(angles[:-1]):
            if metric == 'AUC':
                bio_val_text = f"Biomarker: {vals_bio[i]:.2f}"
                model_val_text = f"MLEF: {vals_model[i]:.2f} {p_to_stars(p_values_ml[i])}"
                if vals_dl is not None:
                    dl_val_text = f"DLIF: {vals_dl[i]:.2f} {p_to_stars(p_values_dl[i])}"
            else:
                bio_val_text = f"Biomarker: {vals_bio[i]:.2f}"
                model_val_text = f"MLEF: {vals_model[i]:.2f}"
                if vals_dl is not None:
                    dl_val_text = f"DLIF: {vals_dl[i]:.2f}"
            angle_deg = np.rad2deg(angle)

            if 85 < angle_deg < 95: # Top
                ha = 'center'
                ax.text(angle, 1.15, model_val_text, color=colors[1], ha=ha, va='bottom', fontsize=fontsize)
                ax.text(angle, 1, bio_val_text, color=colors[0], ha=ha, va='bottom', fontsize=fontsize)
                if vals_dl is not None:
                    ax.text(angle, 1.07, dl_val_text, color=colors[2], ha=ha, va='bottom', fontsize=fontsize)
            elif 265 < angle_deg < 275: # Bottom
                ha = 'center'
                ax.text(angle, 1.01, model_val_text, color=colors[1], ha=ha, va='top', fontsize=fontsize)
                ax.text(angle, 1.16, bio_val_text, color=colors[0], ha=ha, va='top', fontsize=fontsize)
                if vals_dl is not None:
                    ax.text(angle, 1.09, dl_val_text, color=colors[2], ha=ha, va='top', fontsize=fontsize)
            elif angle_deg < 15 or angle_deg > 345: # Right
                ha = 'left'
                ax.text(angle - 0.16, 1.1, model_val_text, color=colors[1], ha=ha, va='bottom', fontsize=fontsize)
                ax.text(angle - 0.23, 1.1, bio_val_text, color=colors[0], ha=ha, va='top', fontsize=fontsize)
                if vals_dl is not None:
                    ax.text(angle - 0.16, 1.11, dl_val_text, color=colors[2], ha=ha, va='top', fontsize=fontsize)
            else: # Left
                ha = 'right'
                ax.text(angle + 0.16, 1.1, model_val_text, color=colors[1], ha=ha, va='bottom', fontsize=fontsize)
                ax.text(angle + 0.23, 1.1, bio_val_text, color=colors[0], ha=ha, va='top', fontsize=fontsize)
                if vals_dl is not None:
                    ax.text(angle + 0.16, 1.1, dl_val_text, color=colors[2], ha=ha, va='top', fontsize=fontsize)


         # --- ADD SIGNIFICANCE LEGEND ONLY FOR AUC PLOT ---
        if metric == 'AUC':
            significance_text = (
                'P-value significance:\n'
                r'ns: P > 0.05' '\n'
                r'*: P $\leq$ 0.05' '\n'
                r'**: P $\leq$ 0.01' '\n'
                r'***: P $\leq$ 0.001' '\n'
                r'****: P $\leq$ 0.0001'
            )
            # Add text to the figure, positioned at the bottom left
            plt.figtext(
                0.25, 0.9,
                significance_text,
                ha="right", va="top",
                fontsize=14,
                color="#474545",
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
            )

        ax.legend(loc='upper left', bbox_to_anchor=(0.9, 1.1), fontsize=17)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.2)
        plt.savefig(f"graphs/{title_prefix}_{metric}_biomarker_plot.png", dpi=600)
        plt.show()

def shap_beeswarm(model, X_train: pd.DataFrame, X_test: pd.DataFrame, mapping = {}) -> None:
    if isinstance(model, LogisticRegression):
        explainer = shap.Explainer(model.predict_proba, X_train)
    else:
        explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap_values.feature_names = [mapping.get(name, name) for name in shap_values.feature_names]
    shap.plots.beeswarm(shap_values[:,:,1], show=False, max_display=20)
    plt.show()


def generate_dlif_metric_files(
    preds_base_path: Union[str, Path] = "dlif_pipeline/preds",
    outcomes: Optional[List[str]] = None,
    modalities: Optional[List[str]] = None,
    task: str = "classification",
    training_type: str = "standard",
    overwrite: bool = False
) -> None:
    """
    Generate evaluation metric files (eval_auc_ci.csv, eval_classification_metrics.csv)
    from prediction parquet files in the preds directory.

    This function enables plotting and analysis using the pre-computed predictions
    from the paper without needing to retrain models. Since deep learning model
    reproducibility can be challenging and system-dependent, we provide these
    prediction files for recreating the exact plots from the paper.

    Args:
        preds_base_path: Path to the preds directory containing prediction parquet files
        outcomes: List of outcomes to process (e.g., ['DCR', 'os_months_24']).
                 If None, processes all outcomes found.
        modalities: List of modalities to process (e.g., ['rwd', 'rwd_dp']).
                   If None, processes all modalities found.
        task: Task type ('classification' or 'survival')
        training_type: Training type ('standard', 'cross_validation', or 'evaluation')
        overwrite: If True, regenerate files even if they already exist

    Example:
        >>> # Generate metrics for all outcomes and modalities
        >>> generate_dlif_metric_files()

        >>> # Generate metrics for specific outcome
        >>> generate_dlif_metric_files(outcomes=['DCR'])

        >>> # Regenerate all metrics (overwrite existing)
        >>> generate_dlif_metric_files(overwrite=True)
    """
    preds_base_path = Path(preds_base_path)

    if not preds_base_path.exists():
        raise FileNotFoundError(f"Preds directory not found: {preds_base_path}")

    # Helper function to compute classification metrics
    def _compute_classification_metrics(y_true, y_pred):
        """Compute F1, specificity, and sensitivity from predictions."""
        y_pred_binary = (y_pred >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Compute metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = sensitivity
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'f1': f1,
            'specificity': specificity,
            'sensitivity': sensitivity
        }

    # Find all outcome directories
    if outcomes is None:
        outcome_dirs = [d for d in preds_base_path.iterdir() if d.is_dir()]
    else:
        outcome_dirs = [preds_base_path / outcome for outcome in outcomes]

    total_processed = 0
    total_skipped = 0

    for outcome_dir in outcome_dirs:
        if not outcome_dir.exists():
            print(f"⚠️  Outcome directory not found: {outcome_dir}")
            continue

        outcome_name = outcome_dir.name
        print(f"\n📊 Processing outcome: {outcome_name}")

        # Navigate to task/training_type
        task_dir = outcome_dir / task / training_type
        if not task_dir.exists():
            print(f"  ⚠️  Task directory not found: {task_dir}")
            continue

        # Find all modality directories
        if modalities is None:
            modality_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith('rwd')]
        else:
            modality_dirs = [task_dir / mod for mod in modalities]

        for modality_dir in modality_dirs:
            if not modality_dir.exists():
                continue

            modality_name = modality_dir.name
            pred_file = modality_dir / "predictions.parquet"

            if not pred_file.exists():
                print(f"  ⚠️  No predictions.parquet found in {modality_dir}")
                continue

            # Check if metric files already exist
            auc_file = modality_dir / "eval_auc_ci.csv"
            metrics_file = modality_dir / "eval_classification_metrics.csv"

            if not overwrite and auc_file.exists() and metrics_file.exists():
                print(f"  ⏭️  {modality_name}: Metrics already exist (use overwrite=True to regenerate)")
                total_skipped += 1
                continue

            try:
                # Read predictions
                predictions = pd.read_parquet(pred_file)

                # Convert logits to probabilities using softmax
                if 'y_pred0' in predictions.columns and 'y_pred1' in predictions.columns:
                    # Apply softmax: exp(logit) / sum(exp(logits))
                    exp_pred0 = np.exp(predictions['y_pred0'])
                    exp_pred1 = np.exp(predictions['y_pred1'])
                    predictions['pred'] = exp_pred1 / (exp_pred0 + exp_pred1)
                elif 'pred' not in predictions.columns:
                    print(f"  ❌ {modality_name}: Missing prediction columns")
                    continue

                y_true = predictions['y_true'].values
                y_pred = predictions['pred'].values

                # Compute AUC with DeLong CI
                auc, ci = auc_roc_ci(y_true, y_pred, 0.95)

                # Compute classification metrics
                metrics = _compute_classification_metrics(y_true, y_pred)

                # Save AUC results
                with open(auc_file, 'w') as f:
                    f.write('auc,ci_lower,ci_upper\n')
                    f.write(f'{auc:.6f},{ci[0]:.6f},{ci[1]:.6f}\n')

                # Save classification metrics
                with open(metrics_file, 'w') as f:
                    f.write('f1,specificity,sensitivity\n')
                    f.write(f'{metrics["f1"]:.6f},{metrics["specificity"]:.6f},{metrics["sensitivity"]:.6f}\n')

                print(f"  ✅ {modality_name}: AUC={auc:.4f} [{ci[0]:.4f}, {ci[1]:.4f}], "
                      f"F1={metrics['f1']:.4f}, Sens={metrics['sensitivity']:.4f}, Spec={metrics['specificity']:.4f}")
                total_processed += 1

            except Exception as e:
                print(f"  ❌ {modality_name}: Error - {str(e)}")
                continue

    print(f"\n{'='*60}")
    print(f"✨ Summary: Processed {total_processed} modality/outcome combinations")
    if total_skipped > 0:
        print(f"⏭️  Skipped {total_skipped} (already exist)")
    print(f"{'='*60}")

#-------------------------- Fairness Functions -------------------------

def _read_dlif_predictions(parquet_path):
    """
    Read a DLIF predictions parquet file and return a DataFrame with
    [Subject, y_true, y_pred] matching the MLEF prediction format.

    Converts logits (y_pred0, y_pred1) to binary predictions via softmax + threshold.
    """
    parquet_path = Path(parquet_path)
    df = pd.read_parquet(parquet_path)

    # Convert logits to probability of class 1 via softmax
    exp0 = np.exp(df['y_pred0'])
    exp1 = np.exp(df['y_pred1'])
    y_proba = exp1 / (exp0 + exp1)

    # Store probability for AUC and binarize at 0.5 for TPR/FPR
    df['y_proba'] = y_proba
    df['y_pred'] = (y_proba >= 0.5).astype(int)

    # Rename slide -> Subject and convert to int to match RWD format
    df = df.rename(columns={'slide': 'Subject'})
    df['Subject'] = df['Subject'].astype(int)

    return df[['Subject', 'y_true', 'y_proba', 'y_pred']]


def load_predictions_and_data(outcome, base_path='mlef_pipeline/results',
                              architecture='MLEF',
                              dlif_base_path=None,
                              dlif_feature_type='hypothesis_driven',
                              dlif_extraction='pyrad-noimp',
                              dlif_modality='rwd',
                              dlif_seed=0,
                              analysis='C23'):
    """
    Load predictions and test data for a specific outcome.

    Parameters:
    -----------
    outcome : str
        Outcome name (e.g., 'DCR', 'OS6', 'OS24')
    base_path : str
        Base directory path for MLEF (default: 'mlef_pipeline/results')
    architecture : str
        'MLEF' or 'DLIF'
    dlif_base_path : str
        Base directory for DLIF results (e.g., 'dlif_pipeline/results')
    dlif_feature_type : str
        DLIF feature type (default: 'hypothesis_driven')
    dlif_extraction : str
        DLIF extraction method (default: 'pyrad-noimp')
    dlif_modality : str
        DLIF modality (default: 'rwd')
    dlif_seed : int
        DLIF seed number (default: 0)
    analysis : str
        Analysis name (default: 'C23')

    Returns:
    --------
    tuple : (predictions_df, test_df)
        - predictions_df: DataFrame with columns [Subject, y_true, y_pred]
        - test_df: DataFrame with CENTER and SEX columns, indexed by Subject
    """
    if architecture == 'DLIF':
        # Build DLIF standard path
        if dlif_base_path is None:
            dlif_base_path = 'dlif_pipeline/results'
        dlif_outcome = outcome_to_dlif(outcome)
        dlif_mod = map_mlef_to_dlif_modality(dlif_modality) if dlif_modality == dlif_modality.upper() else dlif_modality
        pred_dir = (Path(dlif_base_path) / analysis / dlif_outcome / 'classification' /
                    'standard' / dlif_feature_type / dlif_extraction / dlif_mod /
                    f'seed_{dlif_seed}')
        # Prefer eval predictions (full test set) over root predictions (CV val fold)
        eval_path = pred_dir / 'eval' / '00000-mb_attention_mil' / 'predictions.parquet'
        pred_path = eval_path if eval_path.exists() else pred_dir / 'predictions.parquet'
        predictions_df = _read_dlif_predictions(pred_path)
    else:
        # MLEF path
        pred_path = os.path.join(base_path, outcome, 'C23', 'RWD', 'prediction_TEST.csv')
        predictions_df = pd.read_csv(pred_path)

    # Load RWD data and join for CENTER/SEX
    rwd_path = 'data/rwd.csv'
    rwd_df = pd.read_csv(rwd_path)

    test_df = predictions_df[['Subject']].merge(
        rwd_df[['Subject', 'CENTER', 'SEX']],
        on='Subject',
        how='left'
    )
    test_df = test_df.set_index('Subject')

    print(f"Loaded predictions for {outcome} ({architecture}):")
    print(f"  Samples: {len(predictions_df)}")
    print(f"  Columns in test data: {test_df.shape[1]} ({', '.join(test_df.columns)})")
    print(f"  Subject order preserved: {(predictions_df['Subject'].values == test_df.index.values).all()}")

    return predictions_df, test_df

def load_predictions_and_data_exval(outcome, base_path='mlef_pipeline/results',
                                    architecture='MLEF',
                                    dlif_base_path=None,
                                    dlif_feature_type='hypothesis_driven',
                                    dlif_extraction='pyrad-noimp',
                                    dlif_modality='rwd',
                                    dlif_seed=0,
                                    analysis='C23'):
    """
    Load predictions and exval data for a specific outcome.

    Parameters:
    -----------
    outcome : str
        Outcome name (e.g., 'DCR', 'OS6', 'OS24')
    base_path : str
        Base directory path for MLEF (default: 'mlef_pipeline/results')
    architecture : str
        'MLEF' or 'DLIF'
    dlif_base_path : str
        Base directory for DLIF results (e.g., 'dlif_pipeline/results')
    dlif_feature_type : str
        DLIF feature type (default: 'hypothesis_driven')
    dlif_extraction : str
        DLIF extraction method (default: 'pyrad-noimp')
    dlif_modality : str
        DLIF modality (default: 'rwd')
    dlif_seed : int
        DLIF seed number (default: 0)
    analysis : str
        Analysis name (default: 'C23')

    Returns:
    --------
    tuple : (predictions_df, exval_df)
        - predictions_df: DataFrame with columns [Subject, y_true, y_pred]
        - exval_df: DataFrame with SEX and RACE columns, indexed by Subject
    """
    if architecture == 'DLIF':
        if dlif_base_path is None:
            dlif_base_path = 'dlif_pipeline/results'
        dlif_outcome = outcome_to_dlif(outcome)
        dlif_mod = map_mlef_to_dlif_modality(dlif_modality) if dlif_modality == dlif_modality.upper() else dlif_modality
        pred_dir = (Path(dlif_base_path) / analysis / dlif_outcome / 'classification' /
                    'evaluation' / dlif_feature_type / dlif_extraction / dlif_mod /
                    f'seed_{dlif_seed}')
        # Try direct predictions.parquet first, then eval subdirectory
        pred_path = pred_dir / 'predictions.parquet'
        if not pred_path.exists():
            pred_path = pred_dir / 'eval' / '00000-mb_attention_mil' / 'predictions.parquet'
        predictions_df = _read_dlif_predictions(pred_path)
    else:
        pred_path = os.path.join(base_path, outcome, 'C23', 'RWD', 'prediction_EXVAL.csv')
        predictions_df = pd.read_csv(pred_path)

    # Load RWD data and join for SEX/RACE
    rwd_path = 'data/rwd.csv'
    rwd_df = pd.read_csv(rwd_path)

    exval_df = predictions_df[['Subject']].merge(
        rwd_df[['Subject', 'SEX', 'RACE']],
        on='Subject',
        how='left'
    )
    exval_df = exval_df.set_index('Subject')

    print(f"Loaded EXVAL predictions for {outcome} ({architecture}):")
    print(f"  Samples: {len(predictions_df)}")
    print(f"  Columns in exval data: {exval_df.shape[1]} ({', '.join(exval_df.columns)})")
    print(f"  Subject order preserved: {(predictions_df['Subject'].values == exval_df.index.values).all()}")

    return predictions_df, exval_df

def compute_tpr_fpr(y_true, y_pred):
    """
    Compute True Positive Rate and False Positive Rate.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels (binary)
    
    Returns:
    --------
    tuple : (tpr, fpr)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr, fpr


def compute_metrics_by_center(y_true, y_pred, test_folds):
    """
    Compute TPR and FPR for each center.
    
    Parameters:
    -----------
    y_true : pd.Series
        True labels
    y_pred : array-like
        Predicted probabilities
    test_folds : pd.Series
        Center assignment for each sample
    
    Returns:
    --------
    pd.DataFrame : TPR and FPR by center
    """
    tpr_fpr = {}
    for site in test_folds.unique():
        if site is None:
            continue
        indices = test_folds == site
        tpr, fpr = compute_tpr_fpr(y_true[indices], y_pred[indices])
        tpr_fpr[site] = {'TPR': tpr, 'FPR': fpr}
    return pd.DataFrame(tpr_fpr).T

def _rate_by_group_conditional(df, group_col, cond_col, cond_val, pred_col='pred'):
    """Per-group rate Pr(pred==1 | cond_col==cond_val, group=g) and denominators."""
    sub = df[df[cond_col] == cond_val]
    if sub.empty:
        raise ValueError(f"No rows with {cond_col}=={cond_val}.")
    denom = sub.groupby(group_col).size().rename('denom')
    numer = sub[sub[pred_col] == 1].groupby(group_col).size().reindex(denom.index, fill_value=0)
    rate = (numer / denom).rename('rate')
    out = pd.concat([rate, denom], axis=1).reset_index().rename(columns={group_col: 'group'})
    return out


def _weighted_var_stat(rates_df):
    """Omnibus stat: sum w_g * (r_g - rbar)^2, w_g=denom."""
    w = rates_df['denom'].to_numpy()
    r = rates_df['rate'].to_numpy()
    if w.sum() == 0:
        return 0.0
    rbar = np.average(r, weights=w)
    return float(np.sum(w * (r - rbar)**2))


def _range_stat(rates_df):
    """Range statistic: max - min rate."""
    r = rates_df['rate'].to_numpy()
    return float(r.max() - r.min())


def _permute_groups_within_condition(df, group_col, cond_col, cond_val, rng):
    """Shuffle group labels only within rows where cond_col==cond_val."""
    sub_idx = df.index[df[cond_col] == cond_val]
    if len(sub_idx) == 0:
        return df
    shuffled = df.loc[sub_idx, group_col].to_numpy().copy()
    rng.shuffle(shuffled)
    out = df.copy()
    out.loc[sub_idx, group_col] = shuffled
    return out


def permutation_test_two_groups(
    df,
    group_col='group',
    groups=('A', 'B'),
    pred_col='pred',
    actual_col='actual',
    n_perms=1000,
    n_boot=1000,
    ci_level=0.95,
    random_state=42
):
    """
    Perform permutation test for TPR and FPR between two groups.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with columns: group_col, pred_col, actual_col
    group_col : str
        Column name for groups
    groups : tuple
        Two group values to compare
    pred_col : str
        Column name for predictions (0/1)
    actual_col : str
        Column name for actual labels (0/1)
    n_perms : int
        Number of permutations
    n_boot : int
        Number of bootstrap iterations for CIs
    ci_level : float
        Confidence interval level
    random_state : int
        Random seed
    
    Returns:
    --------
    dict : Results for TPR and FPR
    """
    a, b = groups
    rng = np.random.default_rng(random_state)

    def _obs_and_p(cond_val):
        sub = df[df[actual_col] == cond_val]
        if sub.empty:
            raise ValueError(f"No rows with {actual_col}=={cond_val}.")
        
        def _rate(g):
            gsub = sub[sub[group_col] == g]
            if gsub.empty:
                return np.nan, 0
            denom = len(gsub)
            numer = (gsub[pred_col] == 1).sum()
            return numer/denom, denom

        def _bootstrap_ci(g):
            gsub = sub[sub[group_col] == g]
            if gsub.empty or len(gsub) < 2:
                return np.nan, np.nan
            
            preds = gsub[pred_col].to_numpy()
            boot_rates = np.empty(n_boot)
            
            for i in range(n_boot):
                sample_idx = rng.choice(len(preds), size=len(preds), replace=True)
                boot_rates[i] = preds[sample_idx].mean()
            
            alpha = 1 - ci_level
            lower = np.percentile(boot_rates, 100 * alpha / 2)
            upper = np.percentile(boot_rates, 100 * (1 - alpha / 2))
            return lower, upper

        rA, nA = _rate(a)
        rB, nB = _rate(b)
        if np.isnan(rA) or np.isnan(rB):
            raise ValueError(f"One of the groups {groups} has no data under condition {cond_val}.")
        obs_diff = rA - rB

        ci_A_lower, ci_A_upper = _bootstrap_ci(a)
        ci_B_lower, ci_B_upper = _bootstrap_ci(b)

        # Permutation test
        diffs = np.empty(n_perms, dtype=float)
        labels = sub[group_col].to_numpy()
        
        for i in range(n_perms):
            shuffled = labels.copy()
            rng.shuffle(shuffled)
            A_idx = (shuffled == a)
            B_idx = (shuffled == b)
            rA_p = (sub[pred_col].to_numpy()[A_idx].mean() if A_idx.any() else np.nan)
            rB_p = (sub[pred_col].to_numpy()[B_idx].mean() if B_idx.any() else np.nan)
            diffs[i] = rA_p - rB_p

        p_two_sided = (np.sum(np.abs(diffs) >= abs(obs_diff)) + 1) / (n_perms + 1)
        
        return {
            'rate_A': rA, 'n_A': nA,
            'ci_A': (ci_A_lower, ci_A_upper),
            'rate_B': rB, 'n_B': nB,
            'ci_B': (ci_B_lower, ci_B_upper),
            'obs_diff': obs_diff,
            'p_value': p_two_sided
        }

    out_tpr = _obs_and_p(cond_val=1)
    out_fpr = _obs_and_p(cond_val=0)

    return {'TPR': out_tpr, 'FPR': out_fpr}


def omnibus_and_pairs(
    df,
    group_col='group',
    pred_col='pred',
    actual_col='actual',
    cond_val=1,
    n_perms=1000,
    n_boot=1000,
    ci_level=0.95,
    random_state=42,
    stat='weighted_var',
    use_maxT=True
):
    """
    Omnibus test + pairwise comparisons with max-T correction.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with columns: group_col, pred_col, actual_col
    group_col : str
        Column name for groups
    pred_col : str
        Column name for predictions (0/1)
    actual_col : str
        Column name for actual labels (0/1)
    cond_val : int
        Condition value (1 for TPR, 0 for FPR)
    n_perms : int
        Number of permutations
    n_boot : int
        Number of bootstrap iterations
    ci_level : float
        Confidence interval level
    random_state : int
        Random seed
    stat : str
        Statistic type ('weighted_var' or 'range')
    use_maxT : bool
        Use max-T correction for pairwise tests
    
    Returns:
    --------
    dict : Results including omnibus p-value and pairwise comparisons
    """
    rng = np.random.default_rng(random_state)

    # Observed rates
    obs_rates = _rate_by_group_conditional(df, group_col, actual_col, cond_val, pred_col)
    if stat == 'weighted_var':
        obs_stat = _weighted_var_stat(obs_rates)
    elif stat == 'range':
        obs_stat = _range_stat(obs_rates)
    else:
        raise ValueError("stat must be 'weighted_var' or 'range'.")

    # Bootstrap CIs
    def _bootstrap_ci_group(group_name):
        sub = df[df[actual_col] == cond_val]
        gsub = sub[sub[group_col] == group_name]
        
        if gsub.empty or len(gsub) < 2:
            return np.nan, np.nan
        
        preds = gsub[pred_col].to_numpy()
        boot_rates = np.empty(n_boot)
        
        for i in range(n_boot):
            sample_idx = rng.choice(len(preds), size=len(preds), replace=True)
            boot_rates[i] = preds[sample_idx].mean()
        
        alpha = 1 - ci_level
        lower = np.percentile(boot_rates, 100 * alpha / 2)
        upper = np.percentile(boot_rates, 100 * (1 - alpha / 2))
        return lower, upper

    ci_lower = []
    ci_upper = []
    for g in obs_rates['group']:
        lower, upper = _bootstrap_ci_group(g)
        ci_lower.append(lower)
        ci_upper.append(upper)
    
    obs_rates['ci_lower'] = ci_lower
    obs_rates['ci_upper'] = ci_upper

    groups = obs_rates['group'].tolist()
    pairs = list(combinations(groups, 2))
    rate_map = dict(zip(obs_rates['group'], obs_rates['rate']))
    obs_pair_diffs = {pair: abs(rate_map[pair[0]] - rate_map[pair[1]]) for pair in pairs}

    # Permutation nulls
    stat_null = np.empty(n_perms, dtype=float)
    max_pair_null = np.empty(n_perms, dtype=float) if use_maxT else None

    for i in range(n_perms):
        perm_df = _permute_groups_within_condition(df, group_col, actual_col, cond_val, rng)
        perm_rates = _rate_by_group_conditional(perm_df, group_col, actual_col, cond_val, pred_col)
        stat_null[i] = _weighted_var_stat(perm_rates) if stat == 'weighted_var' else _range_stat(perm_rates)

        if use_maxT:
            prm_map = dict(zip(perm_rates['group'], perm_rates['rate']))
            max_pair_null[i] = max(abs(prm_map[a] - prm_map[b]) for a, b in pairs)

    p_omni = (np.sum(stat_null >= obs_stat) + 1) / (n_perms + 1)

    results = {
        'observed_rates': obs_rates.sort_values('group').reset_index(drop=True),
        'omnibus_stat': obs_stat,
        'omnibus_pvalue': p_omni,
    }

    if use_maxT:
        pair_rows = []
        for (a, b), d in obs_pair_diffs.items():
            p_adj = (np.sum(max_pair_null >= d) + 1) / (n_perms + 1)
            pair_rows.append({'group_a': a, 'group_b': b, 'abs_diff': d, 'pvalue_maxT': p_adj})
        results['pairwise'] = pd.DataFrame(pair_rows).sort_values(['pvalue_maxT', 'abs_diff'], ascending=[True, False])

    results['null_stat'] = stat_null
    if use_maxT:
        results['null_max_pair_diff'] = max_pair_null

    return results


def permutation_fairness_TPR(df, **kw):
    """K-group omnibus + pairwise (maxT) for TPR (condition on actual=1)."""
    return omnibus_and_pairs(df, cond_val=1, **kw)


def permutation_fairness_FPR(df, **kw):
    """K-group omnibus + pairwise (maxT) for FPR (condition on actual=0)."""
    return omnibus_and_pairs(df, cond_val=0, **kw)

def _collect_bar_positions(ax, hue_order):
    """Extract bar positions and heights from seaborn plot."""
    xticks = np.array(ax.get_xticks())
    xlabels = [t.get_text() for t in ax.get_xticklabels()]
    groups = {lab: [] for lab in xlabels}

    for p in ax.patches:
        xc = p.get_x() + p.get_width()/2.0
        h = p.get_height()
        cat_idx = np.argmin(np.abs(xticks - xc))
        groups[xlabels[cat_idx]].append((xc, h))

    pos = {}
    tops = {}
    group_top = {}
    for outcome, bars in groups.items():
        bars = sorted(bars, key=lambda z: z[0])
        group_top[outcome] = max([h for _, h in bars]) if bars else 0.0
        for i, center in enumerate(hue_order):
            if i < len(bars):
                pos[(outcome, center)] = bars[i][0]
                tops[(outcome, center)] = bars[i][1]
    return pos, tops, group_top, xlabels


def _bracket(ax, x1, x2, y, h=0.01, star="*", lw=1.0, ls=":"):
    """Draw significance bracket."""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y],
            color="k", linewidth=lw, linestyle=ls, clip_on=False)
    ax.text((x1 + x2) / 2.0, y + h, star, ha="center", va="bottom",
            fontsize=12, fontweight="bold")


def _pairwise_df_to_matrix(pairwise_df, centers=None, pcol='pvalue_maxT', outcome=None):
    """Convert pairwise p-value table to symmetric matrix."""
    if pairwise_df is None or pairwise_df.empty:
        return pd.DataFrame()

    if centers is None:
        centers = sorted(pd.unique(pairwise_df[['group_a', 'group_b']].values.ravel()))

    arr = np.ones((len(centers), len(centers)))
    np.fill_diagonal(arr, 0.0)
    mat = pd.DataFrame(arr, index=centers, columns=centers)

    for _, row in pairwise_df.iterrows():
        a, b = row['group_a'], row['group_b']
        p = row.get(pcol, row.iloc[-1])
        if a in mat.index and b in mat.columns:
            mat.loc[a, b] = p
            mat.loc[b, a] = p

    if outcome is not None:
        mat['outcome'] = outcome

    return mat


def annotate_pairwise_significance(ax, pairwise_df, hue_order, alpha=0.05):
    """
    Annotate significant pairwise differences on grouped barplot.
    
    Parameters:
    -----------
    ax : matplotlib Axes
        Plot axes
    pairwise_df : pd.DataFrame
        Pairwise comparison results
    hue_order : list
        Order of groups in plot
    alpha : float
        Significance threshold
    """
    pos, tops, group_top, outcomes_on_plot = _collect_bar_positions(ax, hue_order)
    centers = list(hue_order)

    y0, y1 = ax.get_ylim()
    step = (y1 - y0) * 0.06
    bump = (y1 - y0) * 0.08
    tick_extra = (y1 - y0) * 0.08

    max_needed = y1

    for outcome in outcomes_on_plot:
        block = pairwise_df[pairwise_df["outcome"] == outcome].drop(columns=["outcome"]).copy()
        block = block.reindex(index=centers, columns=centers)

        sig_pairs = []
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                p = block.iloc[i, j]
                if pd.notna(p) and p < alpha:
                    sig_pairs.append((centers[i], centers[j], p))

        if not sig_pairs:
            continue

        base_y = group_top[outcome] + bump
        for k, (c1, c2, p) in enumerate(sig_pairs):
            x1 = pos[(outcome, c1)]
            x2 = pos[(outcome, c2)]
            y = base_y + k * step
            _bracket(ax, x1, x2, y, h=step*0.35, star=p_to_stars(p))

        max_needed = max(max_needed, base_y + len(sig_pairs)*step + tick_extra)

    ax.set_ylim(y0, max_needed)


def plot_fairness_by_center(
    center_fairness,
    pairwise_tpr,
    pairwise_fpr,
    metric='TPR',
    center_order=['INT', 'GHD', 'VHIO', 'SZMC', 'MH'],
    center_colors=None,
    patients_per_outcome=None,
    outcome_thresholds=None,
    figsize=(14, 6)
):
    """
    Plot TPR or FPR by center with statistical annotations.
    
    Parameters:
    -----------
    center_fairness : pd.DataFrame
        Fairness metrics by center
    pairwise_tpr/fpr : pd.DataFrame
        Pairwise comparison results
    metric : str
        'TPR' or 'FPR'
    center_order : list
        Order of centers
    center_colors : dict
        Color mapping for centers
    patients_per_outcome : dict
        Patient counts per center/outcome
    outcome_thresholds : dict
        Reference thresholds per outcome
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    if center_colors is None:
        center_colors = {
            'INT': '#08306B', 'GHD': '#2171B5', 'VHIO': '#4292C6',
            'SZMC': '#6BAED6', 'MH': '#9ECAE1'
        }
    
    plt.figure(figsize=figsize)
    
    # Prepare data
    center_fairness = center_fairness[center_fairness.index.isin(center_order)].copy()
    center_fairness['center'] = pd.Categorical(center_fairness.index, categories=center_order, ordered=True)
    
    # Create plot
    ax = sns.barplot(
        x='outcome',
        y=metric,
        hue='center',
        data=center_fairness,
        palette=center_colors,
        width=0.9,
        edgecolor='k',
        linewidth=0.3
    )
    
    if metric == 'FPR':
        for patch in ax.patches:
            patch.set_hatch('\\\\')
    
    # Add error bars
    outcomes = center_fairness['outcome'].unique()
    centers = center_order
    
    x_positions = []
    y_values = []
    yerr_lower = []
    yerr_upper = []
    
    group_width = 0.9
    bar_width = group_width / len(centers)
    
    for i, outcome in enumerate(outcomes):
        for j, center in enumerate(centers):
            sub = center_fairness.loc[
                (center_fairness['outcome'] == outcome) &
                (center_fairness['center'] == center)
            ]
            if not sub.empty:
                val = sub[metric].values[0]
                ci_low = sub[f'{metric}_ci_lower'].values[0]
                ci_up = sub[f'{metric}_ci_upper'].values[0]
                
                x_pos = i + (j - (len(centers) - 1) / 2) * bar_width
                
                x_positions.append(x_pos)
                y_values.append(val)
                yerr_lower.append(val - ci_low)
                yerr_upper.append(ci_up - val)
    
    ax.errorbar(
        x_positions, y_values,
        yerr=[yerr_lower, yerr_upper],
        fmt='none',
        ecolor='black',
        capsize=3,
        capthick=1,
        elinewidth=1,
        zorder=10
    )
    
    # Add value labels
    if patients_per_outcome is not None:
        label_offset = 0.02
        for container in ax.containers:
            if not all(isinstance(c, mpatches.Rectangle) for c in container):
                continue
            try:
                if len(container.patches) == 0:
                    continue
                rgb = np.array(container.patches[0].get_facecolor()[:3])
                palette_rgb = {name: np.array(plt.matplotlib.colors.to_rgb(col)) 
                              for name, col in center_colors.items()}
                center_name = min(palette_rgb, key=lambda name: np.linalg.norm(palette_rgb[name] - rgb))
                
                xlabels = [t.get_text() for t in ax.get_xticklabels()]
                
                for i, patch in enumerate(container):
                    outcome_lbl = xlabels[i] if i < len(xlabels) else None
                    if outcome_lbl and center_name:
                        outcome_data = center_fairness[
                            (center_fairness['outcome'] == outcome_lbl) &
                            (center_fairness['center'] == center_name)
                        ]
                        if not outcome_data.empty:
                            val = outcome_data[metric].values[0]
                            ci_lower = outcome_data[f'{metric}_ci_lower'].values[0]
                            ci_upper = outcome_data[f'{metric}_ci_upper'].values[0]
                            n = patients_per_outcome.get(outcome_lbl, {}).get(center_name)
                            
                            bar_x = patch.get_x() + patch.get_width() / 2
                            bar_y = val + (ci_upper - val) + label_offset
                            
                            ax.text(bar_x, bar_y, f"{val:.2f} ± {ci_upper - val:.2f}\n(n. {n})",
                                    ha='center', va='bottom', fontsize=8, color='black')
            except Exception as e:
                print(f"Failed to add labels: {e}")
    
    # Add thresholds
    if outcome_thresholds is not None:
        xticks = np.array(ax.get_xticks())
        xlabels = [t.get_text() for t in ax.get_xticklabels()]
        bars_by_outcome = {lab: [] for lab in xlabels}
        for p in ax.patches:
            xc = p.get_x() + p.get_width() / 2.0
            idx = int(np.argmin(np.abs(xticks - xc)))
            if 0 <= idx < len(xlabels):
                bars_by_outcome[xlabels[idx]].append(p)
        
        for outcome in xlabels:
            threshold = outcome_thresholds.get(outcome, 0.5)
            bars = bars_by_outcome.get(outcome, [])
            if not bars:
                continue
            start_x = min(b.get_x() for b in bars)
            end_x = max(b.get_x() + b.get_width() for b in bars)
            ax.plot([start_x, end_x], [threshold, threshold], 
                    color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    
    # Annotate significance
    pairwise_data = pairwise_tpr if metric == 'TPR' else pairwise_fpr
    annotate_pairwise_significance(ax, pairwise_data, hue_order=center_order, alpha=0.05)
    
    # Formatting
    direction = '↑ higher is better' if metric == 'TPR' else '↓ lower is better'
    plt.ylabel(f'{metric} - {direction}')
    plt.title(f'{metric} by Center and Outcome (with 95% CI)', pad=20)
    plt.ylim(0, 1.4)
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.legend(title='Center', loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_fairness_by_group(
    fairness_df,
    metric='TPR',
    group_col='Sex',
    group_colors=None,
    patients_per_outcome=None,
    outcome_thresholds=None,
    figsize=(6, 7)
    
):
    """
    Plot TPR or FPR by demographic group (sex/race).
    
    Parameters:
    -----------
    fairness_df : pd.DataFrame
        Fairness results by group
    metric : str
        'TPR' or 'FPR'
    group_col : str
        Column name for groups ('Sex' or 'Race')
    group_colors : list
        Colors for groups
    patients_per_outcome : dict
        Patient counts per group/outcome
    outcome_thresholds : dict
        Reference thresholds per outcome
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    if group_colors is None:
        group_colors = ["#E78FD4", "#7BB8E0"] if group_col == 'Sex' else [ '#4292C6',"#1E52A0"]
    
    plt.figure(figsize=figsize)
    
    # Filter and prepare data
    df_plot = fairness_df[fairness_df["Metric"] == metric].copy()
    df_plot[["ci_low", "ci_up"]] = pd.DataFrame(
        df_plot["CI"].apply(lambda t: t if isinstance(t, (tuple, list, np.ndarray)) and len(t) == 2 
                           else (np.nan, np.nan)).tolist(),
        index=df_plot.index
    )
    
    # Create plot
    ax = sns.barplot(
        x='outcome',
        y='Value',
        hue=group_col,
        data=df_plot,
        palette=group_colors,
        width=0.8,
        edgecolor='k',
        linewidth=0.2
    )
    
    if metric == 'FPR':
        for patch in ax.patches:
            patch.set_hatch('\\\\')
    
    # Add error bars
    outcomes = df_plot['outcome'].unique()
    groups = df_plot[group_col].unique()
    
    x_positions = []
    y_values = []
    yerr_lower = []
    yerr_upper = []
    
    group_width = 0.8
    bar_width = group_width / len(groups)
    
    for i, outcome in enumerate(outcomes):
        for j, group in enumerate(groups):
            sub = df_plot[
                (df_plot['outcome'] == outcome) & (df_plot[group_col] == group)
            ]
            if not sub.empty:
                val = sub['Value'].values[0]
                ci_low = sub['ci_low'].values[0]
                ci_up = sub['ci_up'].values[0]
                
                x_pos = i + (j - (len(groups) - 1) / 2) * bar_width
                
                x_positions.append(x_pos)
                y_values.append(val)
                yerr_lower.append(val - ci_low)
                yerr_upper.append(ci_up - val)
    
    if x_positions:
        ax.errorbar(
            x_positions, y_values,
            yerr=[yerr_lower, yerr_upper],
            fmt='none',
            ecolor='black',
            capsize=3,
            capthick=1,
            elinewidth=1,
            zorder=10
        )
    
    # Add value labels
    if patients_per_outcome is not None:
        label_offset = 0.02
        for container in ax.containers:
            if not all(isinstance(c, mpatches.Rectangle) for c in container):
                continue
            try:
                if len(container.patches) == 0:
                    continue
                rgb = np.array(container.patches[0].get_facecolor()[:3])
                palette_rgb = {name: np.array(plt.matplotlib.colors.to_rgb(col)) 
                              for name, col in zip(groups, group_colors)}
                group_name = min(palette_rgb, key=lambda name: np.linalg.norm(palette_rgb[name] - rgb))
                
                xlabels = [t.get_text() for t in ax.get_xticklabels()]
                
                for i, patch in enumerate(container):
                    outcome_lbl = xlabels[i] if i < len(xlabels) else None
                    if outcome_lbl and group_name:
                        outcome_data = df_plot[
                            (df_plot['outcome'] == outcome_lbl) &
                            (df_plot[group_col] == group_name)
                        ]
                        if not outcome_data.empty:
                            val = outcome_data['Value'].values[0]
                            ci_lower = outcome_data['ci_low'].values[0]
                            ci_upper = outcome_data['ci_up'].values[0]
                            n = patients_per_outcome.get(outcome_lbl, {}).get(group_name)
                            
                            bar_x = patch.get_x() + patch.get_width() / 2
                            bar_y = val + (ci_upper - val) + label_offset
                            
                            ax.text(bar_x, bar_y, f"{val:.2f} ± {ci_upper - val:.2f}\n(n. {n})",
                                    ha='center', va='bottom', fontsize=8, color='black')
            except Exception as e:
                print(f"Failed to add labels: {e}")
    
    # Add thresholds
    if outcome_thresholds is not None:
        xticks = np.array(ax.get_xticks())
        xlabels = [t.get_text() for t in ax.get_xticklabels()]
        bars_by_outcome = {lab: [] for lab in xlabels}
        for p in ax.patches:
            xc = p.get_x() + p.get_width() / 2.0
            idx = int(np.argmin(np.abs(xticks - xc)))
            if 0 <= idx < len(xlabels):
                bars_by_outcome[xlabels[idx]].append(p)
        
        for outcome in xlabels:
            threshold = outcome_thresholds.get(outcome, 0.5)
            bars = bars_by_outcome.get(outcome, [])
            if not bars:
                continue
            start_x = min(b.get_x() for b in bars)
            end_x = max(b.get_x() + b.get_width() for b in bars)
            ax.plot([start_x, end_x], [threshold, threshold],
                    color='black', linestyle='--', linewidth=0.8, alpha=0.7)

    # Add significance brackets between the two groups per outcome
    if 'p-value' in df_plot.columns:
        for i, outcome in enumerate(outcomes):
            sub = df_plot[df_plot['outcome'] == outcome]
            p = sub['p-value'].dropna()
            if p.empty or p.iloc[0] >= 0.05:
                continue
            p = p.iloc[0]
            groups_list = list(groups)
            x_coords, y_tops = [], []
            for j, group in enumerate(groups_list):
                x_coords.append(i + (j - (len(groups_list) - 1) / 2) * bar_width)
                row = sub[sub[group_col] == group]
                ci_up = row['ci_up'].values[0] if not row.empty else 0
                y_tops.append(ci_up if not np.isnan(ci_up) else (row['Value'].values[0] if not row.empty else 0))
            if len(x_coords) == 2:
                y_bracket = max(y_tops) + (0.12 if patients_per_outcome is not None else 0.04)
                _bracket(ax, x_coords[0], x_coords[1], y_bracket, h=0.025, star=p_to_stars(p))

    # Formatting
    direction = '↑ higher is better' if metric == 'TPR' else '↓ lower is better'
    plt.ylabel(f'{metric} - {direction}')
    plt.title(f'{metric} by {group_col} and Outcome (with 95% CI)', pad=20)
    plt.ylim(0, 1.3)
    plt.yticks(np.arange(0, 1.1, 0.2))
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if metric == 'FPR':
        for h in handles:
            h.set_hatch('')
    plt.legend(handles, labels, title=group_col, loc='upper left')
    plt.tight_layout()
    plt.show()

def analyze_outcome_fairness(
    outcome,
    outcome_name,
    base_path='mlef_pipeline/results',
    n_perms=1000,
    random_state=42,
    architecture='MLEF',
    dlif_base_path=None,
    dlif_feature_type='hypothesis_driven',
    dlif_extraction='pyrad-noimp',
    dlif_modality='rwd',
    dlif_seed=0,
    analysis='C23'
):
    """
    Complete fairness analysis for a single outcome.

    Parameters:
    -----------
    outcome : str
        Outcome folder name (e.g., 'DCR', 'OS6', 'OS24')
    outcome_name : str
        Display name for outcome (e.g., 'OS 24')
    base_path : str
        Base directory path where predictions are stored (MLEF)
    n_perms : int
        Number of permutations for tests
    random_state : int
        Random seed
    architecture : str
        'MLEF' or 'DLIF'
    dlif_base_path : str
        Base directory for DLIF results (e.g., 'dlif_pipeline/results')
    dlif_feature_type : str
        DLIF feature type (default: 'hypothesis_driven')
    dlif_extraction : str
        DLIF extraction method (default: 'pyrad-noimp')
    dlif_modality : str
        DLIF modality (default: 'rwd')
    dlif_seed : int
        DLIF seed number (default: 0)
    analysis : str
        Analysis name (default: 'C23')

    Returns:
    --------
    dict : Results including fairness metrics and test results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {outcome_name} ({architecture})")
    print(f"{'='*60}\n")

    # Load predictions and data
    predictions_df, test = load_predictions_and_data(
        outcome, base_path,
        architecture=architecture,
        dlif_base_path=dlif_base_path,
        dlif_feature_type=dlif_feature_type,
        dlif_extraction=dlif_extraction,
        dlif_modality=dlif_modality,
        dlif_seed=dlif_seed,
        analysis=analysis
    )
    
    # Extract predictions
    y_test = predictions_df['y_true'].values
    y_pred = predictions_df['y_pred'].values
    # Use probabilities for AUC when available (DLIF), fall back to binary (MLEF)
    y_score = predictions_df['y_proba'].values if 'y_proba' in predictions_df.columns else y_pred

    # Overall performance
    auc = roc_auc_score(y_test, y_score)
    tpr, fpr = compute_tpr_fpr(y_test, y_pred)
    print(f"Overall Test Performance:")
    print(f"  AUC: {auc:.3f}")
    print(f"  TPR (threshold): {tpr:.3f}")
    print(f"  FPR (threshold): {fpr:.3f}\n")

    # Fairness by center
    print("Analyzing fairness by center...")
    test_folds = test['CENTER']
    
    # Compute patient counts per center
    patients_per_center = test_folds.value_counts().to_dict()
    print(f"Patients per center: {patients_per_center}")
    
    # Check if SEX exists
    gender_col = 'SEX'
    sex_data = test[gender_col]
    
    # Compute patient counts per sex
    patients_per_sex = sex_data.value_counts().to_dict()
    print(f"Patients per sex: {patients_per_sex}\n")
    
    # Prepare data for permutation tests
    df_outcome = pd.DataFrame({
        'center': test_folds,
        'sex': sex_data,
        'pred': (y_pred).astype(int),
        'actual': y_test
    })
    
    # Sex-based fairness (test set)
    if gender_col is not None:
        print("\nSex-based fairness analysis (test set)...")
        try:
            sex_fair = permutation_test_two_groups(
                df_outcome, 
                group_col='sex', 
                groups=(0, 1),  # 0=Female, 1=Male
                n_perms=n_perms,
                random_state=random_state
            )

            print(f"  TPR - Female: {sex_fair['TPR']['rate_A']:.3f}, Male: {sex_fair['TPR']['rate_B']:.3f}, p={sex_fair['TPR']['p_value']:.4f}")
            print(f"  FPR - Female: {sex_fair['FPR']['rate_A']:.3f}, Male: {sex_fair['FPR']['rate_B']:.3f}, p={sex_fair['FPR']['p_value']:.4f}")
        except Exception as e:
            print(f"  Warning: Sex-based analysis failed: {e}")
            sex_fair = None
    else:
        sex_fair = None

    # Center-based fairness
    print("\nCenter-based fairness analysis...")
    tpr_out = permutation_fairness_TPR(
        df_outcome, 
        group_col='center', 
        n_perms=n_perms, 
        stat='weighted_var', 
        random_state=random_state
    )
    
    fpr_out = permutation_fairness_FPR(
        df_outcome, 
        group_col='center', 
        n_perms=n_perms, 
        stat='weighted_var', 
        random_state=random_state
    )
    
    print(f"  TPR omnibus p-value: {tpr_out['omnibus_pvalue']:.4f}")
    print(f"  FPR omnibus p-value: {fpr_out['omnibus_pvalue']:.4f}")
    
    
    return {
        'outcome_name': outcome_name,
        'predictions': predictions_df,
        'test': test,
        'auc': auc,
        'overall_tpr': tpr,
        'overall_fpr': fpr,
        'threshold_tpr': tpr,
        'threshold_fpr': fpr,
        'patients_per_center': patients_per_center,
        'patients_per_sex': patients_per_sex,
        'sex_fairness': sex_fair,
        'tpr_by_center': tpr_out,
        'fpr_by_center': fpr_out
    }

def analyze_outcome_fairness_exval(
    outcome,
    outcome_name,
    base_path='mlef_pipeline/results',
    n_perms=1000,
    random_state=42
):
    """
    Complete fairness analysis for a single outcome on external validation set.
    
    Parameters:
    -----------
    outcome : str
        Outcome folder name (e.g., 'DCR', 'OS6', 'OS24')
    outcome_name : str
        Display name for outcome (e.g., 'OS 24')
    base_path : str
        Base directory path where predictions are stored
    n_perms : int
        Number of permutations for tests
    random_state : int
        Random seed
    
    Returns:
    --------
    dict : Results including fairness metrics and test results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {outcome_name} (External Validation)")
    print(f"{'='*60}\n")
    
    # Load predictions and data
    predictions_df, test = load_predictions_and_data_exval(outcome, base_path)
    
    # Extract predictions
    y_test = predictions_df['y_true'].values
    y_pred = predictions_df['y_pred'].values
    y_score = predictions_df['y_proba'].values if 'y_proba' in predictions_df.columns else y_pred

    # Overall performance
    auc = roc_auc_score(y_test, y_score)
    tpr, fpr = compute_tpr_fpr(y_test, y_pred)
    print(f"Overall ExVal Performance:")
    print(f"  AUC: {auc:.3f}")
    print(f"  TPR (threshold): {tpr:.3f}")
    print(f"  FPR (threshold): {fpr:.3f}\n")

    # Check if RACE exists
    race_col = 'RACE'
    if race_col in test.columns:
        race_data = test[race_col]
        # Compute patient counts per race
        patients_per_race = race_data.value_counts().to_dict()
        print(f"Patients per race: {patients_per_race}\n")
    else:
        race_data = pd.Series([None]*len(test), index=test.index)
        patients_per_race = {}
    
    # Check if SEX exists
    gender_col = 'SEX'
    if gender_col in test.columns:
        sex_data = test[gender_col]
        patients_per_sex = sex_data.value_counts().to_dict()
        print(f"Patients per sex: {patients_per_sex}\n")
    else:
        sex_data = pd.Series([None]*len(test), index=test.index)
        patients_per_sex = {}
    
    # Prepare data for permutation tests
    df_outcome = pd.DataFrame({
        'race': race_data,
        'sex': sex_data,
        'pred': (y_pred).astype(int),
        'actual': y_test
    })
    
    # Sex-based fairness (binary)
    sex_fair = None
    if gender_col in test.columns:
        print("\nSex-based fairness analysis (exval set)...")
        try:
            sex_fair = permutation_test_two_groups(
                df_outcome,
                group_col='sex',
                groups=(0, 1),  # 0=Female, 1=Male
                n_perms=n_perms,
                random_state=random_state
            )

            print(f"  TPR - Female: {sex_fair['TPR']['rate_A']:.3f}, Male: {sex_fair['TPR']['rate_B']:.3f}, p={sex_fair['TPR']['p_value']:.4f}")
            print(f"  FPR - Female: {sex_fair['FPR']['rate_A']:.3f}, Male: {sex_fair['FPR']['rate_B']:.3f}, p={sex_fair['FPR']['p_value']:.4f}")
        except Exception as e:
            print(f"  Warning: Sex-based analysis failed: {e}")

    # Race-based fairness (multi-class)
    race_fair = None
    if race_col in test.columns:
        print("\nRace-based fairness analysis (exval set)...")
        try:
            # Use permutation_fairness_TPR/FPR for multi-class variable
            # This calculates rates per group and omnibus p-value
            race_tpr_out = permutation_fairness_TPR(
                df_outcome, 
                group_col='race', 
                n_perms=n_perms, 
                stat='weighted_var', 
                random_state=random_state
            )
            
            race_fpr_out = permutation_fairness_FPR(
                df_outcome, 
                group_col='race', 
                n_perms=n_perms, 
                stat='weighted_var', 
                random_state=random_state
            )
            
            print(f"  TPR omnibus p-value: {race_tpr_out['omnibus_pvalue']:.4f}")
            print(f"  FPR omnibus p-value: {race_fpr_out['omnibus_pvalue']:.4f}")
            
            # Structure similar to center_fairness in analyze_outcome_fairness
            race_fair = {'TPR': race_tpr_out, 'FPR': race_fpr_out}
            
        except Exception as e:
            print(f"  Warning: Race-based analysis failed: {e}")
    
    return {
        'outcome_name': outcome_name,
        'predictions': predictions_df,
        'test': test,
        'auc': auc,
        'overall_tpr': tpr,
        'overall_fpr': fpr,
        'threshold_tpr': tpr,
        'threshold_fpr': fpr,
        'patients_per_race': patients_per_race,
        'patients_per_sex': patients_per_sex,
        'sex_fairness': sex_fair,
        'race_fairness': race_fair
    }


def create_fairness_summary(results_dict):
    """
    Create summary DataFrames from multiple outcome analyses.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with outcome names as keys and analysis results as values
    
    Returns:
    --------
    tuple : (sex_fairness_df, center_fairness_df, pairwise_tpr, pairwise_fpr)
    """
    sex_fairness = pd.DataFrame(columns=['Metric', 'Sex', 'Value', 'CI', 'p-value', 'outcome'])
    center_fairness_list = []
    pairwise_tpr_list = []
    pairwise_fpr_list = []
    
    for outcome_name, results in results_dict.items():
        # Sex fairness
        for metric, values in results['sex_fairness'].items():
            ciA = values.get('ci_A')
            ciB = values.get('ci_B')
            ciA_rounded = (round(float(ciA[0]), 2), round(float(ciA[1]), 2)) if isinstance(ciA, (tuple, list, np.ndarray)) and len(ciA) == 2 else ciA
            ciB_rounded = (round(float(ciB[0]), 2), round(float(ciB[1]), 2)) if isinstance(ciB, (tuple, list, np.ndarray)) and len(ciB) == 2 else ciB
            
            sex_fairness.loc[len(sex_fairness)] = {
                'Metric': metric,
                'Sex': 'Female',
                'Value': values['rate_A'],
                'CI': ciA_rounded,
                'p-value': values['p_value'],
                'outcome': outcome_name
            }
            sex_fairness.loc[len(sex_fairness)] = {
                'Metric': metric,
                'Sex': 'Male',
                'Value': values['rate_B'],
                'CI': ciB_rounded,
                'p-value': values['p_value'],
                'outcome': outcome_name
            }
        
        # Center fairness
        tpr_rates = results['tpr_by_center']['observed_rates'].rename(
            columns={'group': 'center', 'rate': 'TPR', 'denom': 'TPR_denom'}
        )
        fpr_rates = results['fpr_by_center']['observed_rates'].rename(
            columns={'group': 'center', 'rate': 'FPR', 'denom': 'FPR_denom'}
        )
        
        tpr_rates['TPR_ci_lower'] = round(tpr_rates['ci_lower'], 2)
        tpr_rates['TPR_ci_upper'] = round(tpr_rates['ci_upper'], 2)
        fpr_rates['FPR_ci_lower'] = round(fpr_rates['ci_lower'], 2)
        fpr_rates['FPR_ci_upper'] = round(fpr_rates['ci_upper'], 2)
        
        center_df = pd.merge(
            tpr_rates[['center', 'TPR', 'TPR_ci_lower', 'TPR_ci_upper']],
            fpr_rates[['center', 'FPR', 'FPR_ci_lower', 'FPR_ci_upper']],
            on='center',
            how='outer'
        ).set_index('center').sort_index()
        
        center_df['outcome'] = outcome_name
        
        center_fairness_list.append(center_df)
        
        # Pairwise matrices
        centers = center_df.index.tolist()
        pairwise_tpr = _pairwise_df_to_matrix(
            results['tpr_by_center'].get('pairwise', pd.DataFrame()), 
            centers=centers, 
            outcome=outcome_name
        )
        pairwise_fpr = _pairwise_df_to_matrix(
            results['fpr_by_center'].get('pairwise', pd.DataFrame()), 
            centers=centers, 
            outcome=outcome_name
        )
        
        pairwise_tpr_list.append(pairwise_tpr)
        pairwise_fpr_list.append(pairwise_fpr)
    
    center_fairness = pd.concat(center_fairness_list, axis=0).sort_index()
    pairwise_tpr = pd.concat(pairwise_tpr_list, axis=0)
    pairwise_fpr = pd.concat(pairwise_fpr_list, axis=0)
    
    return sex_fairness, center_fairness, pairwise_tpr, pairwise_fpr

def analyze_exval_outcome_fairness_by_race(
    outcome,
    outcome_name,
    base_path='mlef_pipeline/results',
    n_perms=1000,
    random_state=42,
    architecture='MLEF',
    dlif_base_path=None,
    dlif_feature_type='hypothesis_driven',
    dlif_extraction='pyrad-noimp',
    dlif_modality='rwd',
    dlif_seed=0,
    analysis='C23'
):
    """
    Complete fairness analysis by race for external validation set.
    Filters to only include WHITE and BLACK OR AFRICAN AMERICAN races (excludes ASIAN).
    Also performs sex-based fairness analysis on the same filtered dataset.
    
    Parameters:
    -----------
    outcome : str
        Outcome folder name (e.g., 'DCR', 'OS6', 'OS24')
    outcome_name : str
        Display name for outcome (e.g., 'OS 24')
    base_path : str
        Base directory path where predictions are stored
    n_perms : int
        Number of permutations for tests
    random_state : int
        Random seed
    
    Returns:
    --------
    dict : Results including race fairness metrics and test results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing EXVAL Race & Sex Fairness for {outcome_name}")
    print(f"{'='*60}\n")
    
    # Load EXVAL predictions and data
    predictions_df, exval = load_predictions_and_data_exval(
        outcome, base_path,
        architecture=architecture,
        dlif_base_path=dlif_base_path,
        dlif_feature_type=dlif_feature_type,
        dlif_extraction=dlif_extraction,
        dlif_modality=dlif_modality,
        dlif_seed=dlif_seed,
        analysis=analysis
    )
    
    # Filter to only include WHITE and BLACK OR AFRICAN AMERICAN races
    valid_races = ['WHITE', 'BLACK OR AFRICAN AMERICAN']
    race_mask = exval['RACE'].isin(valid_races)
    
    # Apply filter to both predictions and exval data
    exval_filtered = exval[race_mask]
    predictions_filtered = predictions_df[predictions_df['Subject'].isin(exval_filtered.index)]
    
    if len(predictions_filtered) == 0:
        print(f"  Warning: No patients remain after race filtering!")
        return None
    
    # Extract predictions for race
    y_true_race = predictions_filtered['y_true'].values
    y_pred_race = predictions_filtered['y_pred'].values
    
    # Extract predictions for sex
    y_true_sex = predictions_df['y_true'].values
    y_pred_sex = predictions_df['y_pred'].values
    
    # Overall performance (filtered)
    tpr, fpr = compute_tpr_fpr(y_true_sex, y_pred_sex)
    print(f"\nOverall EXVAL Performance (all patients):")
    print(f"  TPR (threshold): {tpr:.3f}")
    print(f"  FPR (threshold): {fpr:.3f}\n")
    
    # Get race data for filtered subjects
    race_data = exval_filtered['RACE']
    sex_data = exval['SEX']
    
    # Compute patient counts per race and sex
    patients_per_race = race_data.value_counts().to_dict()
    patients_per_sex = sex_data.value_counts().to_dict()
    print(f"Patients per race: {patients_per_race}")
    print(f"Patients per sex: {patients_per_sex}\n")
    
    # Prepare data for permutation tests
    df_outcome_race = pd.DataFrame({
        'race': race_data.values,
        'pred': y_pred_race.astype(int),
        'actual': y_true_race
    })
    df_outcome_sex = pd.DataFrame({
        'sex': sex_data.values,
        'pred': y_pred_sex.astype(int),
        'actual': y_true_sex
    })

    
    # Race-based fairness
    print("\nRace-based fairness analysis (EXVAL set)...")
    try:
        race_fair = permutation_test_two_groups(
            df_outcome_race, 
            group_col='race', 
            groups=valid_races, 
            n_perms=n_perms, 
            random_state=random_state
        )
        
        print(f"  TPR - WHITE: {race_fair['TPR']['rate_A']:.3f}, BLACK OR AFRICAN AMERICAN: {race_fair['TPR']['rate_B']:.3f}, p={race_fair['TPR']['p_value']:.4f}")
        print(f"  FPR - WHITE: {race_fair['FPR']['rate_A']:.3f}, BLACK OR AFRICAN AMERICAN: {race_fair['FPR']['rate_B']:.3f}, p={race_fair['FPR']['p_value']:.4f}")
    except Exception as e:
        print(f"  Warning: Race-based analysis failed: {e}")
        race_fair = None
        
    # Sex-based fairness
    print("\nSex-based fairness analysis (EXVAL set)...")
    try:
        sex_fair = permutation_test_two_groups(
            df_outcome_sex,
            group_col='sex',
            groups=(0, 1),  # 0=Female, 1=Male
            n_perms=n_perms,
            random_state=random_state
        )
        
        print(f"  TPR - Female: {sex_fair['TPR']['rate_A']:.3f}, Male: {sex_fair['TPR']['rate_B']:.3f}, p={sex_fair['TPR']['p_value']:.4f}")
        print(f"  FPR - Female: {sex_fair['FPR']['rate_A']:.3f}, Male: {sex_fair['FPR']['rate_B']:.3f}, p={sex_fair['FPR']['p_value']:.4f}")
    except Exception as e:
        print(f"  Warning: Sex-based analysis failed: {e}")
        sex_fair = None
    
    return {
        'outcome_name': outcome_name,
        'overall_tpr': tpr,
        'overall_fpr': fpr,
        'patients_per_race': patients_per_race,
        'patients_per_sex': patients_per_sex,
        'race_fairness': race_fair,
        'sex_fairness': sex_fair,
        'threshold_tpr': tpr,
        'threshold_fpr': fpr
    }


def create_race_fairness_summary(results_dict):
    """
    Create summary DataFrame from multiple outcome race fairness analyses.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with outcome names as keys and analysis results as values
    
    Returns:
    --------
    tuple : (race_fairness_df, sex_fairness_df)
    """
    race_fairness = pd.DataFrame(columns=['Metric', 'Race', 'Value', 'CI', 'p-value', 'outcome'])
    sex_fairness = pd.DataFrame(columns=['Metric', 'Sex', 'Value', 'CI', 'p-value', 'outcome'])
    
    for outcome_name, results in results_dict.items():
        if results is None:
            continue
            
        # Race fairness
        if results['race_fairness'] is not None:
            for metric, values in results['race_fairness'].items():
                ciA = values.get('ci_A')
                ciB = values.get('ci_B')
                ciA_rounded = (round(float(ciA[0]), 2), round(float(ciA[1]), 2)) if isinstance(ciA, (tuple, list, np.ndarray)) and len(ciA) == 2 else ciA
                ciB_rounded = (round(float(ciB[0]), 2), round(float(ciB[1]), 2)) if isinstance(ciB, (tuple, list, np.ndarray)) and len(ciB) == 2 else ciB
                
                race_fairness.loc[len(race_fairness)] = {
                    'Metric': metric,
                    'Race': 'WHITE',
                    'Value': values['rate_A'],
                    'CI': ciA_rounded,
                    'p-value': values['p_value'],
                    'outcome': outcome_name
                }
                race_fairness.loc[len(race_fairness)] = {
                    'Metric': metric,
                    'Race': 'BLACK OR AFRICAN AMERICAN',
                    'Value': values['rate_B'],
                    'CI': ciB_rounded,
                    'p-value': values['p_value'],
                    'outcome': outcome_name
                }
                
        # Sex fairness
        if results['sex_fairness'] is not None:
            for metric, values in results['sex_fairness'].items():
                ciA = values.get('ci_A')
                ciB = values.get('ci_B')
                ciA_rounded = (round(float(ciA[0]), 2), round(float(ciA[1]), 2)) if isinstance(ciA, (tuple, list, np.ndarray)) and len(ciA) == 2 else ciA
                ciB_rounded = (round(float(ciB[0]), 2), round(float(ciB[1]), 2)) if isinstance(ciB, (tuple, list, np.ndarray)) and len(ciB) == 2 else ciB
                
                sex_fairness.loc[len(sex_fairness)] = {
                    'Metric': metric,
                    'Sex': 'Female',
                    'Value': values['rate_A'],
                    'CI': ciA_rounded,
                    'p-value': values['p_value'],
                    'outcome': outcome_name
                }
                sex_fairness.loc[len(sex_fairness)] = {
                    'Metric': metric,
                    'Sex': 'Male',
                    'Value': values['rate_B'],
                    'CI': ciB_rounded,
                    'p-value': values['p_value'],
                    'outcome': outcome_name
                }
    
    return race_fairness, sex_fairness


def plot_km_combined(target, train_set, test_set, uoc_set, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    
    colors = {'TRAIN': '#511635', 'TEST': '#1f77b4', 'EXVAL': '#800080'}
    
    datasets = {
        'TRAIN': train_set,
        'TEST': test_set,
        'EXVAL': uoc_set
    }
    
    median_annotations = []
    followup_annotations = []
    kmfs = []

    for label, dataset in datasets.items():
        kdf = dataset.dropna(subset=['OS MONTHS', 'DEATH EVENT'])

        # ---- KM for OS ----
        kmf = KaplanMeierFitter()
        kmf.fit(kdf['OS MONTHS'], kdf['DEATH EVENT'], label=label)
        kmf.plot(ax=ax, color=colors[label], show_censors=True, ci_show=True)
        kmfs.append(kmf)

        # Median OS + CI
        median_target = kmf.median_survival_time_
        median_ci = median_survival_times(kmf.confidence_interval_)
        lower_bound = median_ci.iloc[0, 0]
        upper_bound = median_ci.iloc[0, 1]

        median_annotations.append(
            f"{label}: {median_target:.1f} mo (95% CI: {lower_bound:.1f}-{upper_bound:.1f})"
        )

        # ---- Reverse KM for follow-up ----
        kmf_fu = KaplanMeierFitter()
        kmf_fu.fit(
            durations=kdf['OS MONTHS'],
            event_observed=1 - kdf['DEATH EVENT']
        )

        median_fu = kmf_fu.median_survival_time_
        min_fu = kdf['OS MONTHS'].min()
        max_fu = kdf['OS MONTHS'].max()

        followup_annotations.append(
            f"{label} FU: {median_fu:.1f} mo (range {min_fu:.1f}-{max_fu:.1f})"
        )

    # ---- Annotations ----
    ax.text(
        0.5, 0.95,
        "\n".join(median_annotations),
        transform=ax.transAxes,
        fontsize=14,
        ha='center',
        va='top',
        bbox=dict(facecolor='white', alpha=0.5)
    )


    add_at_risk_counts(
        *kmfs,
        ax=ax,
        labels=list(datasets.keys()),
        rows_to_show=["At risk"],
        fontsize=15
    )
    print('FOLLOW-UP STATISTICS:')
    for annotation in followup_annotations:
        print(annotation)

    ax.set_xlim(0, 100)
    ax.set_xlabel('Months')
    ax.set_ylabel('OS Probability')
    ax.set_title(title)
    plt.show()


