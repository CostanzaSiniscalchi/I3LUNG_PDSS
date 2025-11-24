import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from .DeLong_test import auc_roc_ci, delong_roc_variance, fastDeLong_no_weights, compute_ground_truth_statistics
from scipy import stats
import shap
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from typing import Iterable, Optional, Union, List, Tuple, Literal, Dict


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
    dlif_seed: int = 0                                         # seed number
):
    """
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

    DLIF:
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
                    # Approximate std from 95% CI: (upper - lower) / (2 * 1.96)
                    std = (ci_upper - ci_lower) / (2 * 1.96)
                    return (auc, std)
                return (auc, 0.0)
        except Exception:
            pass
        return (np.nan, np.nan)

    def _read_n_train_dlif(predictions_train_path: Path) -> int:
        """Read number of training samples from DLIF's predictions_train.parquet."""
        if predictions_train_path is None or not predictions_train_path.exists():
            return int(np.nan)
        try:
            df = pd.read_parquet(predictions_train_path)
            return int(len(df))
        except Exception:
            return int(np.nan)

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

    def _read_n_train(train_path: Path) -> int:
        if train_path is None or not train_path.exists():
            return int(np.nan)
        try:
            if train_path.suffix.lower() in ['.xlsx', '.xls']:
                return int(pd.read_excel(train_path).shape[0])
            else:
                return int(pd.read_csv(train_path).shape[0])
        except Exception:
            return int(np.nan)

    def _scale_sizes(raw_sizes: Iterable[Union[int, float]]) -> np.ndarray:
        arr = np.array(list(raw_sizes), dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return np.array([])
        amin, amax = np.nanmin(arr), np.nanmax(arr)
        if not np.isfinite(amin) or not np.isfinite(amax) or (amax - amin < 1e-9):
            return np.full_like(arr, (min_bubble + max_bubble) / 2.0)
        norm = (arr - amin) / (amax - amin + 1e-6)
        return norm * (max_bubble - min_bubble) + min_bubble

    def _ordered_modalities(mods: List[str]) -> List[str]:
        filtered = [m for m in mods if m not in exclude_modalities]
        if modality_order:
            in_order = [m for m in modality_order if m in filtered]
            leftovers = [m for m in filtered if m not in in_order]
            return in_order + leftovers
        return filtered

    def _map_dlif_to_mlef_modality(dlif_name: str) -> str:
        """
        Map DLIF modality names to MLEF-style names.
        DLIF: rwd, rwd_dp, rwd_radfm, rwd_radpy, rwd_radfm_dp, rwd_radpy_dp
        MLEF: RWD, RWD_DP, RWD_FMRAD, RWD_PYRAD, RWD_DP_FMRAD, RWD_DP_PYRAD
        """
        mapping = {
            'rwd': 'RWD',
            'rwd_dp': 'RWD_DP',
            'rwd_radfm': 'RWD_FMRAD',
            'rwd_radpy': 'RWD_PYRAD',
            'rwd_radfm_dp': 'RWD_DP_FMRAD',
            'rwd_radpy_dp': 'RWD_DP_PYRAD',
        }
        return mapping.get(dlif_name.lower(), dlif_name.upper())

    def _map_mlef_to_dlif_modality(mlef_name: str) -> str:
        """
        Map MLEF modality names to DLIF-style names.
        """
        mapping = {
            'RWD': 'rwd',
            'RWD_DP': 'rwd_dp',
            'RWD_FMRAD': 'rwd_radfm',
            'RWD_PYRAD': 'rwd_radpy',
            'RWD_DP_FMRAD': 'rwd_radfm_dp',
            'RWD_DP_PYRAD': 'rwd_radpy_dp',
        }
        return mapping.get(mlef_name.upper(), mlef_name.lower())

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

    def _pair_paths_dlif(base_dir: Path, modality: str) -> dict:
        """
        Path resolver for DLIF architecture.
        Returns dict with paths to DLIF files.
        """
        # DLIF modality directories use lowercase with underscores
        dlif_modality = _map_mlef_to_dlif_modality(modality)

        # Build path: base_dir / modality / seed_X /
        mod_dir = base_dir / dlif_modality / f"seed_{dlif_seed}"

        if not mod_dir.exists():
            return {
                "mod": {
                    "results": None,
                    "pred": None,
                    "model": None,
                    "train": None,
                },
                "rwd_only": None
            }

        paths = {
            "mod": {
                "results": mod_dir / "eval_auc_ci.csv" if (mod_dir / "eval_auc_ci.csv").exists() else None,
                "pred": mod_dir / "predictions.parquet" if (mod_dir / "predictions.parquet").exists() else None,
                "model": None,  # DLIF stores models differently
                "train": mod_dir / "predictions_train.parquet" if (mod_dir / "predictions_train.parquet").exists() else None,
            },
            "rwd_only": None  # DLIF doesn't have RWD_ONLY subdirectories
        }

        return paths


    def _compute_pvalue(pred_mod_path: Path, pred_ro_path: Path) -> Optional[float]:
        """
        Read predictions directly from prediction files (Subject, y_pred, y_true),
        align on Subject, then run DeLong.
        Supports both .xlsx and .csv files.
        """
        if not (pred_mod_path and pred_ro_path and pred_mod_path.exists() and pred_ro_path.exists()):
            return None
        try:
            # Read modality predictions
            if pred_mod_path.suffix.lower() in ['.xlsx', '.xls']:
                dm = pd.read_excel(pred_mod_path)
            else:
                dm = pd.read_csv(pred_mod_path)

            # Read RWD-only predictions
            if pred_ro_path.suffix.lower() in ['.xlsx', '.xls']:
                dr = pd.read_excel(pred_ro_path)
            else:
                dr = pd.read_csv(pred_ro_path)
        except Exception:
            return None

        # minimal schema check
        for col in ("Subject", "y_pred", "y_true"):
            if col not in dm.columns:
                return None
        if "Subject" not in dr.columns or "y_pred" not in dr.columns:
            return None

        m = dm.rename(columns={"y_pred": "y_pred_mod"})
        r = dr.rename(columns={"y_pred": "y_pred_ro"})
        merged = pd.merge(m[["Subject", "y_true", "y_pred_mod"]],
                          r[["Subject", "y_pred_ro"]],
                          on="Subject", how="inner")
        if merged.empty:
            return None
        return float(delong_test_comparison(
            merged["y_true"].to_numpy(),
            merged["y_pred_mod"].to_numpy(),
            merged["y_pred_ro"].to_numpy()
        )['p_value'])

    def _plot_one(analysis: str, rows: List[dict]) -> plt.Figure:
        modalities = [r["modality"] for r in rows]
        X = np.arange(len(modalities))

        auc_mod_mean = np.array([r["auc_mod_mean"] for r in rows], dtype=float)
        auc_mod_std  = np.array([r["auc_mod_std"]  for r in rows], dtype=float)
        n_train_mod  = np.array([r["n_train_mod"]  for r in rows], dtype=float)
        sizes        = _scale_sizes(n_train_mod)
        model_names  = [r["model_mod"] for r in rows]

        fig = plt.figure(figsize=(12, 6))
        # BLUE: modality
        plt.plot(X, auc_mod_mean, linestyle="-", marker="o", label="CV AUC", color="#1a80bb")
        plt.scatter(X, auc_mod_mean, s=sizes, color="#1a80bb", zorder=3)
        plt.fill_between(X, auc_mod_mean - auc_mod_std, auc_mod_mean + auc_mod_std,
                         alpha=0.3, color="#8cc5e3", label="Confidence interval")

        multimodal_better = None

        if arch_name == "MLEF":
            auc_ro_mean = np.array([r["auc_ro_mean"] for r in rows], dtype=float)
            auc_ro_std  = np.array([r["auc_ro_std"]  for r in rows], dtype=float)
            plt.plot(X, auc_ro_mean, linestyle="-", marker="o", label="CV AUC - RWD-only matched", color="#a00000")
            plt.scatter(X, auc_ro_mean, s=sizes, color="#a00000", zorder=3)
            plt.fill_between(X, auc_ro_mean - auc_ro_std, auc_ro_mean + auc_ro_std,
                             alpha=0.3, color="#d8a6a6", label="Confidence interval - RWD-only matched")
            multimodal_better = (auc_mod_mean > auc_ro_mean)

        xticks = [f"{m}\n(n. {int(n) if np.isfinite(n) else 'NA'})" for m, n in zip(modalities, n_train_mod)]
        plt.xticks(X, xticks)

        # --- BLUE annotations (now show stars here) ---
        for i, (mval, mstd, mname) in enumerate(zip(auc_mod_mean, auc_mod_std, model_names)):
            base_y = 0.06 if (multimodal_better is not None and multimodal_better[i]) else 0.02
            plt.text(i, base_y, f"{mval:.2f} ± {mstd:.2f} ({mname})",
                    fontsize=9, ha="center", color="#1a80bb")
            # stars belong to the multimodal-vs-RWD comparison
            star = rows[i].get("stars", "")
            if star:
                # nudge a bit to the right of the blue text
                plt.text(i + 0.33, base_y + 0.006, star, fontsize=10, ha="left", va="center", color="black")


        # --- RED annotations (keep values but REMOVE stars here) ---
        if arch_name == "MLEF":
            for i, rrow in enumerate(rows):
                rv, rs, rname = rrow["auc_ro_mean"], rrow["auc_ro_std"], rrow["model_ro"]
                base_y = 0.02 if multimodal_better[i] else 0.06
                if np.isfinite(rv) and np.isfinite(rs):
                    plt.text(i, base_y, f"{rv:.2f} ± {rs:.2f} ({rname})",
                            fontsize=9, ha="center", color="#a00000")

        ttl = title_prefix or f"CV AUC - {arch_name}"
        plt.title(f"{ttl} - {outcome} {analysis}", pad=18)
        plt.ylabel("AUC")
        plt.ylim(0, 1)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.xlim(-0.4, len(modalities) - 0.55)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out = Path(save_dir) / f"{ttl.replace(' ', '_')}_{analysis}.png"
            plt.savefig(out, dpi=600, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ------------------------- main logic -------------------------
    if isinstance(analyses, str):
        analyses = [analyses]
    
    # Normalize architecture to both string name and Path
    if isinstance(architecture, Path):
        arch_path = architecture
        arch_name = architecture.name  # Get the last part of the path (e.g., "MLEF" or "DLIF")
    else:
        arch_path = Path(architecture)
        arch_name = architecture

    # Convert outcome to DLIF format if needed
    def _outcome_to_dlif(outcome_str: str) -> str:
        """Convert MLEF outcome names to DLIF format."""
        mapping = {
            'OS_24': 'os_months_24',
            'OS_6': 'os_months_6',
            'DCR': 'DCR'
        }
        return mapping.get(outcome_str, outcome_str)

    for analysis in analyses:
        # Build the correct path based on architecture
        if architecture == "MLEF":
            # MLEF: mlef_pipeline/results/outcome/analysis/ 
            base_path = Path("mlef_pipeline/results") / outcome / analysis
            analysis_dir = base_path
        else:  # DLIF
            # DLIF: dlif_pipeline/results/analysis/outcome/classification/eval_type/feature_type/extraction/
            if dlif_base_path is None:
                dlif_base_path = Path("dlif_pipeline/results")
            else:
                dlif_base_path = Path(dlif_base_path)

            dlif_outcome = _outcome_to_dlif(outcome)
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
            modalities = [_map_dlif_to_mlef_modality(m) for m in dlif_modalities]
            modalities = _ordered_modalities(modalities)
        else:
            modalities = _collect_modalities(analysis_dir)

        if not modalities:
            continue

        rows: List[dict] = []
        for mod in modalities:
            # Get paths based on architecture
            if architecture == "DLIF":
                paths = _pair_paths_dlif(analysis_dir, mod)
                # Read DLIF-specific files
                auc_m, std_m = _read_auc_dlif(paths["mod"]["results"])
                model_m = "MIL"  # DLIF uses MIL models
                ntrain_m = _read_n_train_dlif(paths["mod"]["train"])
            else:
                paths = _pair_paths(analysis_dir, mod)
                # modality side
                auc_m, std_m = _read_auc_cv(paths["mod"]["results"])
                model_m = _read_model_name(paths["mod"]["model"])
                ntrain_m = _read_n_train(paths["mod"]["train"])

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
                            "n_train_ro":  _read_n_train(ro["train"]),
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
    fontsize = 12
    # --- Plotting Loop ---
    for metric in metrics:
        plt.figure(figsize=(8, 8))
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
        ax.tick_params(axis='x', pad=42)
        
        ax.set_ylim(0, 1)
        ax.set_title(f'{title_prefix} - {metric} Comparison', pad=25, fontsize=14)
        
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
                ax.text(angle, 1.14, model_val_text, color=colors[1], ha=ha, va='bottom', fontsize=fontsize)
                ax.text(angle, 1.0, bio_val_text, color=colors[0], ha=ha, va='bottom', fontsize=fontsize)
                if vals_dl is not None:
                    ax.text(angle, 1.07, dl_val_text, color=colors[2], ha=ha, va='bottom', fontsize=fontsize)
            elif 265 < angle_deg < 275: # Bottom
                ha = 'center'
                ax.text(angle, 1.01, model_val_text, color=colors[1], ha=ha, va='top', fontsize=fontsize)
                ax.text(angle, 1.16, bio_val_text, color=colors[0], ha=ha, va='top', fontsize=fontsize)
                if vals_dl is not None:
                    ax.text(angle, 1.08, dl_val_text, color=colors[2], ha=ha, va='top', fontsize=fontsize)
            elif angle_deg < 15 or angle_deg > 345: # Right
                ha = 'left'
                ax.text(angle - 0.16, 1.1, model_val_text, color=colors[1], ha=ha, va='bottom', fontsize=fontsize)
                ax.text(angle - 0.22, 1.1, bio_val_text, color=colors[0], ha=ha, va='top', fontsize=fontsize)
                if vals_dl is not None:
                    ax.text(angle - 0.16, 1.11, dl_val_text, color=colors[2], ha=ha, va='top', fontsize=fontsize)
            else: # Left
                ha = 'right'
                ax.text(angle + 0.16, 1.1, model_val_text, color=colors[1], ha=ha, va='bottom', fontsize=fontsize)
                ax.text(angle + 0.22, 1.1, bio_val_text, color=colors[0], ha=ha, va='top', fontsize=fontsize)
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
                0.2, 0.9,
                significance_text,
                ha="right", va="top",
                fontsize=9,
                color="#474545",
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
            )

        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=fontsize)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.2)
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