import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Literal, Union, Iterable, Tuple
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sksurv.metrics import concordance_index_censored
import joblib
import warnings
from itertools import combinations
from scipy.stats import norm
from .dlif_helper import *
warnings.filterwarnings('ignore')


def bootstrap_pvalue_cindex(event, time, risk1, risk2, n_boot=1000, seed=42) -> float:
    rng = np.random.default_rng(seed)
    event, time, r1, r2 = [np.asarray(x) for x in [event, time, risk1, risk2]]
    valid = ~np.isnan(r1) & ~np.isnan(r2)
    event, time, r1, r2 = event[valid], time[valid], r1[valid], r2[valid]
    
    diffs = [concordance_index_censored(event[idx], time[idx], r1[idx])[0] - 
             concordance_index_censored(event[idx], time[idx], r2[idx])[0]
             for idx in [rng.integers(0, len(time), size=len(time)) for _ in range(n_boot)]]
    
    return float(np.clip(2.0 * min(np.mean(np.array(diffs) >= 0.0), np.mean(np.array(diffs) <= 0.0)), 0.0, 1.0))

def p_to_stars(p: float) -> str:
    if p <= 1e-4: return "****"
    if p <= 1e-3: return "***"
    if p <= 1e-2: return "**"
    if p <= 5e-2: return "*"
    return "ns"

def plot_cindex_results(
    architecture: Literal["MLEF", "DLIF"],
    outcome: str,
    analyses: Union[str, Iterable[str]],
    *,
    modality_order: Optional[List[str]] = None,
    exclude_modalities: Iterable[str] = ("DP", "FMRAD", "PYRAD"),
    title_prefix: Optional[str] = None,
    min_bubble: float = 30,
    max_bubble: float = 250,
    n_boot: int = 1000,
    show: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
    dlif_base_path: Optional[Union[str, Path]] = None,
    dlif_eval_type: str = "standard",
    dlif_feature_type: str = "hypothesis_driven",
    dlif_extraction: str = "pyrad-noimp",
    dlif_seed: int = 0,
    use_preds: bool = True,
    save_name: Optional[str] = None,
):
    """
    Expected per-analysis layout (e.g. 'C23'):
        MLEF/ (or DLIF/)
         OS_24/ (or DCR/ OR OS_6/)
          C23/
            CB/
              model_XX.pkl
              train_set.xlsx
              test_set.xlsx
              prediction_CV.xlsx   (columns: Subject, y_pred, EVENT, TIME)
              results.xlsx      (metrics incl. C-INDEX CV)
            CB_DP/
               CB_ONLY/         (MLEF only)
            CB_PYRAD/
            ...
    """

    def _parse_mean_std(val) -> Tuple[float, float]:
        if pd.isna(val):
            return (np.nan, np.nan)
        if isinstance(val, (int, float, np.floating)):
            return (float(val), 0.0)
        s = str(val).replace("+/-", "±")
        parts = [p.strip() for p in s.split("±")]
        try:
            return (float(parts[0]), float(parts[1]) if len(parts) == 2 else 0.0)
        except:
            return (np.nan, np.nan)

    def _read_cindex_cv(results_path: Path) -> Tuple[float, float]:
        if not results_path or not results_path.exists():
            return (np.nan, np.nan)
        df = pd.read_excel(results_path)
        
        for col in df.columns:
            if str(col).strip().upper() in ("C-INDEX CV", "C_INDEX_CV", "CINDEX_CV", "CV C-INDEX"):
                return _parse_mean_std(df[col].iloc[0])
        
        set_cols = [c for c in df.columns if str(c).strip().upper() in ("SET", "SPLIT", "PHASE")]
        cindex_cols = [c for c in df.columns if str(c).strip().upper() in ("C-INDEX", "CINDEX", "C_INDEX")]
        if set_cols and cindex_cols:
            mask = df[set_cols[0]].astype(str).str.upper().str.contains("CV")
            sub = df[mask]
            if not sub.empty:
                return _parse_mean_std(sub[cindex_cols[0]].iloc[0])
        
        if cindex_cols:
            return _parse_mean_std(df[cindex_cols[0]].iloc[0])
        return (np.nan, np.nan)

    def _read_model_name(model_path: Path) -> str:
        if not model_path or not model_path.exists():
            return "UnknownModel"
        stem = model_path.name.split(".", 1)[0] if "." in model_path.name else model_path.name
        return stem.split("_", 1)[1] if "_" in stem else "UnknownModel"

    def _read_n_train(train_path: Path) -> int:
        if not train_path or not train_path.exists():
            return int(np.nan)
        try:
            return int(pd.read_excel(train_path).shape[0])
        except:
            return int(np.nan)

    def _read_cindex_dlif(auc_csv_path: Path) -> Tuple[float, float]:
        """Read C-index from DLIF's eval_auc_ci.csv file."""
        if auc_csv_path is None or not auc_csv_path.exists():
            return (np.nan, np.nan)
        try:
            df = pd.read_csv(auc_csv_path)
            if 'auc' in df.columns:
                cindex = float(df['auc'].iloc[0])
                if 'ci_lower' in df.columns and 'ci_upper' in df.columns:
                    ci_lower = float(df['ci_lower'].iloc[0])
                    ci_upper = float(df['ci_upper'].iloc[0])
                    std = (ci_upper - ci_lower) / 2
                    return (cindex, std)
                return (cindex, 0.0)
        except Exception:
            pass
        return (np.nan, np.nan)

    def _pair_paths_dlif(base_dir: Path, modality: str) -> dict:
        """Path resolver for DLIF architecture."""
        dlif_modality = map_mlef_to_dlif_modality(modality)

        if use_preds:
            mod_dir = base_dir / dlif_modality
        else:
            mod_dir = base_dir / dlif_modality / f"seed_{dlif_seed}"

        if not mod_dir.exists():
            return {
                "mod": {"results": None, "pred": None, "model": None, "train": None},
                "cb_only": None
            }

        # Find eval predictions based on evaluation type
        if dlif_eval_type == "cross_validation":
            pred_fold_paths = sorted(mod_dir.glob("fold_*/eval/predictions.parquet"))
            pred_path = pred_fold_paths if pred_fold_paths else None
        else:
            pred_path = mod_dir / "predictions.parquet"
            if not pred_path.exists():
                pred_path = next(mod_dir.glob("eval/*/predictions.parquet"), None)

        paths = {
            "mod": {
                "results": mod_dir / "eval_auc_ci.csv" if (mod_dir / "eval_auc_ci.csv").exists() else None,
                "pred": pred_path,
                "model": None,
                "train": None,
            },
            "cb_only": None
        }

        return paths

    def _scale_sizes(raw_sizes: Iterable[Union[int, float]]) -> np.ndarray:
        arr = np.array(list(raw_sizes), dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return np.array([])
        amin, amax = np.nanmin(arr), np.nanmax(arr)
        if not np.isfinite(amin) or not np.isfinite(amax) or (amax - amin < 1e-9):
            return np.full_like(arr, (min_bubble + max_bubble) / 2.0)
        return ((arr - amin) / (amax - amin + 1e-6)) * (max_bubble - min_bubble) + min_bubble

    def _ordered_modalities(mods: List[str]) -> List[str]:
        filtered = [m for m in mods if m not in exclude_modalities]
        if modality_order:
            in_order = [m for m in modality_order if m in filtered]
            leftovers = [m for m in filtered if m not in in_order]
            return in_order + leftovers
        return filtered

    def _collect_modalities(analysis_dir: Path) -> List[str]:
        return _ordered_modalities([p.name for p in analysis_dir.iterdir()
                                    if p.is_dir() and p.name.upper().startswith("CB")])

    def _find_first(dir_: Path, stems: List[str]) -> Optional[Path]:
        if not dir_.exists():
            return None
        for st in stems:
            for ext in ("", ".xlsx", ".xls"):
                p = dir_ / f"{st}{ext}"
                if p.exists():
                    return p
        cand = []
        for st in stems:
            cand += list(dir_.glob(f"{st}*")) + list(dir_.glob(f"{st.upper()}*")) + list(dir_.glob(f"{st.lower()}*"))
        cand = sorted(cand, key=lambda x: (x.suffix.lower() not in {".xlsx", ".xls"}, len(x.name)))
        return cand[0] if cand else None

    def _find_subdir(dir_: Path, names: List[str]) -> Optional[Path]:
        if not dir_.exists():
            return None
        for n in names:
            if (p := dir_ / n).exists() and p.is_dir():
                return p
        low_targets = {n.lower(): n for n in names}
        for p in dir_.iterdir():
            if p.is_dir() and p.name.lower() in low_targets:
                return p
        return None

    def _pair_paths(analysis_dir: Path, modality: str):
        mod_dir = analysis_dir / modality
        paths = {
            "mod": {
                "results": _find_first(mod_dir, ["results", "Results"]),
                "pred": _find_first(mod_dir, ["prediction_CV", "prediction", "Prediction"]),
                "model": _find_first(mod_dir, ["model_", "model"]),
                "train": _find_first(mod_dir, ["train_set", "Train_set", "train"]),
            },
            "cb_only": None
        }
        
        if architecture == "MLEF":
            if (ro_dir := _find_subdir(mod_dir, ["CB_ONLY", "rwd-only", "RWD_only", "CB-ONLY"])):
                paths["cb_only"] = {
                    "results": _find_first(ro_dir, ["results", "Results"]),
                    "pred": _find_first(ro_dir, ["prediction_CV", "prediction", "Prediction"]),
                    "model": _find_first(ro_dir, ["model_", "model"]),
                    "train": _find_first(ro_dir, ["train_set", "Train_set", "train"]),
                }
        return paths

    def _compute_pvalue(pred_mod_path: Path, pred_ro_path: Path) -> Optional[float]:
        if not (pred_mod_path and pred_ro_path and pred_mod_path.exists() and pred_ro_path.exists()):
            return None
        dm = pd.read_excel(pred_mod_path)
        dr = pd.read_excel(pred_ro_path)
        
        for col in ("Subject", "y_pred", "EVENT", "TIME"):
            if col not in dm.columns:
                return None
        if "Subject" not in dr.columns or "y_pred" not in dr.columns:
            return None
        
        merged = pd.merge(
            dm[["Subject", "y_pred", "EVENT", "TIME"]].rename(columns={"y_pred": "y_pred_mod"}),
            dr[["Subject", "y_pred"]].rename(columns={"y_pred": "y_pred_ro"}),
            on="Subject", how="inner"
        )
        if merged.empty:
            return None
        
        return bootstrap_pvalue_cindex(
            merged["EVENT"].astype(bool).to_numpy(),
            merged["TIME"].astype(float).to_numpy(),
            merged["y_pred_mod"].to_numpy(),
            merged["y_pred_ro"].to_numpy(),
            n_boot=n_boot
        )

    def _plot_one(analysis: str, rows: List[dict]) -> plt.Figure:
        modalities = [r["modality"] for r in rows]
        X = np.arange(len(modalities))

        cindex_mod_mean = np.array([r["cindex_mod_mean"] for r in rows], dtype=float)
        cindex_mod_std = np.array([r["cindex_mod_std"] for r in rows], dtype=float)
        n_train_mod = np.array([r["n_train_mod"] for r in rows], dtype=float)
        sizes = _scale_sizes(n_train_mod)
        model_names = [r["model_mod"] for r in rows]

        fig = plt.figure(figsize=(12, 6))

        if architecture == "DLIF":
            label = "TEST C-INDEX"
        else:
            label = "C-INDEX CV"
        plt.plot(X, cindex_mod_mean, linestyle="-", marker="o", label=label, color="#1a80bb")
        if len(sizes) > 0:
            plt.scatter(X, cindex_mod_mean, s=sizes, color="#1a80bb", zorder=3)
        else:
            plt.scatter(X, cindex_mod_mean, s=100, color="#1a80bb", zorder=3)
        plt.fill_between(X, cindex_mod_mean - cindex_mod_std, cindex_mod_mean + cindex_mod_std,
                         alpha=0.3, color="#8cc5e3", label="Confidence interval")

        multimodal_better = None

        if architecture == "MLEF":
            cindex_ro_mean = np.array([r["cindex_ro_mean"] for r in rows], dtype=float)
            cindex_ro_std = np.array([r["cindex_ro_std"] for r in rows], dtype=float)
            plt.plot(X, cindex_ro_mean, linestyle="-", marker="o", label="C-INDEX CV - CB-only matched", color="#a00000")
            if len(sizes) > 0:
                plt.scatter(X, cindex_ro_mean, s=sizes, color="#a00000", zorder=3)
            else:
                plt.scatter(X, cindex_ro_mean, s=100, color="#a00000", zorder=3)
            plt.fill_between(X, cindex_ro_mean - cindex_ro_std, cindex_ro_mean + cindex_ro_std,
                             alpha=0.3, color="#d8a6a6", label="Confidence interval - CB-only matched")
            multimodal_better = (cindex_mod_mean > cindex_ro_mean)

        if architecture == "DLIF":
            xticks = []
            for m, n in zip(modalities, n_train_mod):
                parts = m.split('_')
                parts = ['CB' if p == 'cb' else p for p in parts]
                formatted_name = '\n'.join(parts)
                xticks.append(f"{formatted_name}\n(n. {int(n) if np.isfinite(n) else 'NA'})")
            plt.xticks(X, xticks, rotation=0, fontsize=12)
        else:
            xticks = [f"{m.replace('CB', 'CB')}\n(n. {int(n) if np.isfinite(n) else 'NA'})"
                    for m, n in zip(modalities, n_train_mod)]
            plt.xticks(X, xticks, rotation=0, fontsize=12)

        for i, (mval, mstd, mname) in enumerate(zip(cindex_mod_mean, cindex_mod_std, model_names)):
            base_y = 0.06 if (multimodal_better is not None and multimodal_better[i]) else 0.02
            if architecture == "DLIF":
                plt.text(i, base_y, f"{mval:.2f} ± {mstd:.2f}",
                        fontsize=11, ha="center", color="#1a80bb")
            else:
                plt.text(i, base_y, f"{mval:.2f} ± {mstd:.2f} ({mname})",
                        fontsize=11, ha="center", color="#1a80bb")
            if (star := rows[i].get("stars", "")):
                plt.text(i + 0.40, base_y + 0.006, star, fontsize=11, ha="left", va="center", color="black")

        if architecture == "MLEF":
            for i, rrow in enumerate(rows):
                rv, rs, rname = rrow["cindex_ro_mean"], rrow["cindex_ro_std"], rrow["model_ro"]
                base_y = 0.02 if multimodal_better[i] else 0.06
                if np.isfinite(rv) and np.isfinite(rs):
                    plt.text(i, base_y, f"{rv:.2f} ± {rs:.2f} ({rname})",
                            fontsize=11, ha="center", color="#a00000")

        ttl = title_prefix or f"C-INDEX CV - {architecture}"
        plt.title(f"{ttl} - {outcome} {analysis}", pad=18)
        plt.ylabel("C-INDEX")
        plt.ylim(0, 1)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=12)
        plt.xlim(-0.5, len(modalities) - 0.50)

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

    if isinstance(analyses, str):
        analyses = [analyses]

    for analysis in analyses:
        # Build the correct path based on architecture
        if architecture == "MLEF":
            analysis_dir = Path(architecture) / outcome / analysis
        else:  # DLIF
            if use_preds:
                if dlif_base_path is None:
                    _dlif_base = Path("dlif_pipeline/preds")
                else:
                    _dlif_base = Path(dlif_base_path)
                dlif_outcome = outcome_to_dlif(outcome)
                analysis_dir = _dlif_base / dlif_outcome / "classification" / dlif_eval_type
            else:
                if dlif_base_path is None:
                    _dlif_base = Path("dlif_pipeline/results")
                else:
                    _dlif_base = Path(dlif_base_path)
                dlif_outcome = outcome_to_dlif(outcome)
                analysis_dir = (_dlif_base / analysis / dlif_outcome / "classification" /
                              dlif_eval_type / dlif_feature_type / dlif_extraction)

        if not analysis_dir.exists():
            continue

        # Collect modalities
        if architecture == "DLIF":
            dlif_modalities = [p.name for p in analysis_dir.iterdir()
                             if p.is_dir() and p.name.lower().startswith("cb")]
            modalities = [map_dlif_to_mlef_modality(m) for m in dlif_modalities]
            modalities = _ordered_modalities(modalities)
        else:
            modalities = _collect_modalities(analysis_dir)

        if not modalities:
            continue

        rows = []
        for mod in modalities:
            if architecture == "DLIF":
                paths = _pair_paths_dlif(analysis_dir, mod)
                cindex_m, std_m = _read_cindex_dlif(paths["mod"]["results"])
                model_m = "DLIF"
                ntrain_m = read_n_train_dlif(paths["mod"]["pred"])
            else:
                paths = _pair_paths(analysis_dir, mod)
                cindex_m, std_m = _read_cindex_cv(paths["mod"]["results"])
                model_m = _read_model_name(paths["mod"]["model"])
                ntrain_m = _read_n_train(paths["mod"]["train"])
            
            row = {
                "modality": mod,
                "cindex_mod_mean": float(cindex_m),
                "cindex_mod_std": float(std_m) if np.isfinite(std_m) else 0.0,
                "n_train_mod": ntrain_m,
                "model_mod": model_m,
            }
            
            if architecture == "MLEF":
                if mod == "CB":
                    row.update({
                        "cindex_ro_mean": row["cindex_mod_mean"],
                        "cindex_ro_std": row["cindex_mod_std"],
                        "n_train_ro": row["n_train_mod"],
                        "model_ro": row["model_mod"],
                        "pvalue": None,
                        "stars": ""
                    })
                else:
                    if (ro := paths["cb_only"]):
                        cindex_r, std_r = _read_cindex_cv(ro["results"])
                        pval = _compute_pvalue(paths["mod"]["pred"], ro["pred"])
                        row.update({
                            "cindex_ro_mean": float(cindex_r),
                            "cindex_ro_std": float(std_r) if np.isfinite(std_r) else 0.0,
                            "n_train_ro": _read_n_train(ro["train"]),
                            "model_ro": _read_model_name(ro["model"]),
                            "pvalue": pval,
                            "stars": p_to_stars(pval) if (pval and np.isfinite(pval)) else ""
                        })
                    else:
                        row.update({
                            "cindex_ro_mean": np.nan,
                            "cindex_ro_std": np.nan,
                            "n_train_ro": np.nan,
                            "model_ro": "NA",
                            "pvalue": None,
                            "stars": ""
                        })
            
            rows.append(row)
        
        rows = [r for r in rows if np.isfinite(r["cindex_mod_mean"])]
        if rows:
            _plot_one(analysis, rows)

"""
fairness_utils.py - Helper functions for survival fairness analysis
"""

def load_predictions(file_path):
    """
    Load prediction file with Subject, EVENT, TIME, risk_score columns.
    
    Parameters:
    -----------
    file_path : Path or str
        Path to CSV file with predictions
    
    Returns:
    --------
    pd.DataFrame with required columns
    """
    df = pd.read_csv(file_path)
    required_cols = ['Subject', 'EVENT', 'TIME', 'risk_score']
    
    # Check for required columns
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df[required_cols]

def load_clinical_data(file_path):
    """
    Load clinical data with demographic information.
    
    Expected columns: Subject, SEX, RACE* columns
    """
    return pd.read_csv(file_path)

def compute_cindex(event, time, risk_score):
    """
    Compute concordance index for survival data.
    
    Parameters:
    -----------
    event : array-like
        Boolean or 0/1 indicating if event occurred
    time : array-like
        Survival or censoring time
    risk_score : array-like
        Predicted risk scores (higher = higher risk)
    
    Returns:
    --------
    float: C-index value
    """
    event = np.asarray(event, dtype=bool)
    time = np.asarray(time, dtype=float)
    risk_score = np.asarray(risk_score, dtype=float)
    
    # Remove NaN values
    valid_mask = ~np.isnan(risk_score)
    event = event[valid_mask]
    time = time[valid_mask]
    risk_score = risk_score[valid_mask]
    
    if len(time) == 0:
        return np.nan
    
    return concordance_index_censored(event, time, risk_score)[0]

def bootstrap_cindex(event, time, risk_score, n_boot=1000, alpha=0.05, seed=42):
    """
    Compute C-index with bootstrap confidence intervals.
    
    Parameters:
    -----------
    event, time, risk_score : array-like
        Survival data
    n_boot : int
        Number of bootstrap iterations
    alpha : float
        Significance level for CI
    seed : int
        Random seed
    
    Returns:
    --------
    dict with keys: c_index, ci_lower, ci_upper, ci_width, bootstrap_samples
    """
    np.random.seed(seed)
    
    event = np.asarray(event, dtype=bool)
    time = np.asarray(time, dtype=float)
    risk_score = np.asarray(risk_score, dtype=float)
    
    # Remove NaN values
    valid_mask = ~np.isnan(risk_score)
    event = event[valid_mask]
    time = time[valid_mask]
    risk_score = risk_score[valid_mask]
    
    n = len(time)
    c_indexes = []
    
    # Bootstrap
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        c_idx = compute_cindex(event[idx], time[idx], risk_score[idx])
        if not np.isnan(c_idx):
            c_indexes.append(c_idx)
    
    # Original C-index
    c_index_orig = compute_cindex(event, time, risk_score)
    
    # Confidence intervals
    ci_lower = np.percentile(c_indexes, 100 * (alpha / 2))
    ci_upper = np.percentile(c_indexes, 100 * (1 - alpha / 2))
    ci_width = (ci_upper - ci_lower) / 2
    
    return {
        "c_index": c_index_orig,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_width,
        "bootstrap_samples": np.array(c_indexes),
        "n": n
    }

def format_cindex(result):
    """
    Format C-index result as string: "0.XX ± 0.YY"
    """
    return f"{result['c_index']:.2f} ± {result['ci_width']:.2f}"

def holm_adjust(pvals):
    """
    Holm-Bonferroni correction for multiple testing.
    """
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    adj_sorted = (m - np.arange(m)) * p_sorted
    
    # Enforce monotone non-decreasing
    for i in range(1, m):
        adj_sorted[i] = max(adj_sorted[i - 1], adj_sorted[i])
    
    adj = np.minimum(adj_sorted, 1.0)
    out = np.empty_like(p)
    out[order] = adj
    return out

def pairwise_comparison(group_results):
    """
    Perform pairwise comparisons between groups using bootstrap samples.
    
    Parameters:
    -----------
    group_results : dict
        Dictionary with group names as keys and bootstrap results as values
    
    Returns:
    --------
    pd.DataFrame with pairwise comparison results
    """
    rows, pvals = [], []
    pairs = list(combinations(group_results.keys(), 2))
    
    for g1, g2 in pairs:
        G1, G2 = group_results[g1], group_results[g2]
        c1, c2 = G1["c_index"], G2["c_index"]
        diff_hat = c1 - c2
        
        # Bootstrap difference
        b1 = G1["bootstrap_samples"]
        b2 = G2["bootstrap_samples"]
        
        # Use minimum length for paired comparison
        min_len = min(len(b1), len(b2))
        boot_diff = b1[:min_len] - b2[:min_len]
        
        if len(boot_diff) >= 20:
            se_diff = boot_diff.std(ddof=1)
            z = diff_hat / se_diff if se_diff > 0 else np.nan
            p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
            ci_diff = tuple(np.percentile(boot_diff, [2.5, 97.5]))
        else:
            se_diff, z, p, ci_diff = np.nan, np.nan, np.nan, (np.nan, np.nan)
        
        rows.append({
            "g1": g1, "g2": g2,
            "n1": G1["n"], "n2": G2["n"],
            "c1": c1, "c1_lo": G1["ci_lower"], "c1_hi": G1["ci_upper"],
            "c2": c2, "c2_lo": G2["ci_lower"], "c2_hi": G2["ci_upper"],
            "diff": diff_hat, "diff_lo": ci_diff[0], "diff_hi": ci_diff[1],
            "z": z, "p": p
        })
        pvals.append(p)
    
    out = pd.DataFrame(rows)
    if len(pairs) > 0:
        out["p_holm"] = holm_adjust([pv if np.isfinite(pv) else 1.0 for pv in pvals])
    
    return out.sort_values(["g1", "g2"]).reset_index(drop=True)

def analyze_fairness_by_site(predictions, sites, n_boot=1000, seed=42):
    """
    Analyze C-index fairness across medical centers.
    
    Parameters:
    -----------
    predictions : pd.DataFrame
        DataFrame with Subject, CENTER, EVENT, TIME, risk_score columns
    sites : list
        List of site prefixes (e.g., ['INT', 'GHD', ...])
    
    Returns:
    --------
    tuple: (per_group_df, pairwise_df, overall_cindex)
    """
    print("\n" + "="*60)
    print("C-INDEX BY MEDICAL CENTER (Test Set)")
    print("="*60)
    
    group_results = {}
    
    for site in sites:
        site_data = predictions[predictions['CENTER']==site]
        
        if len(site_data) == 0:
            print(f"\nWarning: No patients found for site {site}")
            continue
        
        result = bootstrap_cindex(
            site_data['EVENT'],
            site_data['TIME'],
            site_data['risk_score'],
            n_boot=n_boot,
            seed=seed
        )
        group_results[site] = result
        
        print(f"\n{site:8s} (n={result['n']:3d}): {format_cindex(result)}")
    
    # Overall
    overall = bootstrap_cindex(
        predictions['EVENT'],
        predictions['TIME'],
        predictions['risk_score'],
        n_boot=n_boot,
        seed=seed
    )
    print(f"\nOverall  (n={overall['n']:3d}): {format_cindex(overall)}")
    
    # Summary DataFrame
    per_group = pd.DataFrame([{
        "group": k,
        "n": v["n"],
        "c": v["c_index"],
        "ci_lo": v["ci_lower"],
        "ci_hi": v["ci_upper"]
    } for k, v in group_results.items()]).sort_values("group").reset_index(drop=True)
    
    # Pairwise comparisons
    pairwise = pairwise_comparison(group_results)
    
    print("\n" + "-"*60)
    print("Pairwise Comparisons (Holm-corrected):")
    print("-"*60)
    for _, row in pairwise.iterrows():
        sig = "***" if row['p_holm'] < 0.001 else "**" if row['p_holm'] < 0.01 else "*" if row['p_holm'] < 0.05 else "ns"
        print(f"{row['g1']} vs {row['g2']:8s}: diff={row['diff']:+.3f}, p={row['p']:.4f}, p_holm={row['p_holm']:.4f} {sig}")
    
    return per_group, pairwise, overall['c_index']

def analyze_fairness_by_race(predictions, clinical_data, n_boot=1000, seed=42):
    """
    Analyze C-index fairness across race groups.
    
    Parameters:
    -----------
    predictions : pd.DataFrame
        DataFrame with Subject, EVENT, TIME, risk_score columns
    clinical_data : pd.DataFrame
        DataFrame with Subject and RACE column
    
    Returns:
    --------
    tuple: (per_group_df, pairwise_df, overall_cindex)
    """
    print("\n" + "="*60)
    print("C-INDEX BY RACE (External Validation)")
    print("="*60)
    
    if 'RACE' not in clinical_data.columns:
        print("\nWarning: RACE column not found in clinical data")
        return pd.DataFrame(), pd.DataFrame(), None
    
    # Merge predictions with clinical data to get race information
    merged = predictions.merge(clinical_data[['Subject', 'RACE']], on='Subject', how='inner')
    
    race_labels = ['WHITE', 'BLACK OR AFRICAN AMERICAN']
    group_results = {}
    
    for label in race_labels:
        race_data = merged[merged['RACE'] == label]
        
        if len(race_data) == 0:
            print(f"\nWarning: No patients found for race {label}")
            continue
        
        result = bootstrap_cindex(
            race_data['EVENT'],
            race_data['TIME'],
            race_data['risk_score'],
            n_boot=n_boot,
            seed=seed
        )
        group_results[label] = result
        
        print(f"\n{label:30s} (n={result['n']:3d}): {format_cindex(result)}")
    
    # Overall
    overall = bootstrap_cindex(
        predictions['EVENT'],
        predictions['TIME'],
        predictions['risk_score'],
        n_boot=n_boot,
        seed=seed
    )
    print(f"\nOverall (n={overall['n']:3d}): {format_cindex(overall)}")
    
    # Summary DataFrame
    per_group = pd.DataFrame([{
        "group": k,
        "n": v["n"],
        "c": v["c_index"],
        "ci_lo": v["ci_lower"],
        "ci_hi": v["ci_upper"]
    } for k, v in group_results.items()]).sort_values("group").reset_index(drop=True)
    
    # Pairwise comparisons
    pairwise = pairwise_comparison(group_results)
    
    if len(pairwise) > 0:
        print("\n" + "-"*60)
        print("Pairwise Comparisons (Holm-corrected):")
        print("-"*60)
        for _, row in pairwise.iterrows():
            sig = "***" if row['p_holm'] < 0.001 else "**" if row['p_holm'] < 0.01 else "*" if row['p_holm'] < 0.05 else "ns"
            print(f"{row['g1']} vs {row['g2']}: diff={row['diff']:+.3f}, p={row['p']:.4f}, p_holm={row['p_holm']:.4f} {sig}")
    
    return per_group, pairwise, overall['c_index']

def analyze_fairness_by_sex(predictions, clinical_data, set_name='test', n_boot=1000, seed=42):
    """
    Analyze C-index fairness across sex groups.
    
    Parameters:
    -----------
    predictions : pd.DataFrame
        DataFrame with Subject, EVENT, TIME, risk_score columns
    clinical_data : pd.DataFrame
        DataFrame with Subject and SEX column
    set_name : str
        Name of the dataset for display
    
    Returns:
    --------
    tuple: (per_group_df, pairwise_df, overall_cindex)
    """
    print("\n" + "="*60)
    print(f"C-INDEX BY SEX ({set_name.upper()} Set)")
    print("="*60)
    
    if 'SEX' not in clinical_data.columns:
        print("\nWarning: SEX column not found in clinical data")
        return pd.DataFrame(), pd.DataFrame(), None
    
    group_results = {}
    
    for sex_num, sex_str in [(0, 'Female'), (1, 'Male')]:
        subjects = clinical_data.loc[clinical_data['SEX'] == sex_num, 'Subject']
        sex_data = predictions[predictions['Subject'].isin(subjects)]
        
        if len(sex_data) == 0:
            print(f"\nWarning: No patients found for sex {sex_str}")
            continue
        
        result = bootstrap_cindex(
            sex_data['EVENT'],
            sex_data['TIME'],
            sex_data['risk_score'],
            n_boot=n_boot,
            seed=seed
        )
        group_results[sex_str] = result
        
        print(f"\n{sex_str:8s} (n={result['n']:3d}): {format_cindex(result)}")
    
    # Overall
    overall = bootstrap_cindex(
        predictions['EVENT'],
        predictions['TIME'],
        predictions['risk_score'],
        n_boot=n_boot,
        seed=seed
    )
    print(f"\nOverall  (n={overall['n']:3d}): {format_cindex(overall)}")
    
    # Summary DataFrame
    per_group = pd.DataFrame([{
        "group": k,
        "n": v["n"],
        "c": v["c_index"],
        "ci_lo": v["ci_lower"],
        "ci_hi": v["ci_upper"]
    } for k, v in group_results.items()]).sort_values("group").reset_index(drop=True)
    
    # Pairwise comparisons
    pairwise = pairwise_comparison(group_results)
    
    if len(pairwise) > 0:
        print("\n" + "-"*60)
        print("Pairwise Comparisons (Holm-corrected):")
        print("-"*60)
        for _, row in pairwise.iterrows():
            sig = "***" if row['p_holm'] < 0.001 else "**" if row['p_holm'] < 0.01 else "*" if row['p_holm'] < 0.05 else "ns"
            print(f"{row['g1']} vs {row['g2']:8s}: diff={row['diff']:+.3f}, p={row['p']:.4f}, p_holm={row['p_holm']:.4f} {sig}")
    
    return per_group, pairwise, overall['c_index']


def plot_fairness_results(per_group_df, title, figsize=(8, 5), palette=None, group_order=None, overall_cindex=None):
    """
    Create bar plot of C-index by group with error bars.
    
    Parameters:
    -----------
    per_group_df : pd.DataFrame
        DataFrame with columns: group, n, c, ci_lo, ci_hi
    title : str
        Plot title
    group_order : list, optional
        Order of groups to display on x-axis
    overall_cindex : float, optional
        Overall C-index to plot as horizontal reference line
    """
    if len(per_group_df) == 0:
        print(f"No data to plot for {title}")
        return
    
    df = per_group_df.copy()
    df['ci_width'] = (df['ci_hi'] - df['ci_lo']) / 2
    
    # Apply group order if specified
    if group_order is not None:
        # Filter to only include groups that exist in the data
        existing_groups = df['group'].tolist()
        group_order = [g for g in group_order if g in existing_groups]
        df['group'] = pd.Categorical(df['group'], categories=group_order, ordered=True)
        df = df.sort_values('group')
    
    if palette is None:
        palette = sns.blend_palette(["#1E52A0", '#4292C6'], n_colors=len(df))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Bar plot
    barplot = sns.barplot(
        x="group", 
        y="c", 
        data=df,
        hue="group",
        palette=palette,
        errorbar=None,
        ax=ax,
        order=group_order if group_order is not None else None
    )
    
    # Remove bar borders
    for p in barplot.patches:
        p.set_edgecolor('none')
        p.set_linewidth(0)
    barplot.xaxis.grid(False)
    
    # Add horizontal line for overall C-index
    if overall_cindex is not None:
        ax.axhline(y=overall_cindex, color='gray', linestyle='--', linewidth=2, 
                   label=f'Overall C-index: {overall_cindex:.2f}', alpha=0.7)
        ax.legend(loc='lower right')
    
    # Add error bars manually
    for i, row in enumerate(df.itertuples(index=False)):
        ax.errorbar(i, row.c, yerr=row.ci_width, color='black', capsize=5, fmt='none')
    
    # Add value annotations
    for p, val, ci_width, n in zip(barplot.patches, df["c"], df["ci_width"], df['n']):
        y = val + ci_width + 0.01
        barplot.annotate(
            f"{val:.2f} ± {ci_width:.2f}\n(n={n})",
            (p.get_x() + p.get_width() / 2., y),
            ha='center', va='bottom',
            fontsize=10, color='black'
        )
    
    ax.set(xlabel=None, ylabel='C-index', title=title)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig
