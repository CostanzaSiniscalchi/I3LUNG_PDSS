import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Literal, Union, Iterable, Tuple
from pathlib import Path
import os

def bootstrap_pvalue_cindex(event, time, risk1, risk2, n_boot=1000, seed=42) -> float:
    from sksurv.metrics import concordance_index_censored
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
    save_dir: Optional[Union[str, Path]] = None
):
    """
    Expected per-analysis layout (e.g. 'C23'):
        MLEF/ (or DLIF/)
         OS_24/ (or DCR/ OR OS_6/)
          C23/
            RWD/
              model_XX.pkl
              train_set.xlsx
              test_set.xlsx
              prediction_CV.xlsx   (columns: Subject, y_pred, EVENT, TIME)
              results.xlsx      (metrics incl. C-INDEX CV)
            RWD_DP/
               RWD_ONLY/         (MLEF only)
            RWD_PYRAD/
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
                                    if p.is_dir() and p.name.upper().startswith("RWD")])

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
            "rwd_only": None
        }
        
        if architecture == "MLEF":
            if (ro_dir := _find_subdir(mod_dir, ["RWD_ONLY", "rwd-only", "Rwd_only", "RWD-ONLY"])):
                paths["rwd_only"] = {
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
        
        plt.plot(X, cindex_mod_mean, linestyle="-", marker="o", label="C-INDEX CV", color="#1a80bb")
        plt.scatter(X, cindex_mod_mean, s=sizes, color="#1a80bb", zorder=3)
        plt.fill_between(X, cindex_mod_mean - cindex_mod_std, cindex_mod_mean + cindex_mod_std,
                         alpha=0.3, color="#8cc5e3", label="Confidence interval")
        
        multimodal_better = None
        
        if architecture == "MLEF":
            cindex_ro_mean = np.array([r["cindex_ro_mean"] for r in rows], dtype=float)
            cindex_ro_std = np.array([r["cindex_ro_std"] for r in rows], dtype=float)
            plt.plot(X, cindex_ro_mean, linestyle="-", marker="o", label="C-INDEX CV - RWD-only matched", color="#a00000")
            plt.scatter(X, cindex_ro_mean, s=sizes, color="#a00000", zorder=3)
            plt.fill_between(X, cindex_ro_mean - cindex_ro_std, cindex_ro_mean + cindex_ro_std,
                             alpha=0.3, color="#d8a6a6", label="Confidence interval - RWD-only matched")
            multimodal_better = (cindex_mod_mean > cindex_ro_mean)
        
        xticks = [f"{m}\n(n. {int(n) if np.isfinite(n) else 'NA'})" for m, n in zip(modalities, n_train_mod)]
        plt.xticks(X, xticks)
        
        for i, (mval, mstd, mname) in enumerate(zip(cindex_mod_mean, cindex_mod_std, model_names)):
            base_y = 0.06 if (multimodal_better is not None and multimodal_better[i]) else 0.02
            plt.text(i, base_y, f"{mval:.2f} ± {mstd:.2f} ({mname})",
                    fontsize=9, ha="center", color="#1a80bb")
            if (star := rows[i].get("stars", "")):
                plt.text(i + 0.33, base_y + 0.006, star, fontsize=10, ha="left", va="center", color="black")
        
        if architecture == "MLEF":
            for i, rrow in enumerate(rows):
                rv, rs, rname = rrow["cindex_ro_mean"], rrow["cindex_ro_std"], rrow["model_ro"]
                base_y = 0.02 if multimodal_better[i] else 0.06
                if np.isfinite(rv) and np.isfinite(rs):
                    plt.text(i, base_y, f"{rv:.2f} ± {rs:.2f} ({rname})",
                            fontsize=9, ha="center", color="#a00000")
        
        ttl = title_prefix or f"C-INDEX CV - {architecture}"
        plt.title(f"{ttl} - {outcome} {analysis}", pad=18)
        plt.ylabel("C-INDEX")
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

    if isinstance(analyses, str):
        analyses = [analyses]
    
    for analysis in analyses:
        analysis_dir = Path(architecture) / outcome / analysis
        if not analysis_dir.exists():
            continue
        
        modalities = _collect_modalities(analysis_dir)
        if not modalities:
            continue
        
        rows = []
        for mod in modalities:
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
                if mod == "RWD":
                    row.update({
                        "cindex_ro_mean": row["cindex_mod_mean"],
                        "cindex_ro_std": row["cindex_mod_std"],
                        "n_train_ro": row["n_train_mod"],
                        "model_ro": row["model_mod"],
                        "pvalue": None,
                        "stars": ""
                    })
                else:
                    if (ro := paths["rwd_only"]):
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