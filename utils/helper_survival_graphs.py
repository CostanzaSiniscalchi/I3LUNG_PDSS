import itertools
import os
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test
from lifelines.utils import median_survival_times
from lifelines.plotting import add_at_risk_counts
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
import shap
from utils.DeLong_test import *
from .dlif_helper import *
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored
from itertools import combinations
from scipy.stats import norm
import seaborn as sns
from sklearn.linear_model import LogisticRegression


def plot_km_combined(datasets, stats: bool=True):
    """
    Curve KM stratificate, con test globale + pairwise, box di annotazione
    con bordo centrato, padding extra attorno al testo, box e testo sollevati,
    e legenda in alto a destra con font ridotto.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300, facecolor='white')
    colors = ["#5AAA46", '#D9A961', "#D15472"]
    colors = {name: color for name, color in zip(datasets.keys(), colors)}
    kmfs, data_dict, median_ann = [], {}, []
    df_size = 0

    for label, df in datasets.items():
        kdf = df.dropna(subset=['TIME','EVENT'])
        d, e = kdf['TIME'].values, kdf['EVENT'].values

        kmf = KaplanMeierFitter().fit(d, e, label=label)
        kmf.plot(ax=ax, color=colors[label], ci_show=stats,
                 show_censors=True, legend=True)
        kmfs.append(kmf)

        lo, hi = median_survival_times(kmf.confidence_interval_).iloc[0]
        median_ann.append(f"{label}: {kmf.median_survival_time_:.1f} m (95% CI {lo:.1f}-{hi:.1f})")
        data_dict[label] = (d, e)
        df_size += df.shape[0]
    
    all_durs = sum((list(v[0]) for v in data_dict.values()), [])
    all_evts = sum((list(v[1]) for v in data_dict.values()), [])
    all_grps = sum(([lbl]*len(v[0]) for lbl,v in data_dict.items()), [])
    mv = multivariate_logrank_test(pd.Series(all_durs),
                                   pd.Series(all_grps),
                                   event_observed=pd.Series(all_evts))
    chi2_glob, p_glob = mv.test_statistic, mv.p_value

    comps = list(itertools.combinations(data_dict.keys(),2))
    m = len(comps)
    pw_rows = []
    for g1,g2 in comps:
        d1,e1 = data_dict[g1]
        d2,e2 = data_dict[g2]
        lr = logrank_test(d1, d2,
                          event_observed_A=e1,
                          event_observed_B=e2)
        p_adj = min(lr.p_value * m, 1.0)
        pw_rows.append((g1,g2, lr.test_statistic, p_adj))
    pw_df = pd.DataFrame(pw_rows, columns=['g1','g2','chi2','p_bonferroni'])

    lines = []
    lines.extend(median_ann)
    if stats:
        lines.append("")
        lines.append(f"Global log-rank: p={p_glob:.2g}")
        for _, r in pw_df.iterrows():
            lines.append(f"{r['g1']} vs {r['g2']}: p={r['p_bonferroni']:.2g}")

    text_artists = []
    x0_text, y0_text = 0.5, 0.97
    line_h = 0.035
    for i, txt in enumerate(lines):
        ta = ax.text(
            x0_text, y0_text - i*line_h, txt,
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize=12, zorder=2
        )
        text_artists.append(ta)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = [t.get_window_extent(renderer) for t in text_artists]
    union = Bbox.union(bboxes)
    axes_bbox = ax.transAxes.inverted().transform_bbox(union)

    pad_x, pad_y = 0.015, 0.02
    x0 = axes_bbox.x0 - pad_x
    y0 = axes_bbox.y0 - pad_y
    width  = axes_bbox.width + 2*pad_x
    height = axes_bbox.height + 2*pad_y

    rect = Rectangle(
        (x0, y0),
        width, height,
        transform=ax.transAxes,
        facecolor='white',
        edgecolor='black',
        alpha=0.7,
        zorder=1
    )
    ax.add_patch(rect)

    for t in text_artists:
        y = t.get_position()[1]
        t.set_ha('left')
        t.set_position((x0 + pad_x, y))

    ax.legend(loc='upper right', fontsize=10)

    add_at_risk_counts(*kmfs, ax=ax, labels=list(datasets.keys()), fontsize=13, ypos=-0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Months', fontsize=13)
    ax.set_ylabel('Survival Probability', fontsize=13)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_title(f'KM Plot (test set, {df_size} patients) - COX - OS', fontsize=14)

    return plt, pw_df


def plot_cindex_results(
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
    dlif_seed: int = 0,                                        # seed number
    save_name: Optional[str] = None                            # custom filename (without extension) for saved plot
):
    """
    Expected per-analysis layout:

    MLEF:
        mlef_pipeline/results/
         OS_24/ (or DCR/ OR OS_6/ or OS/)
          C23/
            CB/
              model_XX.pkl
              train_set.csv
              test_set.csv
              prediction_CV.csv   (columns: Subject, y_pred, y_true)
              prediction_TEST.csv
              prediction_EXVAL.csv
              results.xlsx      (metrics incl. AUC, C-INDEX)
            CB_DP/
               CB_ONLY/         (MLEF only)
            CB_PYRAD/
            CB_FMRAD/
            CB_DP_PYRAD/
            CB_DP_FMRAD/

    DLIF:
        dlif_pipeline/results/
         C23/
          os_months_24/ (or DCR/)
           classification/
            standard/ (or cross_validation/, evaluation/)
             hypothesis_driven/ (or data_driven/)
              pyrad-noimp/ (or other extraction methods)
               cb/
                seed_0/
                 predictions.parquet
                 predictions_train.parquet
                 eval_auc_ci.csv
                 eval_classification_metrics.csv
               cb_dp/
               cb_radfm/
               cb_radpy/
               cb_radfm_dp/
               cb_radpy_dp/
    """
    metric = 'C-INDEX'
    # ------------------------- utilities -------------------------
    def _parse_mean_std(val) -> Tuple[float, float]:
        if pd.isna(val):
            return (np.nan, np.nan)
        if isinstance(val, (int, float, np.floating)):
            return (round(float(val), 2), 0.0)
        s = str(val).replace("+/-", "±")
        parts = [p.strip() for p in s.split("±")]
        try:
            if len(parts) == 2:
                return (round(float(parts[0]), 2), round(float(parts[1]), 2))
            return (round(float(parts[0]), 2), 0.0)
        except Exception:
            return (np.nan, np.nan)

    def _read_result_cv(results_path: Path) -> Tuple[float, float]:
        if results_path is None or not results_path.exists():
            return (np.nan, np.nan)
        
        df = pd.read_excel(results_path)
        
        sub = df[df['SET'] == 'CV']
        if sub.empty:
            return (np.nan, np.nan)
        
        return _parse_mean_std(sub[metric].iloc[0])

    def _read_c_index_dlif(cindex_csv_path: Path) -> Tuple[float, float]:
        """Read C-Index from DLIF's eval_cindex_ci.csv file."""
        if cindex_csv_path is None or not cindex_csv_path.exists():
            return (np.nan, np.nan)
        try:
            df = pd.read_csv(cindex_csv_path)
            # Expected columns: c_index, ci_lower, ci_upper
            if 'c_index' in df.columns:
                c_index = float(df['c_index'].iloc[0])
                
                if 'ci_lower' in df.columns and 'ci_upper' in df.columns:
                    ci_lower = float(df['ci_lower'].iloc[0])
                    ci_upper = float(df['ci_upper'].iloc[0])
                    std = (ci_upper - ci_lower) / 2
                    return (c_index, std)
                return (c_index, 0.0)
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

    def _read_n_train(train_path: Path) -> int:
        return pd.read_csv(train_path).shape[0]

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

    def _collect_modalities(analysis_dir: Path) -> List[str]:
        return _ordered_modalities([p.name for p in analysis_dir.iterdir()
                                    if p.is_dir() and p.name.upper().startswith("CB")])

    def _pair_paths(analysis_dir: Path, modality: str):
        """
        Robust path resolver for MLEF:
        - accepts CB_ONLY or cb-only (any case)
        - accepts files named like results(.xlsx/.xls), prediction(_CV).xlsx, train_set(.xlsx), etc.
        - accepts files with different case
        """
        def find_first(dir_: Path, stems: List[str]) -> Optional[Path]:
            if not dir_.exists():
                return None
            # try exact matches first
            for st in stems:
                for ext in ("", ".xlsx", ".xls"):
                    p = dir_ / f"{st}{ext}"
                    if p.exists():
                        return p
            # then glob by prefix (case-insensitive)
            cand: List[Path] = []
            for st in stems:
                cand += list(dir_.glob(f"{st}*"))
                cand += list(dir_.glob(f"{st.upper()}*"))
                cand += list(dir_.glob(f"{st.lower()}*"))
            # prefer xlsx/xls if multiple
            cand = sorted(cand, key=lambda x: (x.suffix.lower() not in {".xlsx", ".xls"}, len(x.name)))
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
            "cb_only": None
        }

        if arch_name == "MLEF":
            ro_dir = find_subdir_any(
                mod_dir,
                ["CB_ONLY", "cb-only", "rwd_only", "RWD-ONLY"]
            )
            if ro_dir:
                paths["cb_only"] = {
                    "results": find_first(ro_dir, ["results", "Results"]),
                    "pred":    find_first(ro_dir, ["prediction_CV", "prediction", "Prediction"]),
                    "model":   find_first(ro_dir, ["model_", "model"]),
                    "train":   find_first(ro_dir, ["train_set", "Train_set", "train"]),
                }
        return paths
 

    def _compute_pvalue(pred_mod_path: Path, pred_ro_path: Path) -> Optional[float]:
        """
        Compare two survival models using bootstrapping and two-tailed p-values test on C-index difference.
        """
        if not (pred_mod_path and pred_ro_path and pred_mod_path.exists() and pred_ro_path.exists()):
            return None
        
        dm = pd.read_csv(pred_mod_path).dropna(subset=['risk_score'])
        dr = pd.read_csv(pred_ro_path).dropna(subset=['risk_score'])
        
        # merge on Subject to ensure same patients
        merged = dm.merge(dr, on='Subject', suffixes=('_mod', '_ro'))
        
        # handle column names
        if 'TIME_mod' in merged.columns:
            time = merged['TIME_mod'].values
            event = merged['EVENT_mod'].values
        else:
            time = merged['TIME'].values
            event = merged['EVENT'].values
        
        risk_mod = merged['risk_score_mod'].values
        risk_ro = merged['risk_score_ro'].values
        
        # compute observed C-index difference
        c_mod = concordance_index(time, -risk_mod, event)
        c_ro = concordance_index(time, -risk_ro, event)
        observed_diff = c_mod - c_ro
        
        n_bootstrap = 1000
        bootstrap_diffs = []
        
        rng = np.random.default_rng(42)
        n_samples = len(merged)
        
        for _ in range(n_bootstrap):
            # resample with replacement
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            
            try:
                c_mod_boot = concordance_index(time[idx], -risk_mod[idx], event[idx])
                c_ro_boot = concordance_index(time[idx], -risk_ro[idx], event[idx])
                bootstrap_diffs.append(c_mod_boot - c_ro_boot)
            except:
                # skip if bootstrap sample has issues
                continue
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        if len(bootstrap_diffs) == 0:
            return None
        
        # two-tailed p-value: proportion of bootstrap diffs as extreme as observed
        p_value = np.mean(np.abs(bootstrap_diffs - np.mean(bootstrap_diffs)) >= np.abs(observed_diff))
        
        return p_value
    

    def _plot_one(analysis: str, rows: List[dict]) -> plt.Figure:
        modalities = [r["modality"] for r in rows]
        if arch_name == "MLEF":
            X = np.arange(len(modalities)) * 1.6
        else:
            X = np.arange(len(modalities))

        cindex_mod_mean = np.array([r[f"{metric}_mod_mean"] for r in rows], dtype=float)
        cindex_mod_std  = np.array([r[f"{metric}_mod_std"]  for r in rows], dtype=float)
        n_train_mod  = np.array([r["n_train_mod"]  for r in rows], dtype=float)
        sizes        = _scale_sizes(n_train_mod)
        model_names  = [r["model_mod"] for r in rows]

        fig = plt.figure(figsize=(13, 6))
        # BLUE: modality
        if arch_name == 'MLEF':
            _eval_prefix = "CV"
        elif dlif_eval_type == "cross_validation":
            _eval_prefix = "CV"
        elif dlif_eval_type == "evaluation":
            _eval_prefix = "Ext-Val"
        else:
            _eval_prefix = "Test"
        plt.plot(X, cindex_mod_mean, linestyle="-", marker="o", label=f"{_eval_prefix} {metric}", color="#1a80bb")
        plt.scatter(X, cindex_mod_mean, s=sizes, color="#1a80bb", zorder=3)
        plt.fill_between(X, cindex_mod_mean - cindex_mod_std, cindex_mod_mean + cindex_mod_std,
                         alpha=0.3, color="#8cc5e3", label="Confidence interval")

        multimodal_better = None

        if arch_name == "MLEF":
            cindex_ro_mean = np.array([r[f"{metric}_ro_mean"] for r in rows], dtype=float)
            cindex_ro_std  = np.array([r[f"{metric}_ro_std"]  for r in rows], dtype=float)
            plt.plot(X, cindex_ro_mean, linestyle="-", marker="o", label=f"CV {metric} - CB-only matched", color="#a00000")
            plt.scatter(X, cindex_ro_mean, s=sizes, color="#a00000", zorder=3)
            plt.fill_between(X, cindex_ro_mean - cindex_ro_std, cindex_ro_mean + cindex_ro_std,
                             alpha=0.3, color="#d8a6a6", label="Confidence interval - CB-only matched")
            multimodal_better = (cindex_mod_mean > cindex_ro_mean)


        if arch_name == "DLIF":
            ttl = title_prefix or f"{_eval_prefix} {metric} - {arch_name}"
            xticks = []
            for m, n in zip(modalities, n_train_mod):
                parts = m.split('_')
                parts = ['CB' if p == 'cb' else p for p in parts]
                # Join with newlines
                formatted_name = '\n'.join(parts)
                xticks.append(f"{formatted_name}")
            plt.xticks(X, xticks, rotation=0, fontsize=12)
        else:
            ttl = title_prefix or f"CV {metric} - {arch_name}"
            xticks = []
            for m, n in zip(modalities, n_train_mod):
                parts = m.replace('CB', 'CB').split('_')
                label_text = '\n'.join(parts)
                if np.isfinite(n):
                    label_text += f"\n(n={int(n)})"
                xticks.append(label_text)
            plt.xticks(X, xticks, rotation=0, fontsize=12)

        
        def _format_pvalue(p: Optional[float]) -> Optional[str]:
            if p is None or (not np.isfinite(p)):
                return None
            if p >= 0.05:
                return None
            if p < 0.001:
                return "(p<0.001)"
            p_str = f"{float(p):.3f}".rstrip('0').rstrip('.')
            return f"(p={p_str})"

        line_h = 0.04
        gap_h  = 0.008

        # --- BLUE annotations ---
        for i, (mval, mstd, mname) in enumerate(zip(cindex_mod_mean, cindex_mod_std, model_names)):
            p_txt = _format_pvalue(rows[i].get("pvalue"))
            has_p = (p_txt is not None)

            if arch_name == "MLEF":
                blue_above_red = bool(multimodal_better is not None and multimodal_better[i])
                if blue_above_red:
                    y_red    = 0.01
                    y_blue_p = y_red + line_h + gap_h
                else:
                    y_blue_p = 0.01
                    y_red    = y_blue_p + (2 * line_h + gap_h if has_p else line_h + gap_h)
                y_blue_val = y_blue_p + (line_h if has_p else 0.0)
            else:  # DLIF
                y_blue_p   = 0.01
                y_blue_val = 0.02
                y_red      = None

            if arch_name == "DLIF":
                plt.text(X[i], y_blue_val, f"{mval:.2f} ± {mstd:.2f}",
                        fontsize=11, ha="center", va="bottom", color="#1a80bb")
            else:
                plt.text(X[i], y_blue_val, f"{mval:.2f} ± {mstd:.2f} ({mname})",
                        fontsize=11, ha="center", va="bottom", color="#1a80bb")

            if has_p:
                plt.text(X[i], y_blue_p, p_txt,
                        fontsize=11, ha="center", va="bottom", color="#1a80bb")

            star = rows[i].get("stars", "")
            if star:
                plt.text(X[i] + 0.64, y_blue_val + 0.002, star,
                        fontsize=11, ha="left", va="bottom", color="black")

        # --- RED annotations ---
        if arch_name == "MLEF":
            for i, rrow in enumerate(rows):
                rv, rs, rname = rrow[f"{metric}_ro_mean"], rrow[f"{metric}_ro_std"], rrow["model_ro"]
                p_txt  = _format_pvalue(rrow.get("pvalue"))
                has_p  = (p_txt is not None)
                blue_above_red = bool(multimodal_better is not None and multimodal_better[i])
                if blue_above_red:
                    base_y = 0.01
                else:
                    base_y = 0.01 + (2 * line_h + gap_h if has_p else line_h + gap_h)
                if np.isfinite(rv) and np.isfinite(rs):
                    plt.text(X[i], base_y, f"{rv:.2f} ± {rs:.2f} ({rname})",
                            fontsize=11, ha="center", va="bottom", color="#a00000")

        plt.title(f"{ttl} - {outcome} {analysis}", pad=18)
        plt.ylabel(metric, fontsize=14)
        plt.tick_params(axis='y', labelsize=13)
        plt.ylim(0, 1)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=13)
        plt.xlim(X[0] - 0.7, X[-1] + 0.85) if len(X) > 0 else None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            if save_name:
                out = Path(save_dir) / f"{save_name}.png"
            else:
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
        arch_name = architecture.name  # Get the last part of the path (e.g., "MLEF" or "DLIF")
    else:
        arch_name = architecture

    for analysis in analyses:
        # Build the correct path based on architecture
        if architecture == "MLEF":
            # MLEF: mlef_pipeline/results/outcome/analysis/ 
            base_path = Path("mlef_pipeline/results") / outcome / analysis
            analysis_dir = base_path
        else:  # DLIF
            if dlif_base_path is None:
                dlif_base_path = Path("dlif_pipeline/results")
            else:
                dlif_base_path = Path(dlif_base_path)

            # Build full path: base / analysis / OS_MONTHS / survival / eval_type / feature_type / extraction
            analysis_dir = (dlif_base_path / analysis / 'OS_MONTHS' / "survival" /
                          dlif_eval_type / dlif_feature_type / dlif_extraction)
        
        if not analysis_dir.exists():
            continue

        # Collect modalities
        if architecture == "DLIF":
            # For DLIF, list directories and map them to MLEF names
            dlif_modalities = [el for el in os.listdir(analysis_dir) if el.startswith('cb')]
            modalities = [map_dlif_to_mlef_modality(m) for m in dlif_modalities]
            modalities = _ordered_modalities(modalities)
            modalities_map = {dlif: mlef for dlif, mlef in zip(dlif_modalities, modalities)}
        else:
            modalities = _collect_modalities(analysis_dir)
            modalities_map = {m: m for m in modalities}

        if not modalities:
            continue

        rows: List[dict] = []
        for mod_key, mod_value in modalities_map.items():
            # Get paths based on architecture
            if architecture == "DLIF":
                paths = pair_paths_dlif(analysis_dir, mod_key, dlif_eval_type, "survival", dlif_seed)
                # Read DLIF-specific files
                cindex_m, std_m = _read_c_index_dlif(paths["mod"]["results"])
                model_m = "MIL"  # DLIF uses MIL models
                ntrain_m = read_n_train_dlif(paths["mod"]["pred"])
            else:
                paths = _pair_paths(analysis_dir, mod_key)
                # modality side
                cindex_m, std_m = _read_result_cv(paths["mod"]["results"])
                model_m = _read_model_name(paths["mod"]["model"])
                ntrain_m = _read_n_train(paths["mod"]["train"])

            row = {
                "modality": mod_value,
                f"{metric}_mod_mean": float(cindex_m),
                f"{metric}_mod_std": float(std_m) if np.isfinite(std_m) else 0.0,
                "n_train_mod": ntrain_m,
                "model_mod": model_m,
            }

            # paired CB_ONLY (MLEF only) with CB red==blue behavior
            if arch_name == "MLEF":
                if mod_key == "CB":
                    # enforce coincidence for CB: red == blue; no p-value
                    row.update({
                        f"{metric}_ro_mean": row[f"{metric}_mod_mean"],
                        f"{metric}_ro_std":  row[f"{metric}_mod_std"],
                        "n_train_ro":  row["n_train_mod"],
                        "model_ro":    row["model_mod"],
                        "pvalue":      None,
                        "stars":       ""
                    })
                else:
                    ro = paths["cb_only"]
                    if ro['results'] is not None:
                        cindex_r, std_r = _read_result_cv(ro["results"])
                        pval = _compute_pvalue(paths["mod"]["pred"], ro["pred"])
                        row.update({
                            f"{metric}_ro_mean": float(cindex_r),
                            f"{metric}_ro_std":  float(std_r) if np.isfinite(std_r) else 0.0,
                            "n_train_ro":  _read_n_train(ro["train"]),
                            "model_ro":    _read_model_name(ro["model"]),
                            "pvalue":      pval,
                            "stars":       p_to_stars(pval) if (pval is not None and np.isfinite(pval)) else ""
                        })
                    else:
                        row.update({
                            f"{metric}_ro_mean": np.nan,
                            f"{metric}_ro_std":  np.nan,
                            "n_train_ro":  np.nan,
                            "model_ro":    "NA",
                            "pvalue":      None,
                            "stars":       ""
                        })

            rows.append(row)

        rows = [r for r in rows if np.isfinite(r[f"{metric}_mod_mean"])]
        if not rows:
            continue
        
        _ = _plot_one(analysis, rows)

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

def shap_beeswarm(model, X_train: pd.DataFrame, X_test: pd.DataFrame, mapping = {}) -> None:
    explainer = shap.Explainer(model.predict_partial_hazard, X_train)
    
    shap_values = explainer(X_test)

    shap_values.feature_names = [mapping.get(name, name) for name in shap_values.feature_names]
    shap.plots.beeswarm(shap_values, show=False, max_display=20)

    plt.show()