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
            RWD/
              model_XX.pkl
              train_set.csv
              test_set.csv
              prediction_CV.csv   (columns: Subject, y_pred, y_true)
              prediction_TEST.csv
              prediction_EXVAL.csv
              results.xlsx      (metrics incl. AUC, C-INDEX)
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
                                    if p.is_dir() and p.name.upper().startswith("RWD")])

    def _pair_paths(analysis_dir: Path, modality: str):
        """
        Robust path resolver for MLEF:
        - accepts RWD_ONLY or rwd-only (any case)
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
                ["RWD_ONLY", "rwd-only", "Rwd_only", "RWD-ONLY"]
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
        Compare two survival models using permutation test on C-index difference.
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
                parts = m.replace('RWD', 'CB').split('_')
                label_text = '\n'.join(parts)
                if np.isfinite(n):
                    label_text += f"\n(n={int(n)})"
                xticks.append(label_text)
            plt.xticks(X, xticks, rotation=0, fontsize=12)

        
        for i, (mval, mstd, mname) in enumerate(zip(cindex_mod_mean, cindex_mod_std, model_names)):
            base_y = 0.06 if (multimodal_better is not None and multimodal_better[i]) else 0.02
            if arch_name == "DLIF":
                plt.text(X[i], base_y, f"{mval:.2f} ± {mstd:.2f}",
                        fontsize=11, ha="center", color="#1a80bb")
            else:
                plt.text(X[i], base_y, f"{mval:.2f} ± {mstd:.2f} ({mname})",
                        fontsize=11, ha="center", color="#1a80bb")
            star = rows[i].get("stars", "")
            if star:
                plt.text(X[i] + 0.66, base_y + 0.006, star, fontsize=11, ha="left", va="center", color="black")

        '''
        # --- BLUE annotations (now show stars here) ---
        for i, (mval, mstd, mname) in enumerate(zip(cindex_mod_mean, cindex_mod_std, model_names)):
            base_y = 0.06 if (multimodal_better is not None and multimodal_better[i]) else 0.02
            plt.text(i, base_y, f"{mval:.2f} ± {mstd:.2f} ({mname})",
                    fontsize=12, ha="center", color="#1a80bb")
            # stars belong to the multimodal-vs-CB comparison
            star = rows[i].get("stars", "")
            if star:
                # nudge a bit to the right of the blue text
                plt.text(i + 0.36, base_y + 0.006, star, fontsize=12, ha="left", va="center", color="black")

        '''
        # --- RED annotations (keep values but REMOVE stars here) ---
        if arch_name == "MLEF":
            for i, rrow in enumerate(rows):
                rv, rs, rname = rrow[f"{metric}_ro_mean"], rrow[f"{metric}_ro_std"], rrow["model_ro"]
                base_y = 0.02 if multimodal_better[i] else 0.06
                if np.isfinite(rv) and np.isfinite(rs):
                    plt.text(X[i], base_y, f"{rv:.2f} ± {rs:.2f} ({rname})",
                            fontsize=11, ha="center", color="#a00000")

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
                if mod_key == "RWD":
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


def shap_beeswarm(model, X_train: pd.DataFrame, X_test: pd.DataFrame, mapping = {}) -> None:
    explainer = shap.Explainer(model.predict_partial_hazard, X_train)
    
    shap_values = explainer(X_test)

    shap_values.feature_names = [mapping.get(name, name) for name in shap_values.feature_names]
    shap.plots.beeswarm(shap_values, show=False, max_display=20)

    plt.show()