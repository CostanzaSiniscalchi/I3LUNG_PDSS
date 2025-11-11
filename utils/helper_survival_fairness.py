"""
fairness_utils.py - Helper functions for survival fairness analysis
"""

import pandas as pd
import numpy as np
from sksurv.metrics import concordance_index_censored
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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

from itertools import combinations
from scipy.stats import norm

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
        DataFrame with Subject, EVENT, TIME, risk_score columns
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
        site_data = predictions[predictions['Subject'].str.startswith(site)]
        
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
        DataFrame with Subject and race indicator columns
    
    Returns:
    --------
    tuple: (per_group_df, pairwise_df, overall_cindex)
    """
    print("\n" + "="*60)
    print("C-INDEX BY RACE (External Validation)")
    print("="*60)
    
    race_cols = {
        'RACE BLACK OR AFRICAN AMERICAN': 'BLACK OR AFRICAN AMERICAN',
        'RACE WHITE': 'WHITE'
    }
    
    group_results = {}
    
    for race_col, label in race_cols.items():
        if race_col not in clinical_data.columns:
            print(f"\nWarning: Column {race_col} not found in clinical data")
            continue
        
        subjects = clinical_data.loc[clinical_data[race_col] == 1, 'Subject']
        race_data = predictions[predictions['Subject'].isin(subjects)]
        
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
    print(f"\nOverall                        (n={overall['n']:3d}): {format_cindex(overall)}")
    
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

import matplotlib.pyplot as plt
import seaborn as sns

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
