"""
Compare model performance: all-data patient subset vs full population.

For each model (trained on a given modality subset), plot the baseline AUC
(full population) and the AUC on only the patients with all data modalities.
One plot per cohort per outcome, matching the style of plot_results.py.

Usage:
    python dlif_pipeline/scripts/plot_subset_vs_baseline.py
    python dlif_pipeline/scripts/plot_subset_vs_baseline.py --show
    python dlif_pipeline/scripts/plot_subset_vs_baseline.py --consolidated --show
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = "dlif_pipeline/results/models_on_all_data"
SAVE_DIR = "dlif_pipeline/results/models_on_all_data/plots"

COHORTS = ["C2", "C23"]
OUTCOMES_RESPONSE = ["DCR", "ORR", "CBR"]
OUTCOMES_SURVIVAL = ["os_months_6", "os_months_24"]
OUTCOMES = OUTCOMES_RESPONSE + OUTCOMES_SURVIVAL
OUTCOME_LABELS = {
    "ORR": "ORR",
    "DCR": "DCR",
    "CBR": "CBR",
    "os_months_6": "OS 6m",
    "os_months_24": "OS 24m",
}

MODALITY_ORDER = [
    "rwd",
    "rwd_dp",
    "rwd_radfm",
    "rwd_radpy",
    "rwd_radfm_dp",
    "rwd_radpy_dp",
    "rwd_radfm_dp_genomics",
    "rwd_radpy_dp_genomics",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_cohort_data(cohort: str) -> pd.DataFrame:
    """Load all per-modality CSVs for a cohort into one DataFrame."""
    rows = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if not fname.startswith(f"{cohort}_") or not fname.endswith("_on_all.csv"):
            continue
        modality = fname.replace(f"{cohort}_", "").replace("_on_all.csv", "")
        path = os.path.join(RESULTS_DIR, fname)
        df = pd.read_csv(path)
        df["modality"] = modality
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Plotting — one figure per outcome, matching _plot_one style
# ---------------------------------------------------------------------------
def plot_outcome(cohort: str, outcome: str, odf: pd.DataFrame,
                 show: bool = False, save: bool = True):
    """Single outcome line plot: baseline (blue) vs all-data subset (red)."""

    ordered_mods = [m for m in MODALITY_ORDER if m in odf["modality"].values]
    odf = odf.set_index("modality").loc[ordered_mods].reset_index()

    X = np.arange(len(ordered_mods))

    # Baseline = full population
    bl_mean = odf["baseline_auc"].values
    bl_std = (odf["baseline_ci_upper"].values - odf["baseline_ci_lower"].values) / 2

    # Subset = all-data patients
    sb_mean = odf["auc"].values
    sb_std = (odf["ci_upper"].values - odf["ci_lower"].values) / 2
    n_patients = odf["n_patients"].values

    baseline_better = (bl_mean > sb_mean)

    fig = plt.figure(figsize=(12, 6))

    # --- BLUE: baseline (full population) ---
    plt.plot(X, bl_mean, linestyle="-", marker="o",
             label="AUC — Full population", color="#1a80bb")
    plt.scatter(X, bl_mean, s=100, color="#1a80bb", zorder=3)
    plt.fill_between(X, bl_mean - bl_std, bl_mean + bl_std,
                     alpha=0.3, color="#8cc5e3", label="CI — Full population")

    # --- RED: all-data subset ---
    plt.plot(X, sb_mean, linestyle="-", marker="o",
             label="AUC — Complete data patients", color="#a00000")
    plt.scatter(X, sb_mean, s=100, color="#a00000", zorder=3)
    plt.fill_between(X, sb_mean - sb_std, sb_mean + sb_std,
                     alpha=0.3, color="#d8a6a6", label="CI — Complete data patients")

    # --- X-tick labels (modality names + n) ---
    xticks = []
    for m, n in zip(ordered_mods, n_patients):
        parts = m.split('_')
        PART_MAP = {'rwd': 'CB', 'radpy': 'PYRAD', 'radfm': 'FMRAD', 'genomics': 'G'}
        parts = [PART_MAP.get(p, p.upper()) for p in parts]
        formatted = '\n'.join(parts)
        xticks.append(f"{formatted}\n(n={int(n)})")
    plt.xticks(X, xticks, rotation=0, fontsize=8)

    # --- Text annotations ---
    for i in range(len(X)):
        # blue annotation
        blue_y = 0.06 if baseline_better[i] else 0.02
        plt.text(i, blue_y, f"{bl_mean[i]:.2f} \u00b1 {bl_std[i]:.2f}",
                 fontsize=11, ha="center", color="#1a80bb")
        # red annotation
        red_y = 0.02 if baseline_better[i] else 0.06
        plt.text(i, red_y, f"{sb_mean[i]:.2f} \u00b1 {sb_std[i]:.2f}",
                 fontsize=11, ha="center", color="#a00000")

    plt.ylabel("AUC", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.xlim(-0.4, len(ordered_mods) - 0.55)

    if save:
        os.makedirs(SAVE_DIR, exist_ok=True)
        out = os.path.join(SAVE_DIR, f"subset_vs_baseline-{cohort}-{outcome}.png")
        plt.savefig(out, dpi=600, bbox_inches="tight")
        print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Consolidated plot — all outcomes stacked vertically
# ---------------------------------------------------------------------------
def plot_consolidated(cohort: str, data: pd.DataFrame,
                      outcomes=None, suffix: str = "all_outcomes",
                      show: bool = False, save: bool = True):
    """Stack outcomes vertically into one figure per cohort."""

    if outcomes is None:
        outcomes = OUTCOMES
    outcomes_with_data = [o for o in outcomes
                          if not data[data["outcome"] == o].empty]
    n = len(outcomes_with_data)
    if n == 0:
        print(f"  No outcomes with data for {cohort}")
        return

    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for row_idx, outcome in enumerate(outcomes_with_data):
        ax = axes[row_idx]
        odf = data[data["outcome"] == outcome].copy()

        ordered_mods = [m for m in MODALITY_ORDER if m in odf["modality"].values]
        odf = odf.set_index("modality").loc[ordered_mods].reset_index()

        X = np.arange(len(ordered_mods))

        # Baseline = full population
        bl_mean = odf["baseline_auc"].values
        bl_std = (odf["baseline_ci_upper"].values
                  - odf["baseline_ci_lower"].values) / 2

        # Subset = all-data patients
        sb_mean = odf["auc"].values
        sb_std = (odf["ci_upper"].values - odf["ci_lower"].values) / 2
        n_patients = odf["n_patients"].values

        baseline_better = (bl_mean > sb_mean)

        # --- BLUE: baseline (full population) ---
        ax.plot(X, bl_mean, linestyle="-", marker="o", color="#1a80bb")
        ax.scatter(X, bl_mean, s=60, color="#1a80bb", zorder=3)
        ax.fill_between(X, bl_mean - bl_std, bl_mean + bl_std,
                        alpha=0.3, color="#8cc5e3")

        # --- RED: all-data subset ---
        ax.plot(X, sb_mean, linestyle="-", marker="o", color="#a00000")
        ax.scatter(X, sb_mean, s=60, color="#a00000", zorder=3)
        ax.fill_between(X, sb_mean - sb_std, sb_mean + sb_std,
                        alpha=0.3, color="#d8a6a6")

        # --- Text annotations ---
        for i in range(len(X)):
            blue_y = 0.08 if baseline_better[i] else 0.02
            ax.text(i, blue_y, f"{bl_mean[i]:.2f}\u00b1{bl_std[i]:.2f}",
                    fontsize=10, ha="center", color="#1a80bb")
            red_y = 0.02 if baseline_better[i] else 0.08
            ax.text(i, red_y, f"{sb_mean[i]:.2f}\u00b1{sb_std[i]:.2f}",
                    fontsize=10, ha="center", color="#a00000")

        ax.set_ylabel(OUTCOME_LABELS[outcome], fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.4, len(ordered_mods) - 0.55)
        ax.grid(True, linestyle="--", alpha=0.6)

    # --- X-tick labels only on the bottom panel ---
    # Use the max n_patients across all outcomes per modality (some outcomes
    # have fewer patients due to missing labels, e.g. os_months_24).
    ordered_mods = [m for m in MODALITY_ORDER if m in data["modality"].values]
    max_n = (data.groupby("modality")["n_patients"].max()
             .reindex(ordered_mods).values)

    xticks = []
    for m, np_ in zip(ordered_mods, max_n):
        parts = m.split('_')
        PART_MAP = {'rwd': 'CB', 'radpy': 'PYRAD', 'radfm': 'FMRAD', 'genomics': 'G'}
        parts = [PART_MAP.get(p, p.upper()) for p in parts]
        formatted = '\n'.join(parts)
        xticks.append(f"{formatted}\n(n={int(np_)})")
    axes[-1].set_xticks(np.arange(len(ordered_mods)))
    axes[-1].set_xticklabels(xticks, fontsize=8)

    # --- Shared legend ---
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="#1a80bb", linestyle="-",
               markerfacecolor="#1a80bb", markersize=6,
               label="AUC \u2014 Full population"),
        Line2D([0], [0], color="#8cc5e3", linewidth=6, alpha=0.3,
               label="CI \u2014 Full population"),
        Line2D([0], [0], marker="o", color="#a00000", linestyle="-",
               markerfacecolor="#a00000", markersize=6,
               label="AUC \u2014 Complete data patients"),
        Line2D([0], [0], color="#d8a6a6", linewidth=6, alpha=0.3,
               label="CI \u2014 Complete data patients"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=4,
        fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.03),
    )

    plt.tight_layout()

    if save:
        os.makedirs(SAVE_DIR, exist_ok=True)
        out = os.path.join(SAVE_DIR,
                           f"subset_vs_baseline-{cohort}-{suffix}.png")
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plot model AUC: all-data subset vs full population"
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    for cohort in COHORTS:
        print(f"\n--- {cohort} ---")
        data = load_cohort_data(cohort)
        if data.empty:
            print(f"  No data for {cohort}")
            continue

        plot_consolidated(cohort, data, OUTCOMES_RESPONSE, "response", show=args.show)
        plot_consolidated(cohort, data, OUTCOMES_SURVIVAL, "survival", show=args.show)

    print("\nDone.")


if __name__ == "__main__":
    main()
