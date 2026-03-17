"""
Dumbbell plot: RWD-only model performance on patient subsets
defined by data modality availability (DP, Radiomics, Genomics).

5 panels side-by-side (one per outcome), each with 3 rows
(one per modality split). Blue = has modality, Red = missing modality.

Usage:
    python dlif_pipeline/scripts/plot_rwd_subsets.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = "dlif_pipeline/results/rwd_subset_analysis"
SAVE_DIR = "dlif_pipeline/results/rwd_subset_analysis/plots"

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

# Modality splits: (display_name, has_key, no_key)
# Singles, pairwise, and triple combinations for all 4 modalities
MODALITIES = [
    # Singles
    ("DP",                       "has_dp",                        "no_dp"),
    ("FMRAD",                    "has_fmrad",                     "no_fmrad"),
    ("PYRAD",                    "has_radpy",                     "no_radpy"),
    ("G",                        "has_genomics",                  "no_genomics"),
    # Pairs
    ("DP + FMRAD",               "has_fmrad_has_dp",              "no_fmrad_no_dp"),
    ("DP + PYRAD",               "has_radpy_has_dp",              "no_radpy_no_dp"),
    ("DP + G",                   "has_dp_has_genomics",           "no_dp_no_genomics"),
    ("FMRAD + G",                "has_fmrad_has_genomics",        "no_fmrad_no_genomics"),
    ("PYRAD + G",                "has_radpy_has_genomics",        "no_radpy_no_genomics"),
    # Triples
    ("DP + FMRAD + G",           "has_fmrad_has_dp_has_genomics", "no_fmrad_no_dp_no_genomics"),
    ("DP + PYRAD + G",           "has_radpy_has_dp_has_genomics", "no_radpy_no_dp_no_genomics"),
]

COLOR_HAS = "#1a80bb"
COLOR_NO = "#a00000"
COLOR_BASELINE = "#888888"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_results(cohort: str) -> dict[str, pd.DataFrame]:
    """Load all outcome CSVs for a cohort. Returns {outcome: DataFrame}."""
    data = {}
    for outcome in OUTCOMES:
        fname = f"{cohort}_{outcome}_subset_results.csv"
        path = os.path.join(RESULTS_DIR, fname)
        df = pd.read_csv(path).set_index("subset")
        data[outcome] = df
    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_dumbbell(cohort: str, outcomes, suffix: str, save: bool = True, show: bool = True):
    """Create dumbbell plot for given outcomes."""
    data = load_results(cohort)

    n_outcomes = len(outcomes)
    n_mods = len(MODALITIES)

    fig, axes = plt.subplots(
        1, n_outcomes,
        figsize=(3.8 * n_outcomes, 0.5 * n_mods + 1.5),
        sharey=True,
    )
    if n_outcomes == 1:
        axes = [axes]

    y_positions = np.arange(n_mods)[::-1]  # top-to-bottom
    # Show fewer y-ticks to reduce clutter
    ytick_step = max(1, n_mods // 8)
    ytick_indices = list(range(0, n_mods, ytick_step))

    for col_idx, outcome in enumerate(outcomes):
        ax = axes[col_idx]
        df = data[outcome]
        baseline_auc = df.loc["baseline", "auc"]

        for row_idx, (mod_label, has_key, no_key) in enumerate(MODALITIES):
            y = y_positions[row_idx]

            has_row = df.loc[has_key]
            has_auc = has_row["auc"]
            has_ci_lo = has_row["ci_lower"]
            has_ci_hi = has_row["ci_upper"]
            has_n = int(has_row["n_patients"])

            no_row = df.loc[no_key]
            no_auc = no_row["auc"]
            no_ci_lo = no_row["ci_lower"]
            no_ci_hi = no_row["ci_upper"]
            no_n = int(no_row["n_patients"])

            # Connecting line
            ax.plot(
                [has_auc, no_auc], [y, y],
                color="#cccccc", linewidth=1.5, zorder=1,
            )

            # CI error bars, tick caps, and dots
            tick_half = 0.05  # half-height of cap tick in y-axis units
            ci_lw = 1.2      # CI line width
            cap_lw = 1.5     # tick cap line width

            for auc, ci_lo, ci_hi, color in [
                (has_auc, has_ci_lo, has_ci_hi, COLOR_HAS),
                (no_auc, no_ci_lo, no_ci_hi, COLOR_NO),
            ]:
                # horizontal CI line
                ax.plot([ci_lo, ci_hi], [y, y],
                        color=color, linewidth=ci_lw, zorder=2)
                # left tick cap
                ax.plot([ci_lo, ci_lo], [y - tick_half, y + tick_half],
                        color=color, linewidth=cap_lw, zorder=2)
                # right tick cap
                ax.plot([ci_hi, ci_hi], [y - tick_half, y + tick_half],
                        color=color, linewidth=cap_lw, zorder=2)
                # center dot
                ax.plot(auc, y, "o", color=color, markersize=7,
                        markeredgewidth=0, zorder=3)

            # n annotations – stacked vertically, aligned to right edge
            label_x = 1.02  # in axes coordinates (just past right edge)
            ax.text(
                label_x, y + 0.18, f"n={has_n}",
                fontsize=10, color=COLOR_HAS, va="center",
                transform=ax.get_yaxis_transform(),
            )
            ax.text(
                label_x, y - 0.18, f"n={no_n}",
                fontsize=10, color=COLOR_NO, va="center",
                transform=ax.get_yaxis_transform(),
            )

        # Baseline reference
        ax.axvline(
            baseline_auc, color=COLOR_BASELINE, linewidth=0.8,
            linestyle="--", alpha=0.6, zorder=0,
        )

        # Axes formatting — fewer y-ticks
        shown_yticks = [y_positions[i] for i in ytick_indices]
        shown_labels = [MODALITIES[i][0] for i in ytick_indices]
        ax.set_yticks(shown_yticks)
        ax.set_yticklabels(shown_labels, fontsize=11)
        ax.set_xlabel("AUC", fontsize=12)
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=14, fontweight="bold")
        ax.set_xlim(0.3, 1.0)
        ax.tick_params(axis="x", labelsize=11)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_HAS,
               markersize=8, label="Has modality"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_NO,
               markersize=8, label="Missing modality"),
        Line2D([0], [0], color=COLOR_BASELINE, linewidth=0.8,
               linestyle="--", label="Baseline (all patients)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=3,
        fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.06),
    )

    plt.tight_layout()

    if save:
        os.makedirs(SAVE_DIR, exist_ok=True)
        out_path = os.path.join(SAVE_DIR, f"rwd_subset_dumbbell_{cohort}_{suffix}.png")
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        print(f"Saved: {out_path}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for cohort in COHORTS:
        plot_dumbbell(cohort, OUTCOMES_RESPONSE, "response")
        plot_dumbbell(cohort, OUTCOMES_SURVIVAL, "survival")
