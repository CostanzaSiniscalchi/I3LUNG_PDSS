#!/usr/bin/env python3
"""
Visualize masked-bag evaluation results.

Reads the consolidated CSVs from collect_mask_results.py and produces
dumbbell plots showing baseline vs masked AUC with CIs (one per model/cohort).

Usage:
    python dlif_pipeline/pipeline/sensitivity/plot_mask_results.py
    python dlif_pipeline/pipeline/sensitivity/plot_mask_results.py --show
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

DATA_DIR = "dlif_pipeline/results/mask_analysis"
SAVE_DIR = "dlif_pipeline/results/mask_analysis/plots"

COHORTS = ["C2", "C23"]
MODELS = ["rwd_radpy_dp_genomics", "rwd_radfm_dp_genomics"]
OUTCOMES_RESPONSE = ["DCR", "ORR", "CBR"]
OUTCOMES_SURVIVAL = ["os_months_6", "os_months_24"]
OUTCOME_LABELS = {
    "ORR": "ORR",
    "DCR": "DCR",
    "CBR": "CBR",
    "os_months_6": "OS 6m",
    "os_months_24": "OS 24m",
}

COLOR_STD_BASELINE = "#888888"   # gray — all patients
COLOR_SUBSET_BASELINE = "#1a80bb"  # blue — complete-data patients, no masking
COLOR_MASKED = "#a00000"         # red — masked


# ---------------------------------------------------------------------------
# Dumbbell plot with two reference lines
# ---------------------------------------------------------------------------
def plot_dumbbell(cohort, model, df, outcomes, suffix, show=False, save=True):
    """Dumbbell plot: two baselines + masked AUC with CI bars.

    Gray dashed = all-patients baseline (standard test set).
    Blue dashed = complete-data subset baseline (no masking).
    Red dots + CI = masked AUC.
    """
    sub = df[(df["cohort"] == cohort) & (df["model"] == model)].copy()

    # Rename display labels
    LABEL_MAP = {"radpy": "PYRAD", "radfm": "FMRAD", "fmrad": "FMRAD",
                 "genomics": "G", "gen": "G",
                 "RadPy": "PYRAD", "RadFM": "FMRAD",
                 "Genomics": "G"}
    for old, new in LABEL_MAP.items():
        sub["masked_label"] = sub["masked_label"].str.replace(old, new, regex=False)

    mask_order = (
        sub[["masked_label", "n_masked"]]
        .drop_duplicates()
        .sort_values("n_masked")["masked_label"]
        .tolist()
    )

    n_outcomes = len(outcomes)
    n_masks = len(mask_order)

    row_spacing = 0.55
    fig, axes = plt.subplots(
        1,
        n_outcomes,
        figsize=(3.8 * n_outcomes, row_spacing * n_masks + 1.5),
        sharey=True,
    )
    if n_outcomes == 1:
        axes = [axes]

    y_positions = np.arange(n_masks)[::-1] * row_spacing
    # Show every other y-tick to reduce clutter
    ytick_step = max(1, n_masks // 8)
    ytick_indices = list(range(0, n_masks, ytick_step))

    for col_idx, outcome in enumerate(outcomes):
        ax = axes[col_idx]
        odata = sub[sub["outcome"] == outcome]

        if odata.empty:
            ax.set_title(OUTCOME_LABELS[outcome], fontsize=14, fontweight="bold")
            continue

        first_row = odata.iloc[0]

        # All-patients baseline (gray)
        if pd.notna(first_row.get("std_baseline_auc")):
            ax.axvline(
                first_row["std_baseline_auc"],
                color=COLOR_STD_BASELINE,
                linewidth=1.0,
                linestyle="--",
                alpha=0.7,
                zorder=0,
            )

        # Subset baseline — complete-data, no masking (blue)
        if pd.notna(first_row.get("subset_baseline_auc")):
            ax.axvline(
                first_row["subset_baseline_auc"],
                color=COLOR_SUBSET_BASELINE,
                linewidth=1.2,
                linestyle="--",
                alpha=0.7,
                zorder=0,
            )

        for row_idx, mask_label in enumerate(mask_order):
            y = y_positions[row_idx]
            mrow = odata[odata["masked_label"] == mask_label]

            if mrow.empty:
                continue

            mrow = mrow.iloc[0]
            masked_auc = mrow["auc"]
            ci_lo = mrow["ci_lower"]
            ci_hi = mrow["ci_upper"]

            tick_half = 0.05 * row_spacing
            ci_lw = 1.2
            cap_lw = 1.5

            # CI error bar
            ax.plot(
                [ci_lo, ci_hi], [y, y], color=COLOR_MASKED, linewidth=ci_lw, zorder=2
            )
            # Left tick cap
            ax.plot(
                [ci_lo, ci_lo],
                [y - tick_half, y + tick_half],
                color=COLOR_MASKED,
                linewidth=cap_lw,
                zorder=2,
            )
            # Right tick cap
            ax.plot(
                [ci_hi, ci_hi],
                [y - tick_half, y + tick_half],
                color=COLOR_MASKED,
                linewidth=cap_lw,
                zorder=2,
            )
            # Center dot
            ax.plot(
                masked_auc,
                y,
                "o",
                color=COLOR_MASKED,
                markersize=7,
                markeredgewidth=0,
                zorder=3,
            )

        shown_yticks = [y_positions[i] for i in ytick_indices]
        shown_labels = [mask_order[i] for i in ytick_indices]
        ax.set_yticks(shown_yticks)
        ax.set_yticklabels(shown_labels, fontsize=11)
        ax.set_xlabel("AUC", fontsize=12)
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=14, fontweight="bold")
        ax.set_xlim(0.2, 1.0)
        ax.tick_params(axis="x", labelsize=11)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Patient counts for legend (consistent across outcomes for a model/cohort)
    n_all = sub["n_patients_all"].dropna().iloc[0] if "n_patients_all" in sub and sub["n_patients_all"].notna().any() else None
    n_subset = sub["n_patients_subset"].dropna().iloc[0] if "n_patients_subset" in sub and sub["n_patients_subset"].notna().any() else None

    lbl_all = f"All patients baseline (n={int(n_all)})" if n_all else "All patients baseline"
    lbl_subset = f"Patients with all modalities (n={int(n_subset)})" if n_subset else "Patients with all modalities"
    lbl_masked = f"Patients with all modalities, masked (n={int(n_subset)})" if n_subset else "Patients with all modalities, masked"

    # Legend: gray → blue → red
    legend_elements = [
        Line2D(
            [0], [0],
            color=COLOR_STD_BASELINE, linewidth=1.0, linestyle="--",
            label=lbl_all,
        ),
        Line2D(
            [0], [0],
            color=COLOR_SUBSET_BASELINE, linewidth=1.2, linestyle="--",
            label=lbl_subset,
        ),
        Line2D(
            [0], [0],
            marker="o", color="w", markerfacecolor=COLOR_MASKED,
            markersize=8, label=lbl_masked,
        ),
    ]
    plt.tight_layout()

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    if save:
        os.makedirs(SAVE_DIR, exist_ok=True)
        out = os.path.join(SAVE_DIR, f"mask_dumbbell_{cohort}_{model}_{suffix}.png")
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
        description="Plot masked-bag evaluation results"
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    master_path = os.path.join(DATA_DIR, "mask_results_all.csv")
    df = pd.read_csv(master_path)

    for cohort in COHORTS:
        for model in MODELS:
            plot_dumbbell(cohort, model, df, OUTCOMES_RESPONSE, "response", show=args.show)
            plot_dumbbell(cohort, model, df, OUTCOMES_SURVIVAL, "survival", show=args.show)

    print("\nDone.")


if __name__ == "__main__":
    main()
