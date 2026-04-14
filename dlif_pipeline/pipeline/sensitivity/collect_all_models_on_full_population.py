#!/usr/bin/env python3
"""
Collect results from evaluating all models on the full-population subset.

For each modality directory, reads the "all data available" subset metrics
across all outcomes and produces one CSV per modality.

Usage:
    python collect_all_models_on_full_population.py <base_dir>

Example:
    python dlif_pipeline/scripts/collect_all_models_on_full_population.py \
        dlif_pipeline/results/C2/CBR/classification/standard/hypothesis_driven/pyrad-noimp
"""

import os
import sys
import pandas as pd
from pathlib import Path

OUTCOMES = ["CBR", "ORR", "DCR", "os_months_24", "os_months_6"]
OUTPUT_DIR = "dlif_pipeline/results/models_on_all_data/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_subset_dir_name(modality: str) -> str:
    """Return the expected subset directory name for a given modality."""
    if "radpy" in modality:
        return "has_radpy_has_dp_has_genomics"
    elif "radfm" in modality:
        return "has_fmrad_has_dp_has_genomics"
    else:
        return "has_radpy_has_fmrad_has_dp_has_genomics"


def find_mb_attention_dir(eval_path: str):
    """Find the latest *-mb_attention_mil directory inside eval_path."""
    if not os.path.isdir(eval_path):
        return None
    dirs = [d for d in os.listdir(eval_path) if "mb_attention_mil" in d and os.path.isdir(os.path.join(eval_path, d))]
    if not dirs:
        return None
    return sorted(dirs)[-1]


def read_metrics_from_dir(directory, prefix=""):
    """Read eval_auc_ci.csv, eval_classification_metrics.csv,
    eval_cindex_ci.csv, and subset_info.csv from *directory*.

    If *prefix* is given, all keys are prefixed (e.g. "baseline_auc").
    Returns a dict of metric values (may be empty).
    """
    row = {}
    directory = Path(directory)

    auc_path = directory / "eval_auc_ci.csv"
    if auc_path.exists():
        df = pd.read_csv(auc_path)
        row[f"{prefix}auc"] = df.iloc[0].get("auc")
        row[f"{prefix}ci_lower"] = df.iloc[0].get("ci_lower")
        row[f"{prefix}ci_upper"] = df.iloc[0].get("ci_upper")

    cls_path = directory / "eval_classification_metrics.csv"
    if cls_path.exists():
        df = pd.read_csv(cls_path)
        row[f"{prefix}f1"] = df.iloc[0].get("f1")
        row[f"{prefix}specificity"] = df.iloc[0].get("specificity")
        row[f"{prefix}sensitivity"] = df.iloc[0].get("sensitivity")

    cidx_path = directory / "eval_cindex_ci.csv"
    if cidx_path.exists():
        df = pd.read_csv(cidx_path)
        row[f"{prefix}c_index"] = df.iloc[0].get("c_index")
        row[f"{prefix}ci_lower"] = df.iloc[0].get("ci_lower")
        row[f"{prefix}ci_upper"] = df.iloc[0].get("ci_upper")

    if not prefix:
        info_path = directory / "subset_info.csv"
        if info_path.exists():
            df = pd.read_csv(info_path)
            row["n_patients"] = int(df.iloc[0].iloc[0])

    return row


def build_csv_prefix(path_str: str, outcome: str) -> str:
    """Extract a prefix like 'C2' or 'C23_adeno' from the results path."""
    parts = Path(path_str).parts
    try:
        ri = list(parts).index("results")
    except ValueError:
        return ""
    after = parts[ri + 1:]
    try:
        oi = list(after).index(outcome)
    except ValueError:
        return ""
    cohort = after[0]
    analysis = after[1:oi]
    name_parts = [cohort]
    if analysis:
        name_parts.extend(analysis)
    return "_".join(name_parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python collect_all_models_on_full_population.py <base_dir>")
        sys.exit(1)

    base_dir = sys.argv[1].rstrip("/")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find the outcome token in the base path
    original_outcome = None
    for oc in OUTCOMES:
        if f"/{oc}/" in base_dir + "/" or f"/{oc}" == base_dir[-(len(oc) + 1):]:
            original_outcome = oc
            break

    if original_outcome is None:
        print(f"ERROR: Could not find any known outcome in path: {base_dir}")
        print(f"Expected one of: {OUTCOMES}")
        sys.exit(1)

    # Discover modality directories
    modalities = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("cb")
    ])

    if not modalities:
        print(f"ERROR: No modality directories (cb*) found in {base_dir}")
        sys.exit(1)

    print(f"Modalities found: {modalities}")

    # Build a cohort prefix for the output filenames
    prefix = build_csv_prefix(base_dir + "/", original_outcome)

    # ----- collect per-modality -----
    for mod in modalities:
        subset_dir_name = get_subset_dir_name(mod)
        rows = []

        for outcome in OUTCOMES:
            # Swap the outcome in the path
            mod_path = os.path.join(base_dir, mod).replace(
                f"/{original_outcome}/", f"/{outcome}/"
            )
            seed_dir = os.path.join(mod_path, "seed_0")
            eval_path = os.path.join(seed_dir, "eval")

            mb_dir = find_mb_attention_dir(eval_path)
            if mb_dir is None:
                print(f"  SKIP {mod}/{outcome}: no mb_attention_mil dir")
                continue

            subset_path = os.path.join(eval_path, mb_dir, subset_dir_name)
            if not os.path.isdir(subset_path):
                print(f"  SKIP {mod}/{outcome}: {subset_dir_name} not found")
                continue

            row = read_metrics_from_dir(subset_path)
            if row:
                row["outcome"] = outcome
                # Also read baseline (all-patient) metrics from seed_0/
                baseline = read_metrics_from_dir(seed_dir, prefix="baseline_")
                row.update(baseline)
                rows.append(row)

        if not rows:
            print(f"  No results found for modality '{mod}'")
            continue

        df = pd.DataFrame(rows).set_index("outcome")

        csv_name = f"{prefix}_{mod}_on_all.csv" if prefix else f"{mod}_on_all.csv"
        out_path = os.path.join(OUTPUT_DIR, csv_name)
        df.to_csv(out_path)

        print(f"\n{'='*60}")
        print(f"MODALITY: {mod}  (subset: {subset_dir_name})")
        print(f"{'='*60}")
        print(df.to_string())
        print(f"\nSaved to: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
