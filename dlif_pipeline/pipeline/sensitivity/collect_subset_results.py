#!/usr/bin/env python3
"""
Collect subset evaluation results into a single CSV per outcome.

Usage:
    python collect_subset_results.py <path_to_any_outcome_eval_dir>

Example:
    python dlif_pipeline/scripts/collect_subset_results.py \
        dlif_pipeline/results/C2/CBR/classification/standard/hypothesis_driven/pyrad-noimp/rwd/seed_0/eval/00000-mb_attention_mil

The script swaps the outcome in the path to iterate over all outcomes
(CBR, ORR, DCR, os_months_24, os_months_6). For each, it scans all
has_* and no_* subdirectories and writes a subset_results.csv.
"""

import os
import sys
import pandas as pd
from pathlib import Path

OUTCOMES = ["CBR", "ORR", "DCR", "os_months_24", "os_months_6"]
COLLECT_DIR = 'dlif_pipeline/results/all_mods_subset_analysis/'

def build_csv_name(path_str, outcome):
    """Build CSV filename like C2_DCR_subset_results.csv or C23_int_DCR_subset_results.csv.

    Parses the path to extract cohort (C2/C23) and any analysis prefix
    (int, adeno, pdl1_high, etc.) between the cohort and the outcome.
    """
    parts = Path(path_str).parts
    # Find the "results" anchor
    try:
        ri = list(parts).index("results")
    except ValueError:
        # fallback: just use outcome
        return f"{outcome}_subset_results.csv"

    # Everything after "results": cohort, [analysis...], outcome, task, ...
    after = parts[ri + 1:]
    # Find where the outcome sits
    try:
        oi = list(after).index(outcome)
    except ValueError:
        return f"{outcome}_subset_results.csv"

    cohort = after[0]                    # C2 or C23
    analysis = after[1:oi]               # e.g. ("int",) or () or ("pdl1_high",)

    name_parts = [cohort]
    if analysis:
        name_parts.extend(analysis)
    name_parts.append(outcome)
    return "_".join(name_parts) + "_subset_results.csv"


def read_metrics_from_dir(directory):
    """Read eval_auc_ci.csv, eval_classification_metrics.csv, eval_cindex_ci.csv,
    and subset_info.csv from a directory and return a dict of values."""
    row = {}
    directory = Path(directory)

    auc_path = directory / "eval_auc_ci.csv"
    if auc_path.exists():
        df = pd.read_csv(auc_path)
        row["auc"] = df.iloc[0].get("auc")
        row["ci_lower"] = df.iloc[0].get("ci_lower")
        row["ci_upper"] = df.iloc[0].get("ci_upper")

    cls_path = directory / "eval_classification_metrics.csv"
    if cls_path.exists():
        df = pd.read_csv(cls_path)
        row["f1"] = df.iloc[0].get("f1")
        row["specificity"] = df.iloc[0].get("specificity")
        row["sensitivity"] = df.iloc[0].get("sensitivity")

    cidx_path = directory / "eval_cindex_ci.csv"
    if cidx_path.exists():
        df = pd.read_csv(cidx_path)
        row["c_index"] = df.iloc[0].get("c_index")
        row["ci_lower"] = df.iloc[0].get("ci_lower")
        row["ci_upper"] = df.iloc[0].get("ci_upper")

    info_path = directory / "subset_info.csv"
    if info_path.exists():
        df = pd.read_csv(info_path)
        row["n_patients"] = int(df.iloc[0].iloc[0])

    return row


def collect_for_dir(eval_dir):
    """Collect baseline + all has_*/no_* subdirectory results into a list of rows."""
    eval_dir = Path(eval_dir)
    rows = []

    # Baseline: eval_auc_ci.csv etc. sit at the seed_0 level (two dirs up from mb_attention_mil)
    seed_dir = eval_dir.parent.parent  # 00000-mb_attention_mil -> eval -> seed_0
    baseline = read_metrics_from_dir(seed_dir)
    if baseline:
        baseline["subset"] = "baseline"
        rows.append(baseline)

    # Subset directories
    for subdir in sorted(eval_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        if not (name.startswith("has_") or name.startswith("no_")):
            continue

        row = read_metrics_from_dir(subdir)
        if row:
            row["subset"] = name
            rows.append(row)

    return rows


def main():
    if len(sys.argv) != 2:
        print("Usage: python collect_subset_results.py <path_to_any_outcome_eval_dir>")
        sys.exit(1)
    os.makedirs(COLLECT_DIR, exist_ok=True)
    template_path = sys.argv[1]
    template_path = template_path + '/eval/00000-mb_attention_mil' if not template_path.endswith('eval/00000-mb_attention_mil') else template_path

    # Find which outcome is in the path
    original_outcome = None
    for oc in OUTCOMES:
        if f"/{oc}/" in template_path:
            original_outcome = oc
            break

    if original_outcome is None:
        print(f"ERROR: Could not find any known outcome in path: {template_path}")
        print(f"Expected one of: {OUTCOMES}")
        sys.exit(1)

    for outcome in OUTCOMES:
        eval_dir = Path(template_path.replace(f"/{original_outcome}/", f"/{outcome}/"))
        print(eval_dir)

        if not eval_dir.is_dir():
            print(f"SKIP {outcome}: {eval_dir} does not exist")
            continue

        rows = collect_for_dir(eval_dir)
        if not rows:
            print(f"SKIP {outcome}: no has_*/no_* subdirectories found")
            continue

        result = pd.DataFrame(rows).set_index("subset")
        csv_name = build_csv_name(str(eval_dir), outcome)
        out_path = eval_dir / csv_name
        result.to_csv(out_path)
        collect_path = COLLECT_DIR  + csv_name
        print(f" Also saving a copy to: {collect_path}")
        result.to_csv(collect_path)

        print(f"\n{'='*60}")
        print(f"OUTCOME: {outcome}")
        print(f"{'='*60}")
        print(result.to_string())
        print(f"\nSaved to: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
