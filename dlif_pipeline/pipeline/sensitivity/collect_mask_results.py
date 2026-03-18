#!/usr/bin/env python3
"""
Collect masked-bag evaluation results into consolidated CSVs.

For each model (rwd_radpy_dp_genomics, rwd_radfm_dp_genomics),
reads baseline and eval_mask results across all cohorts and outcomes.

Usage:
    python dlif_pipeline/pipeline/sensitivity/collect_mask_results.py

Output:
    dlif_pipeline/results/mask_analysis/rwd_radpy_dp_genomics_mask_results.csv
    dlif_pipeline/results/mask_analysis/rwd_radfm_dp_genomics_mask_results.csv
    dlif_pipeline/results/mask_analysis/mask_results_all.csv
"""

import os
import pandas as pd
from pathlib import Path

COHORTS = ["C2", "C23"]
OUTCOMES = ["CBR", "ORR", "DCR", "os_months_6", "os_months_24"]
MODELS = ["rwd_radpy_dp_genomics", "rwd_radfm_dp_genomics"]

BASE_RESULTS = Path("dlif_pipeline/results")
OUTPUT_DIR = Path("dlif_pipeline/results/mask_analysis")

MASK_PATH_TEMPLATE = "{cohort}/{outcome}/classification/evaluation/hypothesis_driven/pyrad-noimp/{model}/seed_0"
STD_PATH_TEMPLATE = "{cohort}/{outcome}/classification/standard/hypothesis_driven/pyrad-noimp/{model}/seed_0"

SUBSET_DIR_NAMES = {
    "rwd_radpy_dp_genomics": "has_radpy_has_dp_has_genomics",
    "rwd_radfm_dp_genomics": "has_fmrad_has_dp_has_genomics",
}

DISPLAY_NAMES = {
    "rwd": "CB",
    "radpy": "RadPy",
    "radfm": "RadFM",
    "dp": "DP",
    "genomics": "Genomics",
}


def read_metrics_from_dir(directory, prefix=""):
    """Read eval_auc_ci.csv and eval_classification_metrics.csv from directory.

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

    return row


def parse_mask_name(mask_name):
    """Parse a mask directory name into kept and masked modality lists.

    E.g. 'rwd_dp_masked_radpy_genomics' -> kept=['rwd', 'dp'], masked=['radpy', 'genomics']
         'rwd_masked_radpy_dp_genomics' -> kept=['rwd'], masked=['radpy', 'dp', 'genomics']

    Returns (kept_list, masked_list).
    """
    parts = mask_name.split("_")
    try:
        mask_idx = parts.index("masked")
    except ValueError:
        return parts, []
    return parts[:mask_idx], parts[mask_idx + 1:]


def format_masked_label(masked_list):
    """Convert ['radpy', 'genomics'] -> 'RadPy + Genomics masked'."""
    names = [DISPLAY_NAMES.get(m, m) for m in masked_list]
    return " + ".join(names) + " masked"


def format_kept_label(kept_list):
    """Convert ['rwd', 'dp'] -> 'CB + DP kept'."""
    names = [DISPLAY_NAMES.get(m, m) for m in kept_list]
    return " + ".join(names) + " kept"


def find_mb_attention_dir(eval_path):
    """Find the latest *-mb_attention_mil directory inside eval_path."""
    if not eval_path.is_dir():
        return None
    dirs = [d for d in os.listdir(eval_path)
            if "mb_attention_mil" in d and (eval_path / d).is_dir()]
    if not dirs:
        return None
    return sorted(dirs)[-1]


def get_patient_counts(eval_path, mb_dir, subset_dir_name):
    """Get patient counts for all-patients and subset from standard eval path.

    Returns (n_patients_all, n_patients_subset).
    """
    n_all, n_subset = None, None
    mb_path = eval_path / mb_dir

    # All patients: count rows in predictions.parquet
    preds_path = mb_path / "predictions.parquet"
    if preds_path.exists():
        try:
            n_all = len(pd.read_parquet(preds_path))
        except Exception:
            pass

    # Subset patients: read subset_info.csv
    info_path = mb_path / subset_dir_name / "subset_info.csv"
    if info_path.exists():
        try:
            n_subset = int(pd.read_csv(info_path).iloc[0]["n_patients"])
        except Exception:
            pass

    return n_all, n_subset


def collect_model_results(model):
    """Collect mask results + standard-path baselines for one model."""
    subset_dir_name = SUBSET_DIR_NAMES[model]
    rows = []

    for cohort in COHORTS:
        for outcome in OUTCOMES:
            # Masked results (saved under evaluation/ but actually test set)
            mask_seed_dir = BASE_RESULTS / MASK_PATH_TEMPLATE.format(
                cohort=cohort, outcome=outcome, model=model
            )
            mask_dir = mask_seed_dir / "eval_mask"
            if not mask_dir.is_dir():
                print(f"  SKIP {cohort}/{outcome}/{model}: no eval_mask directory")
                continue

            # Standard baselines
            std_seed_dir = BASE_RESULTS / STD_PATH_TEMPLATE.format(
                cohort=cohort, outcome=outcome, model=model
            )

            # 1) All-patients baseline from standard
            std_baseline = read_metrics_from_dir(std_seed_dir, prefix="std_baseline_")
            if not std_baseline:
                print(f"  WARN {cohort}/{outcome}/{model}: no standard baseline")

            # 2) Subset baseline (complete-data patients, no masking)
            subset_baseline = {}
            n_patients_all, n_patients_subset = None, None
            eval_path = std_seed_dir / "eval"
            mb_dir = find_mb_attention_dir(eval_path)
            if mb_dir:
                subset_path = eval_path / mb_dir / subset_dir_name
                subset_baseline = read_metrics_from_dir(subset_path, prefix="subset_baseline_")
                if not subset_baseline:
                    print(f"  WARN {cohort}/{outcome}/{model}: no subset baseline at {subset_path}")
                n_patients_all, n_patients_subset = get_patient_counts(
                    eval_path, mb_dir, subset_dir_name
                )
            else:
                print(f"  WARN {cohort}/{outcome}/{model}: no mb_attention_mil dir in {eval_path}")

            for mask_name in sorted(os.listdir(mask_dir)):
                mask_path = mask_dir / mask_name
                if not mask_path.is_dir():
                    continue

                mask_metrics = read_metrics_from_dir(mask_path)
                if not mask_metrics:
                    print(f"  SKIP {cohort}/{outcome}/{model}/{mask_name}: no metrics")
                    continue

                kept, masked = parse_mask_name(mask_name)

                row = {
                    "cohort": cohort,
                    "outcome": outcome,
                    "model": model,
                    "mask_name": mask_name,
                    "kept_modalities": "+".join(kept),
                    "masked_modalities": "+".join(masked),
                    "masked_label": format_masked_label(masked),
                    "kept_label": format_kept_label(kept),
                    "n_masked": len(masked),
                    "n_patients_all": n_patients_all,
                    "n_patients_subset": n_patients_subset,
                }
                row.update(mask_metrics)
                row.update(std_baseline)
                row.update(subset_baseline)

                if "auc" in mask_metrics and "subset_baseline_auc" in subset_baseline:
                    row["delta_auc"] = subset_baseline["subset_baseline_auc"] - mask_metrics["auc"]

                rows.append(row)

    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows = []
    for model in MODELS:
        print(f"\n{'=' * 60}")
        print(f"Collecting: {model}")
        print(f"{'=' * 60}")

        rows = collect_model_results(model)
        all_rows.extend(rows)

        if rows:
            df = pd.DataFrame(rows)
            out_path = OUTPUT_DIR / f"{model}_mask_results.csv"
            df.to_csv(out_path, index=False)
            print(f"\nSaved {len(rows)} rows to: {out_path}")
            print(df.to_string(index=False))

    if all_rows:
        master_df = pd.DataFrame(all_rows)
        master_path = OUTPUT_DIR / "mask_results_all.csv"
        master_df.to_csv(master_path, index=False)
        print(f"\nMaster CSV: {master_path} ({len(all_rows)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
