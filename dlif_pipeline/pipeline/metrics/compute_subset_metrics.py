#!/usr/bin/env python3
"""
Compute metrics on patient subsets from existing predictions.

Filters existing predictions.parquet by patient subset (e.g., patients who
have radiomics available) and recomputes classification/survival metrics.
Outputs go into subdirectories inside the eval folder.

Usage (config mode):
    python compute_subset_metrics.py \
        --subset_config configs/subset_eval/rwd_subset_eval.yaml \
        --config configs/main/01-config-classification-standard.yaml \
        --base_dir results/results_new \
        --mod_string rwd

Usage (direct path mode):
    python compute_subset_metrics.py \
        --path results/.../rwd/seed_0 \
        --annotations ../data/annotations.csv \
        --subsets has_radpy:HAS_RADPY has_dp:HAS_DP "has_all:HAS_RADPY,HAS_DP,HAS_GENOMICS" \
        --task classification
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from itertools import combinations
from dataclasses import dataclass
from typing import List

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from metrics.delong_n import auc_roc_ci, find_latest_mb_attention_dir
from metrics.survival_cindex_ci import compute_c_index_and_ci
from metrics.other_metrics import compute_classification_metrics
from metrics.compute_metrics_from_config import build_path_from_config, find_result_directories


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SubsetDef:
    """Definition of a patient subset filter.

    filters is a dict mapping column names to required values, e.g.:
        {"HAS_RADPY": 1}                      # patients WITH radpy
        {"HAS_DP": 0}                          # patients WITHOUT dp
        {"HAS_RADPY": 1, "HAS_DP": 0}         # with radpy but without dp
    """
    name: str
    filters: dict   # {column_name: value}


def parse_subset_definitions(subsets_config: dict) -> List[SubsetDef]:
    """
    Parse the YAML subsets block into SubsetDef objects.

    Supports simple toggle format (auto-generates all combinations):
        has_radpy: true
        has_dp: true
        has_genomics: false

    Convention: subset name "has_xyz" maps to annotation column "HAS_XYZ" == 1.
    All pairwise+ combinations of enabled subsets are generated automatically.
    """
    # Detect simple toggle format: all values are bool
    if all(isinstance(v, bool) for v in subsets_config.values()):
        enabled = [name for name, on in subsets_config.items() if on]
        defs = []
        for name in enabled:
            defs.append(SubsetDef(name=name, filters={name.upper(): 1}))
        for r in range(2, len(enabled) + 1):
            for combo in combinations(enabled, r):
                combo_name = "_".join(combo)
                filters = {n.upper(): 1 for n in combo}
                defs.append(SubsetDef(name=combo_name, filters=filters))
        return defs

    # Fallback: verbose format for custom column names / values
    defs = []
    for name, spec in subsets_config.items():
        if "columns" in spec:
            columns = spec["columns"]
        elif "column" in spec:
            columns = [spec["column"]]
        else:
            raise ValueError(f"Subset '{name}' must have 'column' or 'columns' key")
        value = spec.get("value", 1)
        defs.append(SubsetDef(name=name, filters={col: value for col in columns}))
    return defs


KNOWN_MODALITIES = ["has_radpy", "has_fmrad", "has_dp", "has_genomics"]


def subsets_from_flags(has_flags: dict, no_flags: dict) -> List[SubsetDef]:
    """Build SubsetDefs from --has_* and --no_* CLI flags.

    Each enabled flag becomes a single-filter SubsetDef, and all pairwise+
    combinations of ALL enabled flags are generated automatically.
    """
    # Collect individual entries: (name, column, value)
    entries = []
    for name, on in has_flags.items():
        if on:
            entries.append((name, name.upper(), 1))
    for name, on in no_flags.items():
        if on:
            entries.append((name, name.replace("no_", "HAS_", 1).upper(), 0))

    defs = []
    # Singles
    for name, col, val in entries:
        defs.append(SubsetDef(name=name, filters={col: val}))
    # Combinations of size 2..N
    for r in range(2, len(entries) + 1):
        for combo in combinations(entries, r):
            combo_name = "_".join(e[0] for e in combo)
            filters = {e[1]: e[2] for e in combo}
            defs.append(SubsetDef(name=combo_name, filters=filters))
    return defs


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_predictions_by_subset(predictions_df, annotations_df, subset_def, join_column="slide"):
    """
    Filter predictions to patients matching a subset definition.

    Joins predictions with annotations on join_column, then filters rows where
    each column matches its required value in subset_def.filters.

    Returns:
        Filtered predictions DataFrame (same columns as input), or empty DataFrame.
    """
    # Validate columns exist
    for col in subset_def.filters:
        if col not in annotations_df.columns:
            raise ValueError(
                f"Column '{col}' not found in annotations. "
                f"Available HAS_ columns: {[c for c in annotations_df.columns if c.startswith('HAS_')]}"
            )

    # Build mask on annotations
    mask = pd.Series(True, index=annotations_df.index)
    for col, val in subset_def.filters.items():
        mask &= (annotations_df[col] == val)

    subset_patients = set(annotations_df.loc[mask, join_column])

    # Check for slide mismatches
    pred_slides = set(predictions_df[join_column])
    missing = pred_slides - set(annotations_df[join_column])
    if missing:
        print(f"  WARNING: {len(missing)} slides in predictions not found in annotations")

    # Filter predictions
    filtered = predictions_df[predictions_df[join_column].isin(subset_patients)].copy()

    n_total = len(predictions_df)
    n_filtered = len(filtered)
    pct = n_filtered / n_total * 100 if n_total > 0 else 0
    print(f"  Subset '{subset_def.name}': {n_filtered}/{n_total} patients ({pct:.1f}%)")

    return filtered


# ---------------------------------------------------------------------------
# Metric computation and saving
# ---------------------------------------------------------------------------

def compute_and_save_classification_subset(predictions_df, output_dir):
    """Compute AUC (DeLong CI) and classification metrics, save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    # Save filtered predictions
    predictions_df.to_parquet(os.path.join(output_dir, "predictions.parquet"), index=False)

    # Softmax probability from logits
    pred = np.exp(predictions_df["y_pred1"]) / (
        np.exp(predictions_df["y_pred0"]) + np.exp(predictions_df["y_pred1"])
    )
    y_true = predictions_df["y_true"]

    # Check for single-class subset
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        print(f"    WARNING: Only one class present ({unique_labels}), cannot compute AUC")
        with open(os.path.join(output_dir, "eval_auc_ci.csv"), "w") as f:
            f.write("auc,ci_lower,ci_upper\n")
            f.write("NaN,NaN,NaN\n")
        with open(os.path.join(output_dir, "eval_classification_metrics.csv"), "w") as f:
            f.write("f1,specificity,sensitivity\n")
            f.write("NaN,NaN,NaN\n")
    else:
        # AUC with DeLong CI
        auc, ci = auc_roc_ci(y_true, pred, 0.95)
        with open(os.path.join(output_dir, "eval_auc_ci.csv"), "w") as f:
            f.write("auc,ci_lower,ci_upper\n")
            f.write(f"{auc:.6f},{ci[0]:.6f},{ci[1]:.6f}\n")

        # F1, specificity, sensitivity
        metrics = compute_classification_metrics(y_true, pred)
        with open(os.path.join(output_dir, "eval_classification_metrics.csv"), "w") as f:
            f.write("f1,specificity,sensitivity\n")
            f.write(f'{metrics["f1"]:.6f},{metrics["specificity"]:.6f},{metrics["sensitivity"]:.6f}\n')

        print(f"    AUC = {auc:.4f}, CI = [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"    F1 = {metrics['f1']:.4f}, Spec = {metrics['specificity']:.4f}, Sens = {metrics['sensitivity']:.4f}")

    # Save subset info
    n_pos = int((y_true == 1).sum()) if len(y_true) > 0 else 0
    with open(os.path.join(output_dir, "subset_info.csv"), "w") as f:
        f.write("n_patients,n_positive,n_negative\n")
        f.write(f"{len(predictions_df)},{n_pos},{len(predictions_df) - n_pos}\n")

    print(f"    Saved to: {output_dir}")


def compute_and_save_survival_subset(predictions_df, output_dir):
    """Compute C-index with bootstrap CI, save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    # Save filtered predictions
    predictions_df.to_parquet(os.path.join(output_dir, "predictions.parquet"), index=False)

    # Extract survival data (same logic as compute_metrics_from_config.py)
    if "y_true0" in predictions_df.columns and "y_true1" in predictions_df.columns:
        time = predictions_df["y_true0"].values.astype(float)
        event = predictions_df["y_true1"].values.astype(bool)
        risk_score = -predictions_df["y_pred0"].values.astype(float)
    elif "event" in predictions_df.columns and "y_true" in predictions_df.columns:
        event = predictions_df["event"].values.astype(bool)
        time = predictions_df["y_true"].values.astype(float)
        risk_score = -predictions_df["y_pred"].values.astype(float)
    else:
        print(f"    WARNING: Required survival columns not found, skipping")
        return

    # Remove NaN values
    valid = ~(np.isnan(time) | np.isnan(risk_score))
    time, event, risk_score = time[valid], event[valid], risk_score[valid]

    if len(time) < 2:
        print(f"    WARNING: Too few valid samples ({len(time)}), cannot compute C-index")
        with open(os.path.join(output_dir, "eval_cindex_ci.csv"), "w") as f:
            f.write("c_index,ci_lower,ci_upper\n")
            f.write("NaN,NaN,NaN\n")
    else:
        result = compute_c_index_and_ci(event, time, risk_score, n_boot=1000)
        with open(os.path.join(output_dir, "eval_cindex_ci.csv"), "w") as f:
            f.write("c_index,ci_lower,ci_upper\n")
            f.write(f'{result["c_index"]:.6f},{result["ci"][0]:.6f},{result["ci"][1]:.6f}\n')

        print(f"    C-index = {result['c_index']:.4f}, CI = [{result['ci'][0]:.4f}, {result['ci'][1]:.4f}]")

    # Save subset info
    n_events = int(event.sum()) if len(event) > 0 else 0
    with open(os.path.join(output_dir, "subset_info.csv"), "w") as f:
        f.write("n_patients,n_events,n_censored\n")
        f.write(f"{len(time)},{n_events},{len(time) - n_events}\n")

    print(f"    Saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------

def process_standard_eval(seed_dir, task, annotations_df, subsets, join_column="slide"):
    """
    Process standard training results: filter predictions in eval dir by each subset.

    Output goes to: seed_dir/eval/<mb_attention_dir>/<subset_name>/
    """
    eval_path = os.path.join(seed_dir, "eval")
    if not os.path.isdir(eval_path):
        print(f"  No eval directory found in {seed_dir}")
        return

    try:
        latest_mb_dir = find_latest_mb_attention_dir(eval_path)
    except FileNotFoundError:
        print(f"  No mb_attention_mil directory found in {eval_path}")
        return

    pred_path = os.path.join(eval_path, latest_mb_dir, "predictions.parquet")
    if not os.path.exists(pred_path):
        print(f"  Predictions not found: {pred_path}")
        return

    print(f"  Reading predictions: {pred_path}")
    predictions = pd.read_parquet(pred_path)

    compute_fn = (
        compute_and_save_classification_subset if task == "classification"
        else compute_and_save_survival_subset
    )

    for subset in subsets:
        filtered = filter_predictions_by_subset(predictions, annotations_df, subset, join_column)
        if len(filtered) == 0:
            print(f"    Skipping '{subset.name}': no patients in subset")
            continue
        output_dir = os.path.join(eval_path, latest_mb_dir, subset.name)
        compute_fn(filtered, output_dir)


def process_cv_eval(seed_dir, task, annotations_df, subsets, join_column="slide"):
    """
    Process cross-validation results: aggregate predictions across folds,
    then filter by each subset.

    Output goes to: seed_dir/<subset_name>/
    """
    folders = [f.name for f in os.scandir(seed_dir) if f.is_dir() and f.name != "eval"]
    all_predictions = []

    for folder in folders:
        # Try eval path first (new structure)
        eval_folder_path = os.path.join(seed_dir, folder, "eval")
        pred_path = None

        if os.path.exists(eval_folder_path):
            try:
                latest_mb_dir = find_latest_mb_attention_dir(eval_folder_path)
                pred_path = os.path.join(eval_folder_path, latest_mb_dir, "predictions.parquet")
            except FileNotFoundError:
                pass

        # Fallback to direct predictions.parquet (old structure)
        if (pred_path is None or not os.path.exists(pred_path)):
            pred_path = os.path.join(seed_dir, folder, "predictions.parquet")

        if pred_path and os.path.exists(pred_path):
            print(f"  Reading fold predictions: {pred_path}")
            all_predictions.append(pd.read_parquet(pred_path))

    if not all_predictions:
        print(f"  No fold predictions found in {seed_dir}")
        return

    predictions = pd.concat(all_predictions, ignore_index=True)
    print(f"  Aggregated {len(predictions)} predictions across {len(all_predictions)} folds")

    compute_fn = (
        compute_and_save_classification_subset if task == "classification"
        else compute_and_save_survival_subset
    )

    for subset in subsets:
        filtered = filter_predictions_by_subset(predictions, annotations_df, subset, join_column)
        if len(filtered) == 0:
            print(f"    Skipping '{subset.name}': no patients in subset")
            continue
        output_dir = os.path.join(seed_dir, subset.name)
        compute_fn(filtered, output_dir)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_subset_evaluation_for_path(seed_dir, task, annotations_df, subsets, join_column="slide"):
    """Run subset evaluation on a single seed directory."""
    print(f"\nProcessing: {seed_dir}")

    # Determine if standard (has eval/ dir) or CV (has fold dirs)
    has_eval = os.path.isdir(os.path.join(seed_dir, "eval"))

    if has_eval:
        process_standard_eval(seed_dir, task, annotations_df, subsets, join_column)
    else:
        process_cv_eval(seed_dir, task, annotations_df, subsets, join_column)


def run_config_mode(args):
    """Run using config files to resolve paths and subset definitions."""
    # Load subset config
    with open(args.subset_config) as f:
        subset_config = yaml.safe_load(f)

    # Load training config
    with open(args.config) as f:
        training_config = yaml.safe_load(f)

    task = training_config.get("task")
    subsets = parse_subset_definitions(subset_config["subsets"])
    join_column = subset_config.get("join_column", "slide")

    # Resolve annotation file path relative to subset config location
    ann_path = subset_config["annotation_file"]
    if not os.path.isabs(ann_path):
        config_dir = os.path.dirname(os.path.abspath(args.subset_config))
        ann_path = os.path.normpath(os.path.join(config_dir, ann_path))

    print(f"Loading annotations from: {ann_path}")
    annotations_df = pd.read_csv(ann_path)
    # Ensure join column and filter columns are consistent types
    annotations_df[join_column] = annotations_df[join_column].astype(str)
    for subset in subsets:
        for col in subset.filters:
            if col in annotations_df.columns:
                annotations_df[col] = pd.to_numeric(annotations_df[col], errors="coerce")
    print(f"Annotations shape: {annotations_df.shape}")

    # Build path info and find result directories
    path_info = build_path_from_config(training_config, args.mod_string, args.base_dir)

    print(f"\nTask: {task}")
    print(f"Training type: {path_info['training_type']}")
    print(f"Outcomes: {', '.join(path_info['outcomes'])}")
    print(f"Modalities: {path_info['mod_string']}")
    print(f"Subsets: {', '.join(s.name for s in subsets)}")

    result_dirs = find_result_directories(path_info)
    if not result_dirs:
        print("\nNo result directories found!")
        return

    print(f"\nFound {len(result_dirs)} result director{'y' if len(result_dirs) == 1 else 'ies'}")

    for outcome, seed_dir in result_dirs:
        run_subset_evaluation_for_path(seed_dir, task, annotations_df, subsets, join_column)

    print("\nAll subset metrics computed!")


def run_direct_mode(args):
    """Run using direct path and CLI-specified subsets."""
    print(f"Loading annotations from: {args.annotations}")
    annotations_df = pd.read_csv(args.annotations)
    annotations_df["slide"] = annotations_df["slide"].astype(str)

    # Build subsets from --has_* and --no_* flags
    has_flags = {name: getattr(args, name, False) for name in KNOWN_MODALITIES}
    no_flags = {name.replace("has_", "no_"): getattr(args, name.replace("has_", "no_"), False) for name in KNOWN_MODALITIES}
    subsets = subsets_from_flags(has_flags, no_flags)

    if not subsets:
        print("ERROR: No subsets enabled. Use --has_radpy, --no_radpy, --has_dp, etc.")
        sys.exit(1)

    # Coerce filter columns to numeric
    for subset in subsets:
        for col in subset.filters:
            if col in annotations_df.columns:
                annotations_df[col] = pd.to_numeric(annotations_df[col], errors="coerce")

    print(f"Annotations shape: {annotations_df.shape}")
    if "classification" in args.path.lower():
        task = "classification"
    elif "survival" in args.path.lower():
        task = "survival"
    else:        
        raise ValueError(f"ERROR: Unknown task in '{args.path}'. Use 'classification' or 'survival'.")

    print(f"Subsets: {', '.join(s.name for s in subsets)}")

    run_subset_evaluation_for_path(args.path, task, annotations_df, subsets)
    print("\nAll subset metrics computed!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics on patient subsets from existing predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Config mode:
    python compute_subset_metrics.py \\
        --subset_config configs/subset_eval/rwd_subset_eval.yaml \\
        --config configs/main/01-config-classification-standard.yaml \\
        --base_dir results/results_new --mod_string rwd

  Direct path mode:
    python compute_subset_metrics.py \\
        --path results/.../rwd/seed_0 \\
        --annotations ../data/annotations.csv \\
        --has_radpy --has_dp \\
        """,
    )

    # Config mode arguments
    config_group = parser.add_argument_group("Config mode")
    config_group.add_argument("--subset_config", help="Path to subset evaluation config YAML")
    config_group.add_argument("--config", help="Path to training config YAML (for path resolution)")
    config_group.add_argument("--base_dir", help="Base results directory")
    config_group.add_argument("--mod_string", help="Modality string (e.g., rwd)")

    # Direct path mode arguments
    direct_group = parser.add_argument_group("Direct path mode")
    direct_group.add_argument("--path", help="Direct path to seed_0 directory")
    direct_group.add_argument("--annotations", help="Path to annotations.csv", default="data/annotations.csv")

    # Modality subset toggles (used in direct path mode)
    subset_group = parser.add_argument_group("Subset toggles (patients WITH modality)")
    subset_group.add_argument("--has_radpy", action="store_true", help="Patients with radiomics")
    subset_group.add_argument("--has_fmrad", action="store_true", help="Patients with foundation model radiomics")
    subset_group.add_argument("--has_dp", action="store_true", help="Patients with digital pathology")
    subset_group.add_argument("--has_genomics", action="store_true", help="Patients with genomics")

    no_group = parser.add_argument_group("Subset toggles (patients WITHOUT modality)")
    no_group.add_argument("--no_radpy", action="store_true", help="Patients without radiomics")
    no_group.add_argument("--no_fmrad", action="store_true", help="Patients without foundation model radiomics")
    no_group.add_argument("--no_dp", action="store_true", help="Patients without digital pathology")
    no_group.add_argument("--no_genomics", action="store_true", help="Patients without genomics")

    args = parser.parse_args()
    print(args)
    if args.subset_config:
        # Config mode
        if not args.config:
            parser.error("--config is required with --subset_config")
        if not args.mod_string:
            parser.error("--mod_string is required with --subset_config")
        run_config_mode(args)
    elif args.path:
        # Direct path mode
        if not args.annotations:
            parser.error("--annotations is required with --path")
        run_direct_mode(args)
    else:
        parser.error("Either --subset_config or --path is required")


if __name__ == "__main__":
    main()
