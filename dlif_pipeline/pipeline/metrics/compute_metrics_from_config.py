#!/usr/bin/env python3
"""
Compute metrics from config file.

Usage:
    python compute_metrics_from_config.py --config path/to/config.yaml --base_dir ./results

This script:
- Parses the config to determine task type, outcomes, and paths
- Runs appropriate metrics:
  - For classification: DeLong AUC-ROC CI + F1/Specificity/Sensitivity
  - For survival: C-index with bootstrap CI
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import glob

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import metric functions
from metrics.delong_n import auc_roc_ci, find_latest_mb_attention_dir
from metrics.survival_cindex_ci import compute_c_index_and_ci
from metrics.other_metrics import compute_classification_metrics, compute_weighted_metrics


def build_path_from_config(config, mod_string, base_dir):
    """
    Build the results path based on config parameters.
    Uses the same logic as train/run_training.py for path construction.

    Args:
        config: Loaded YAML config
        base_dir: Base directory (e.g. "results_new"). Falls back to config["base_dir"] then "results".

    Returns:
        Path structure components
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    task = config.get('task')  # 'classification' or 'survival'
    training_type = config.get('training_type')  # 'cross_validation', 'standard', 'evaluation'
    data_type = config.get('data-type', 'hypothesis_driven')
    source = config.get('source', 'pyrad')
    imp = config.get('imp', 'noimp')
    outcomes = config.get('task_settings', {}).get('outcomes', [])

    # Build prefix parts the same way as run_training.py
    resolved_base = base_dir or config.get("base_dir", "results")
    prefix_parts = [resolved_base]

    if config.get("USE_COHORT2_FILTER"):
        prefix_parts.append("C2")
    else:
        prefix_parts.append("C23")
    if config.get("FILTER_INT"):
        prefix_parts.append(f"int")
    if config.get("FILTER_GHD"):
        prefix_parts.append(f"ghd")
    if config.get("FILTER_SZMC"):
        prefix_parts.append(f"szmc")
    if config.get("FILTER_VHIO"):
        prefix_parts.append(f"vhio")
    if config.get("FILTER_MH"):
        prefix_parts.append(f"mh")
    if config.get("ADENO"):
        prefix_parts.append(f"adeno")
    if config.get("FILTER_PDL1"):
        prefix_parts.append(f"pdl1_{config['FILTER_PDL1']}")
    if config.get("FILTER_ALL_MODS"):
        prefix_parts.append("all_mods")
    if config.get("FILTER_SQUAMOUS") is not None:
        prefix_parts.append(f"squamous_{config['FILTER_SQUAMOUS']}")
    if config.get("FILTER_CHEMO_IMMUNO") is not None:
        prefix_parts.append(f"chemoio_{config['FILTER_CHEMO_IMMUNO']}")

    # If only "results", no filter active -> add "main_analysis"
    # if len(prefix_parts) == 1:
    #     prefix_parts.append("main_analysis")

    return {
        'task': task,
        'training_type': training_type,
        'data_type': data_type,
        'source': source,
        'imp': imp,
        'outcomes': outcomes,
        'mod_string': mod_string,
        'base_dir': root_dir,
        'prefix_parts': prefix_parts
    }


def find_result_directories(path_info):
    """
    Find all seed directories for the given path configuration.

    Args:
        path_info: Dictionary with path components

    Returns:
        List of (outcome, seed_directory) tuples
    """
    result_dirs = []

    for outcome in path_info['outcomes']:
        # Build path: base_dir/prefix_parts/OUTCOME/task/training_type/data_type/source-imp/mod_string
        # This matches the path construction in train/run_training.py
        base_path = os.path.join(
            path_info['base_dir'],
            *path_info['prefix_parts'],
            outcome,
            path_info['task'],
            path_info['training_type'],
            path_info['data_type'],
            f"{path_info['source']}-{path_info['imp']}",
            path_info['mod_string']
        )

        if not os.path.exists(base_path):
            print(f"  Path does not exist: {base_path}")
            continue

        print(f" Searching in: {base_path}")

        # Find all seed_0 directories
        for root, dirs, files in os.walk(base_path):
            for dir_name in dirs:
                if dir_name == 'seed_0':
                    seed_dir = os.path.join(root, dir_name)
                    result_dirs.append((outcome, seed_dir))
                    print(f"   Found: {seed_dir}")

    return result_dirs


def _compute_and_save_classification_metrics(predictions, output_dir, label=""):
    """
    Compute and save classification metrics for a set of predictions.

    Args:
        predictions: DataFrame with y_true, y_pred0, y_pred1 columns
        output_dir: Directory to save CSV results
        label: Optional label for print output
    """
    predictions['pred'] = np.exp(predictions['y_pred1']) / (
        np.exp(predictions['y_pred0']) + np.exp(predictions['y_pred1'])
    )

    auc, ci = auc_roc_ci(predictions['y_true'], predictions['pred'], 0.95)
    metrics = compute_classification_metrics(predictions['y_true'], predictions['pred'])

    os.makedirs(output_dir, exist_ok=True)

    auc_file = os.path.join(output_dir, 'eval_auc_ci.csv')
    with open(auc_file, 'w') as f:
        f.write('auc,ci_lower,ci_upper\n')
        f.write(f'{auc:.6f},{ci[0]:.6f},{ci[1]:.6f}\n')

    metrics_file = os.path.join(output_dir, 'eval_classification_metrics.csv')
    with open(metrics_file, 'w') as f:
        f.write('f1,specificity,sensitivity\n')
        f.write(f'{metrics["f1"]:.6f},{metrics["specificity"]:.6f},{metrics["sensitivity"]:.6f}\n')

    prefix = f"    [{label}] " if label else "    "
    print(f"{prefix}AUC = {auc:.4f}, CI = [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"{prefix}F1 = {metrics['f1']:.4f}, Spec = {metrics['specificity']:.4f}, Sens = {metrics['sensitivity']:.4f}")
    print(f"{prefix}Saved to: {output_dir}")


def compute_classification_metrics_for_path(path, outcome):
    """
    Compute DeLong AUC-ROC CI and classification metrics for a given path.

    Args:
        path: Path to seed_0 directory
        outcome: Outcome name
    """
    print(f"\n Computing classification metrics for: {path}")

    # Check which evaluation mode
    has_eval = os.path.isdir(os.path.join(path, 'eval'))
    has_eval_mask = os.path.isdir(os.path.join(path, 'eval_mask'))

    if has_eval_mask:
        # Masked evaluation: compute metrics for each masked subset
        print("   Mode: Masked evaluation")
        eval_mask_path = os.path.join(path, 'eval_mask')

        for subset_dir in sorted(os.listdir(eval_mask_path)):
            subset_path = os.path.join(eval_mask_path, subset_dir)
            if not os.path.isdir(subset_path):
                continue

            try:
                latest_mb_dir = find_latest_mb_attention_dir(subset_path)
                pred_path = os.path.join(subset_path, latest_mb_dir, 'predictions.parquet')
            except FileNotFoundError:
                print(f"   Skipping '{subset_dir}': no mb_attention_mil directory")
                continue

            if not os.path.exists(pred_path):
                print(f"   Skipping '{subset_dir}': no predictions.parquet")
                continue

            print(f"   Reading {subset_dir}: {pred_path}")
            predictions = pd.read_parquet(pred_path)
            _compute_and_save_classification_metrics(
                predictions, subset_path, label=subset_dir
            )

    if has_eval:
        # Standard/evaluation: use eval folder
        print("   Mode: Standard/Evaluation")
        eval_path = os.path.join(path, 'eval')
        latest_mb_dir = find_latest_mb_attention_dir(eval_path)
        pred_path = os.path.join(eval_path, latest_mb_dir, 'predictions.parquet')

        print(f"   Reading: {pred_path}")
        predictions = pd.read_parquet(pred_path)
        _compute_and_save_classification_metrics(predictions, path)

    if not has_eval and not has_eval_mask:
        # Cross-validation: aggregate predictions across folds
        print("   Mode: Cross-validation")
        folders = [f.name for f in os.scandir(path) if f.is_dir()]
        predictions = pd.DataFrame(columns=['slide', 'y_true', 'y_pred0', 'y_pred1'])
        fold_metrics = []
        fold_weights = []

        for folder in folders:
            # Try eval path first (new structure)
            eval_folder_path = os.path.join(path, folder, 'eval')
            if os.path.exists(eval_folder_path):
                try:
                    latest_mb_dir = find_latest_mb_attention_dir(eval_folder_path)
                    pred_path = os.path.join(eval_folder_path, latest_mb_dir, 'predictions.parquet')
                except FileNotFoundError:
                    pred_path = None
            else:
                # Try direct predictions.parquet (old structure)
                pred_path = os.path.join(path, folder, 'predictions.parquet')

            if pred_path and os.path.exists(pred_path):
                print(f"   Reading: {pred_path}")
                fold_preds = pd.read_parquet(pred_path)
                predictions = pd.concat([predictions, fold_preds])

                # Compute fold metrics
                fold_preds['pred'] = np.exp(fold_preds['y_pred1']) / (
                    np.exp(fold_preds['y_pred0']) + np.exp(fold_preds['y_pred1'])
                )
                fold_met = compute_classification_metrics(fold_preds['y_true'], fold_preds['pred'])
                fold_metrics.append(fold_met)
                fold_weights.append(len(fold_preds))

        # Compute aggregate metrics
        if len(predictions) > 0:
            predictions['pred'] = np.exp(predictions['y_pred1']) / (
                np.exp(predictions['y_pred0']) + np.exp(predictions['y_pred1'])
            )
            auc, ci = auc_roc_ci(predictions['y_true'], predictions['pred'], 0.95)

            # Weighted average of other metrics
            weighted_metrics = compute_weighted_metrics(fold_metrics, fold_weights)

            # Save AUC results
            auc_file = os.path.join(path, 'eval_auc_ci.csv')
            with open(auc_file, 'w') as f:
                f.write('auc,ci_lower,ci_upper\n')
                f.write(f'{auc:.6f},{ci[0]:.6f},{ci[1]:.6f}\n')

            # Save other metrics
            metrics_file = os.path.join(path, 'eval_classification_metrics.csv')
            with open(metrics_file, 'w') as f:
                f.write('f1,specificity,sensitivity\n')
                f.write(f'{weighted_metrics["f1"]:.6f},{weighted_metrics["specificity"]:.6f},{weighted_metrics["sensitivity"]:.6f}\n')

            print(f"    AUC = {auc:.4f}, CI = [{ci[0]:.4f}, {ci[1]:.4f}]")
            print(f"    F1 = {weighted_metrics['f1']:.4f}, Spec = {weighted_metrics['specificity']:.4f}, Sens = {weighted_metrics['sensitivity']:.4f}")
            print(f"    Saved to: {auc_file} and {metrics_file}")
        else:
            print("    No predictions found")
            return


def _extract_survival_data(predictions):
    """Extract time, event, risk_score from predictions DataFrame. Returns None if columns missing."""
    if 'y_true0' in predictions.columns and 'y_true1' in predictions.columns:
        time = predictions['y_true0'].values.astype(float)
        event = predictions['y_true1'].values.astype(bool)
        risk_score = -predictions['y_pred0'].values.astype(float)
        return time, event, risk_score
    elif 'event' in predictions.columns and 'y_true' in predictions.columns:
        event = predictions['event'].values.astype(bool)
        time = predictions['y_true'].values.astype(float)
        risk_score = -predictions['y_pred'].values.astype(float)
        return time, event, risk_score
    return None


def _compute_and_save_survival_metrics(predictions, output_dir, label=""):
    """
    Compute and save survival metrics (C-index with bootstrap CI) for a set of predictions.

    Args:
        predictions: DataFrame with survival columns
        output_dir: Directory to save CSV results
        label: Optional label for print output
    """
    data = _extract_survival_data(predictions)
    if data is None:
        print("     Required columns not found in predictions, skipping")
        return

    time, event, risk_score = data
    result = compute_c_index_and_ci(event, time, risk_score, n_boot=1000)

    os.makedirs(output_dir, exist_ok=True)

    cindex_file = os.path.join(output_dir, 'eval_cindex_ci.csv')
    with open(cindex_file, 'w') as f:
        f.write('c_index,ci_lower,ci_upper\n')
        f.write(f'{result["c_index"]:.6f},{result["ci"][0]:.6f},{result["ci"][1]:.6f}\n')

    prefix = f"    [{label}] " if label else "    "
    print(f"{prefix}C-index = {result['c_index']:.4f}, CI = [{result['ci'][0]:.4f}, {result['ci'][1]:.4f}]")
    print(f"{prefix}Saved to: {output_dir}")


def compute_survival_metrics_for_path(path, outcome):
    """
    Compute C-index with bootstrap CI for a given path.

    Args:
        path: Path to seed_0 directory
        outcome: Outcome name (e.g., 'OS_MONTHS')
    """
    print(f"\n Computing survival metrics for: {path}")

    # Check which evaluation mode
    has_eval = os.path.isdir(os.path.join(path, 'eval'))
    has_eval_mask = os.path.isdir(os.path.join(path, 'eval_mask'))

    if has_eval_mask:
        # Masked evaluation: compute metrics for each masked subset
        print("   Mode: Masked evaluation")
        eval_mask_path = os.path.join(path, 'eval_mask')

        for subset_dir in sorted(os.listdir(eval_mask_path)):
            subset_path = os.path.join(eval_mask_path, subset_dir)
            if not os.path.isdir(subset_path):
                continue

            try:
                latest_mb_dir = find_latest_mb_attention_dir(subset_path)
                pred_path = os.path.join(subset_path, latest_mb_dir, 'predictions.parquet')
            except FileNotFoundError:
                print(f"   Skipping '{subset_dir}': no mb_attention_mil directory")
                continue

            if not os.path.exists(pred_path):
                print(f"   Skipping '{subset_dir}': no predictions.parquet")
                continue

            print(f"   Reading {subset_dir}: {pred_path}")
            predictions = pd.read_parquet(pred_path)
            _compute_and_save_survival_metrics(
                predictions, subset_path, label=subset_dir
            )

    if has_eval:
        # Standard/evaluation: use eval folder
        print("   Mode: Standard/Evaluation")
        eval_path = os.path.join(path, 'eval')
        latest_mb_dir = find_latest_mb_attention_dir(eval_path)
        pred_path = os.path.join(eval_path, latest_mb_dir, 'predictions.parquet')

        print(f"   Reading: {pred_path}")
        predictions = pd.read_parquet(pred_path)
        _compute_and_save_survival_metrics(predictions, path)

    if not has_eval and not has_eval_mask:
        # Cross-validation: aggregate predictions across folds
        print("   Mode: Cross-validation")
        folders = [f.name for f in os.scandir(path) if f.is_dir()]
        predictions = pd.DataFrame()

        for folder in folders:
            # Try eval path first
            eval_folder_path = os.path.join(path, folder, 'eval')
            if os.path.exists(eval_folder_path):
                try:
                    latest_mb_dir = find_latest_mb_attention_dir(eval_folder_path)
                    pred_path = os.path.join(eval_folder_path, latest_mb_dir, 'predictions.parquet')
                except FileNotFoundError:
                    pred_path = None
            else:
                pred_path = os.path.join(path, folder, 'predictions.parquet')

            if pred_path and os.path.exists(pred_path):
                print(f"   Reading: {pred_path}")
                fold_preds = pd.read_parquet(pred_path)
                predictions = pd.concat([predictions, fold_preds])

        if len(predictions) == 0:
            print("    No predictions found")
            return

        _compute_and_save_survival_metrics(predictions, path)


def run_task_metrics(config_arg, mod_string, base_dir):
    """
    Compute extended metrics (DeLong CI, F1, etc.) for trained models.

    Args:
        config_arg: Either a path to a config YAML file (str) or a config dictionary (dict)
        base_dir: Base results directory
    """
    # Load config
    if isinstance(config_arg, str):
        with open(config_arg) as f:
            config = yaml.safe_load(f)
    else:
        config = config_arg

    # Parse path info
    path_info = build_path_from_config(config, mod_string, base_dir)
    task = path_info['task']

    print(f"\n Task: {task}")
    print(f" Training type: {path_info['training_type']}")
    print(f" Outcomes: {', '.join(path_info['outcomes'])}")
    print(f" Modalities: {path_info['mod_string']}")

    # Find result directories
    result_dirs = find_result_directories(path_info)

    if len(result_dirs) == 0:
        print("\n No result directories found!")
        return

    print(f"\n Found {len(result_dirs)} result director{'y' if len(result_dirs) == 1 else 'ies'}")

    # Compute metrics based on task type
    for outcome, seed_dir in result_dirs:
        if task == 'classification':
            compute_classification_metrics_for_path(seed_dir, outcome)
        elif task == 'survival':
            compute_survival_metrics_for_path(seed_dir, outcome)
        else:
            print(f"  Unknown task type: {task}")

    print("\n All metrics computed!")


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics from config file"
    )
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--base_dir", required=True, help="Base results directory")
    parser.add_argument("--mod_string", required=True, 
        help="Modality string to plot (e.g., dp_rwd_radfm or just dp)")
    args = parser.parse_args()
    config = args.config
    base_dir = args.base_dir
    mod_string = args.mod_string
    run_task_metrics(config, mod_string, base_dir)

if __name__ == "__main__":
    main()
