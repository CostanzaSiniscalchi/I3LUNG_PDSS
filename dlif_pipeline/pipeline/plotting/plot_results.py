#!/usr/bin/env python3
"""
Generate result plots from config file.

Usage:
    python dlif_pipeline/pipeline/plotting/plot_results.py --config dlif_pipeline/configs/00-config-classification-cv.yaml
    python dlif_pipeline/pipeline/plotting/plot_results.py --config dlif_pipeline/configs/00-config-classification-cv.yaml --use_preds
    python dlif_pipeline/pipeline/plotting/plot_results.py --config dlif_pipeline/configs/00-config-classification-cv.yaml --show

This script:
- Parses the config to determine task type, outcomes, and paths
- For classification: delegates to plot_auc_results from utils/helper_classification.py
- For survival: delegates to plot_cindex_results from utils/helper_survival_graphs.py
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add parent directory to path (for metrics imports)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# Add project root to path for utils imports (insert at front to avoid shadowing)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Import path building functions from metrics script
from metrics.compute_metrics_from_config import build_path_from_config
from utils.helper_classification import plot_auc_results, map_dlif_to_mlef_modality
from utils.helper_survival_graphs import plot_cindex_results


# Reverse of outcome_to_dlif: DLIF outcome names -> MLEF style
_DLIF_TO_MLEF_OUTCOME = {
    'os_months_24': 'OS_24',
    'os_months_6': 'OS_6',
    'DCR': 'DCR',
}


def plot_results_from_config(config_arg, base_dir=None, use_preds=True, show=False, output_dir=None):
    """
    Generate result plots for trained models based on config.

    For classification tasks, calls plot_auc_results.
    For survival tasks, calls plot_cindex_results.

    Args:
        config_arg: Either a path to a config YAML file (str) or a config dictionary (dict)
        base_dir: Base results directory (optional, computed from script location if not provided)
        use_preds: If True, use prediction files from preds/ directory for p-value computation (classification only)
        show: If True, display plots interactively
        output_dir: If provided, save all plots to this directory instead of the default nested path
    """
    # Load config
    if isinstance(config_arg, str):
        with open(config_arg) as f:
            config = yaml.safe_load(f)
    else:
        config = config_arg

    # Parse path info using a sample modality string (just to get base path structure)
    path_info = build_path_from_config(config, "rwd", base_dir)
    task = path_info['task']
    training_type = path_info['training_type']
    outcomes = path_info['outcomes']

    print(f"\n Generating plots")
    print(f"   Task: {task}")
    print(f"   Training type: {training_type}")
    print(f"   Outcomes: {', '.join(outcomes)}")
    print(f"   use_preds: {use_preds}")

    # Build analyses from prefix_parts (skip leading "results")
    prefix_parts = path_info['prefix_parts']
    analyses = ["/".join(prefix_parts[1:])] if len(prefix_parts) > 1 else ["C23"]

    # Build modality_order from config mods
    modality_order = []
    seen = set()
    for mod_dict in config.get('mods', []):
        mod_string = "_".join(k for k, v in mod_dict.items() if v)
        mlef_name = map_dlif_to_mlef_modality(mod_string)
        if mlef_name not in seen:
            modality_order.append(mlef_name)
            seen.add(mlef_name)

    # Extract DLIF-specific parameters from config
    dlif_eval_type = training_type
    dlif_feature_type = path_info['data_type']
    dlif_extraction = f"{path_info['source']}-{path_info['imp']}"
    seeds = config.get('seed', [0])
    dlif_seed = seeds[0] if seeds else 0

    is_survival = (task == 'survival')

    for outcome in outcomes:
        mlef_outcome = _DLIF_TO_MLEF_OUTCOME.get(outcome, outcome)
        print(f"\n   Processing outcome: {outcome} (as {mlef_outcome})")

        # Build save_dir with full result path:
        # base_dir/prefix_parts/outcome/task/training_type/data_type/source-imp
        if output_dir:
            save_dir = output_dir
        else:
            save_dir = os.path.join(
                path_info['base_dir'],
                *prefix_parts,
                outcome,
                task,
                training_type,
                dlif_feature_type,
                dlif_extraction,
            )
        os.makedirs(save_dir, exist_ok=True)
        print(f"   Save dir: {save_dir}")

        # Build save_name: line_plot-{COHORT}-{subanalysis}-{training_type}-{outcome}
        name_parts = list(prefix_parts[1:]) + [training_type, outcome]
        save_name = "line_plot-" + "-".join(name_parts)
        print(f"   Save name: {save_name}.png")

        if is_survival:
            # Derive title_prefix for survival
            if training_type == 'standard':
                title_prefix = 'TEST C-Index - DLIF'
            elif training_type == 'cross_validation':
                title_prefix = 'CV C-Index - DLIF'
            else:
                title_prefix = 'C-Index - DLIF'

            # Build dlif_base_path for survival (same logic as classification use_preds=False)
            if use_preds:
                surv_dlif_base_path = None
            else:
                surv_dlif_base_path = os.path.join(path_info['base_dir'], prefix_parts[0])

            plot_cindex_results(
                architecture="DLIF",
                outcome=mlef_outcome,
                analyses=analyses,
                modality_order=modality_order,
                exclude_modalities=("DP", "FMRAD", "PYRAD"),
                title_prefix=title_prefix,
                show=show,
                save_dir=save_dir,
                save_name=save_name,
                dlif_base_path=surv_dlif_base_path,
                dlif_eval_type=dlif_eval_type,
                dlif_feature_type=dlif_feature_type,
                dlif_extraction=dlif_extraction,
                dlif_seed=dlif_seed,
            )
        else:
            # Derive title_prefix for classification
            if training_type == 'standard':
                title_prefix = 'TEST AUC - DLIF'
            elif training_type == 'cross_validation':
                title_prefix = 'CV AUC - DLIF'
            else:
                title_prefix = 'AUC - DLIF'

            # Build dlif_base_path: only needed when use_preds=False
            # When use_preds=True, plot_auc_results defaults to "dlif_pipeline/preds"
            if use_preds:
                dlif_base_path = None
            else:
                dlif_base_path = os.path.join(path_info['base_dir'], prefix_parts[0])

            plot_auc_results(
                architecture="DLIF",
                outcome=mlef_outcome,
                analyses=analyses,
                modality_order=modality_order,
                exclude_modalities=("DP", "FMRAD", "PYRAD"),
                title_prefix=title_prefix,
                show=show,
                save_dir=save_dir,
                save_name=save_name,
                dlif_base_path=dlif_base_path,
                dlif_eval_type=dlif_eval_type,
                dlif_feature_type=dlif_feature_type,
                dlif_extraction=dlif_extraction,
                dlif_seed=dlif_seed,
                use_preds=use_preds,
            )

    print("\n All plots generated!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate result plots (AUC or C-Index) from config file"
    )
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--base_dir", default=None,
                       help="Base results directory (optional)")
    parser.add_argument("--use_preds", action="store_true", default=False,
                       help="Use prediction files from preds/ directory for p-value computation")
    parser.add_argument("--show", action="store_true", default=False,
                       help="Display plots interactively")

    args = parser.parse_args()

    plot_results_from_config(args.config, args.base_dir, args.use_preds, args.show)


if __name__ == "__main__":
    main()
