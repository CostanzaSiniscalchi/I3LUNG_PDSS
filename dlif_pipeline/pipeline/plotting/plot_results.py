#!/usr/bin/env python3
"""
Generate line plots from config file.

Usage:
    python dlif_pipeline/pipeline/plotting/plot_results.py --config dlif_pipeline/configs/00-config-classification-cv.yaml

This script:
- Parses the config to determine task type, outcomes, and paths
- Generates line plots for AUC or C-index with confidence intervals for all available modalities
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import path building functions from metrics script
from metrics.compute_metrics_from_config import build_path_from_config


def plot_results_from_config(config_arg, base_dir=None):
    """
    Generate line plots for trained models based on config.

    Args:
        config_arg: Either a path to a config YAML file (str) or a config dictionary (dict)
        base_dir: Base results directory (optional, computed from script location if not provided)
    """
    # Load config
    if isinstance(config_arg, str):
        with open(config_arg) as f:
            config = yaml.safe_load(f)
    else:
        config = config_arg
    
    # Define modality display names
    modality_dict = {
        "dp": "DP",
        "rwd": "RWD-only",
        "radfm": "FM RAD",
        "radpy": "PYRAD",
        "radfm_dp": "DP\nFM RAD",
        "radpy_dp": "DP\nPYRAD",
        "rwd_dp": "RWD\nDP",
        "rwd_radfm": "RWD\nFM RAD",
        "rwd_radpy": "RWD\nPYRAD",
        "rwd_radfm_dp": "RWD\nDP\nFM RAD",
        "rwd_radpy_dp": "RWD\nDP\nPYRAD",
        "rwd_radfm_dp_genomics": "RWD\nDP\nFM RAD\nGenomics",
        "rwd_radpy_dp_genomics": "RWD\nDP\nPYRAD\nGenomics",
    }
    
    # Parse path info using a sample modality string (just to get base path structure)
    path_info = build_path_from_config(config, list(modality_dict.keys())[0], base_dir)
    print(path_info)
    task = path_info['task']
    training_type = path_info['training_type']
    outcomes = path_info['outcomes']
    
    print(f"\n Generating plots")
    print(f"   Task: {task}")
    print(f"   Training type: {training_type}")
    print(f"   Outcomes: {', '.join(outcomes)}")
    print(f"   Plotting all available modalities")

    is_survival = (task == 'survival')

    for outcome in outcomes:
        print(f"\n   Processing outcome: {outcome}")
        
        # Build base path for this outcome
        base_path = os.path.join(
            path_info['base_dir'],
            *path_info['prefix_parts'],
            outcome,
            task,
            training_type,
            path_info['data_type'],
            f"{path_info['source']}-{path_info['imp']}"
        )

        if not os.path.exists(base_path):
            print(f"      Path does not exist: {base_path}")
            continue

        # Collect data for plotting across all modalities
        modality_labels = []
        modality_keys = []
        score_values = []
        ci_lowers = []
        ci_uppers = []

        # Iterate through all modalities in the defined order
        for mod_string in modality_dict.keys():
            folder_path = os.path.join(base_path, mod_string)

            if not os.path.exists(folder_path):
                print(f"      Modality folder not found: {mod_string}")
                continue

            # Check for seed_0 folder
            seed_path = os.path.join(folder_path, 'seed_0')
            if not os.path.exists(seed_path):
                print(f"      seed_0 folder not found in: {folder_path}")
                continue

            # Determine which CSV to read
            if is_survival:
                csv_path = os.path.join(seed_path, 'eval_cindex_ci.csv')
                score_col = 'c_index'
            else:
                csv_path = os.path.join(seed_path, 'eval_auc_ci.csv')
                score_col = 'auc'

            if not os.path.exists(csv_path):
                print(f"      CSV not found: {csv_path}")
                continue

            # Read the CSV file
            try:
                df = pd.read_csv(csv_path)
                score = df[score_col].iloc[0]
                ci_lower = df['ci_lower'].iloc[0]
                ci_upper = df['ci_upper'].iloc[0]

                modality_labels.append(modality_dict[mod_string])
                modality_keys.append(mod_string)
                score_values.append(score)
                ci_lowers.append(ci_lower)
                ci_uppers.append(ci_upper)
                
                print(f"      ✓ Found data for: {mod_string}")

            except Exception as e:
                print(f"      Error reading CSV file {csv_path}: {e}")
                continue

        if len(score_values) == 0:
            print(f"      No data found for outcome: {outcome}")
            continue

        print(f"      Found {len(score_values)} modalities with data")

        # Convert to numpy arrays
        score_values = np.array(score_values)
        ci_lowers = np.array(ci_lowers)
        ci_uppers = np.array(ci_uppers)

        # Create the plot - EXACT STYLE FROM ORIGINAL SCRIPT
        plt.figure(figsize=(12, 6))

        # Create title components
        title_parts = [part for part in path_info['prefix_parts'] if part != 'results']
        title_parts.extend([training_type, outcome])
        title_suffix = '-'.join(title_parts)

        if is_survival:
            if training_type == 'cross_validation':
                metric_label = 'CV C-Index'
                plot_title = f'CV C-Index - {title_suffix}'
                y_label = 'C-Index'
            elif training_type == 'standard':
                metric_label = 'Test C-Index'
                plot_title = f'Test C-Index - {title_suffix}'
                y_label = 'C-Index'
            else:
                metric_label = 'C-Index'
                plot_title = f'C-Index - {title_suffix}'
                y_label = 'C-Index'
        else:
            if training_type == 'cross_validation':
                metric_label = 'CV AUC'
                plot_title = f'AUC - {title_suffix}'
                y_label = 'AUC'
            elif training_type == 'standard':
                metric_label = 'Test AUC'
                plot_title = f'Test AUC - {title_suffix}'
                y_label = 'AUC'
            else:
                metric_label = 'AUC'
                plot_title = f'AUC - {title_suffix}'
                y_label = 'AUC'

        # Plot based on number of points - EXACT LOGIC FROM ORIGINAL
        plt.plot(modality_labels, score_values, marker='o', linestyle='-', color='#1a80bb', label=metric_label)
        
        if len(modality_labels) == 1:
            plt.errorbar(modality_labels, score_values, yerr=[score_values - ci_lowers, ci_uppers - score_values], 
                        fmt='o', color='#1a80bb', capsize=5, label=metric_label)
        else:
            plt.fill_between(modality_labels, ci_lowers, ci_uppers, color='#8cc5e3', alpha=0.3, label='Confidence interval')

        # Add value labels
        for i, (val, lower, upper) in enumerate(zip(score_values, ci_lowers, ci_uppers)):
            ci_range = val - lower
            plt.text(i, 0.02, f"{val:.2f} ± {ci_range:.2f}", fontsize=10, ha='center', color='#1a80bb')

        plt.title(plot_title, pad=20)
        plt.ylabel(y_label)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        # Create output directory
        output_dir = os.path.join(
            path_info['base_dir'],
            *path_info['prefix_parts']
        )
        os.makedirs(output_dir, exist_ok=True)

        # Save plot
        output_prefix = '-'.join(title_parts)
        png_path = os.path.join(output_dir, f'line_plot-{output_prefix}-600.png')
        plt.savefig(png_path, dpi=600, bbox_inches='tight')
        print(f"      Saved plot: {png_path}")

        # Save plot data as CSV
        plot_data = pd.DataFrame({
            'Modality': modality_keys,
            'Score': score_values,
            'CI_Lower': ci_lowers,
            'CI_Upper': ci_uppers
        })
        csv_path = os.path.join(output_dir, f'line_plot-{output_prefix}.csv')
        plot_data.to_csv(csv_path, index=False)
        print(f"      Saved data: {csv_path}")

        plt.close()

    print("\n✅ All plots generated!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate line plots from config file for all available modalities"
    )
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--base_dir", default=None, 
                       help="Base results directory (optional)")
    
    args = parser.parse_args()
    
    plot_results_from_config(args.config, args.base_dir)


if __name__ == "__main__":
    main()