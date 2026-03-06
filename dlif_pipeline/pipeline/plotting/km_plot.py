#!/usr/bin/env python3
"""
Generate Kaplan-Meier plots from config file.

Usage:
    python plot_km.py --config path/to/config.yaml [--annotations_data path/to/annotations.csv] [--rwd_data path/to/rwd.csv]

This script:
- Parses the config to determine outcomes and paths
- Generates KM curves stratified by model predictions (tertiles)
- Optionally generates KM curves stratified by LIPI scores for comparison
- Saves plots and statistical comparison tables
- Defaults to using data/annotations.csv for survival data and data/rwd.csv for LIPI scores
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# Add repo root to path
repo_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, repo_root)

# Import path building and plotting functions
from dlif_pipeline.pipeline.metrics.compute_metrics_from_config import build_path_from_config
from utils.helper_survival_graphs import plot_km_combined


def create_risk_groups(merged_df, score_col='proba1'):
    """
    Create tertile risk groups based on predicted scores.
    
    Args:
        merged_df: DataFrame with predictions and clinical data
        score_col: Column name for risk scores
        
    Returns:
        Dictionary with LOW, INTERMEDIATE, and HIGH risk group dataframes
    """
    # Create tertile groups by predicted probability
    tertiles = np.nanpercentile(merged_df[score_col], [33.33, 66.67])
    merged_df['risk_group'] = pd.cut(
        merged_df[score_col],
        bins=[-np.inf, tertiles[0], tertiles[1], np.inf],
        labels=['LOW RISK', 'INTERMEDIATE RISK', 'HIGH RISK'],
        include_lowest=True
    )

    low = merged_df[merged_df['risk_group'] == 'LOW RISK'][['TIME', 'EVENT']]
    mid = merged_df[merged_df['risk_group'] == 'INTERMEDIATE RISK'][['TIME', 'EVENT']]
    high = merged_df[merged_df['risk_group'] == 'HIGH RISK'][['TIME', 'EVENT']]

    datasets = {
        'LOW RISK': low,
        'INTERMEDIATE RISK': mid,
        'HIGH RISK': high,
    }

    # Filter out empty groups if any
    datasets = {k: v for k, v in datasets.items() if len(v) > 0}
    
    return datasets


def create_lipi_groups(merged_df):
    """
    Create risk groups based on LIPI scores.
    
    Args:
        merged_df: DataFrame with LIPI and clinical data
        
    Returns:
        Dictionary with LIPI risk group dataframes
    """
    # Filter to patients with LIPI scores
    lipi_data = merged_df[merged_df['LIPI'].notna()].copy()
    
    if len(lipi_data) == 0:
        print("      No patients with LIPI scores found")
        return {}
    
    # LIPI is categorical: 0=low risk, 1=intermediate risk, 2=high risk
    lipi_data['LIPI'] = lipi_data['LIPI'].astype(int)

    # Create groups by LIPI categories
    low = lipi_data[lipi_data['LIPI'] == 0][['TIME', 'EVENT']]
    mid = lipi_data[lipi_data['LIPI'] == 1][['TIME', 'EVENT']]
    high = lipi_data[lipi_data['LIPI'] == 2][['TIME', 'EVENT']]

    datasets = {
        'LOW RISK': low,
        'INTERMEDIATE RISK': mid,
        'HIGH RISK': high,
    }
    
    # Filter out empty groups if any
    datasets = {k: v for k, v in datasets.items() if len(v) > 0}
    
    return datasets


def plot_km_from_predictions(parquet_path, annotations_data, rwd_data=None,
                             time_col='OS_MONTHS', event_col='DEATH_EVENT_OC',
                             output_dir=None, modality_name='model',
                             training_type='standard', plot_lipi=True,
                             save_prefix=None):
    """
    Generate KM plots from prediction parquet and clinical data.

    Args:
        parquet_path: Path to predictions parquet file
        annotations_data: Path to CSV/Excel file with survival outcomes (uses 'slide' as ID column)
        rwd_data: Path to CSV/Excel file with RWD data including LIPI scores (uses 'Subject' as ID, optional)
        time_col: Column name for survival time
        event_col: Column name for event indicator
        output_dir: Directory to save plots
        modality_name: Name of modality for plot titles
        training_type: Type of training (standard, cross_validation, etc.)
        plot_lipi: Whether to also generate LIPI comparison plot
        save_prefix: Custom filename prefix (e.g. "km_plot-C23-standard-OS_MONTHS-rwd").
                     If None, uses default naming.

    Returns:
        Tuple of (pairwise_df_predictions_all, pairwise_df_lipi_subset)
        - pairwise_df_predictions_all: Stats for model predictions on all patients
        - pairwise_df_lipi_subset: Stats for LIPI comparison on LIPI-subset patients
    """
    # Load data
    pred_df = pd.read_parquet(parquet_path)
    
    # Load annotations data - support both CSV and Excel
    if annotations_data.endswith('.csv'):
        annotations_df = pd.read_csv(annotations_data)
    else:
        annotations_df = pd.read_excel(annotations_data)

    # Normalize IDs as strings for join
    pred_df['slide'] = pred_df['slide'].astype(str).str.strip()
    annotations_df['slide'] = annotations_df['slide'].astype(str).str.strip()

    # Merge predictions with annotations data
    merged = pred_df.merge(
        annotations_df[['slide', time_col, event_col]], 
        on='slide',
        how='inner'
    )

    # If RWD data is provided, merge LIPI scores
    if rwd_data and os.path.exists(rwd_data):
        if rwd_data.endswith('.csv'):
            rwd_df = pd.read_csv(rwd_data)
        else:
            rwd_df = pd.read_excel(rwd_data)
        
        rwd_df['Subject'] = rwd_df['Subject'].astype(str).str.strip()
        # Match slide from merged to Subject in rwd_df
        merged['Subject'] = merged['slide']
        merged = merged.merge(
            rwd_df[['Subject', 'LIPI']], 
            on='Subject', 
            how='left'
        )

    # Use y_pred0 as the risk score (inverted: low scores = high risk)
    if 'y_pred0' in merged.columns:
        merged['proba1'] = -merged['y_pred0']
    else:
        raise ValueError("Expected column 'y_pred0' in predictions parquet")

    # Clean survival columns
    merged = merged.copy()
    merged['TIME'] = pd.to_numeric(merged[time_col], errors='coerce')
    merged['EVENT'] = pd.to_numeric(merged[event_col], errors='coerce').fillna(0).astype(int)
    merged = merged.dropna(subset=['TIME'])

    print(f"      Total patients in analysis: {len(merged)}")
    if 'LIPI' in merged.columns:
        n_with_lipi = len(merged[merged['LIPI'].notna()])
        print(f"      Patients with LIPI scores: {n_with_lipi}")

    # Create risk groups from predictions (all patients)
    datasets_pred = create_risk_groups(merged, score_col='proba1')

    # Plot KM curves for predictions (all patients)
    plt_pred, pw_df_pred = plot_km_combined(datasets_pred, stats=True)
    
    # Build filename base
    if save_prefix:
        base_name = save_prefix
    else:
        base_name = f'km_plot-{modality_name}-{training_type}'

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save prediction plot
        pred_path = os.path.join(output_dir, f'{base_name}.png')
        plt_pred.savefig(pred_path, bbox_inches='tight', dpi=600)
        print(f"      Saved KM plot (predictions): {pred_path}")

        # Save pairwise comparison table
        csv_path = os.path.join(output_dir, f'{base_name}_pairwise.csv')
        pw_df_pred.to_csv(csv_path, index=False)
        print(f"      Saved pairwise comparisons: {csv_path}")

    plt_pred.close()

    # Plot predictions on LIPI-subset patients
    pw_df_pred_lipi_subset = None
    if plot_lipi and 'LIPI' in merged.columns:
        merged_lipi_subset = merged[merged['LIPI'].notna()].copy()

        if len(merged_lipi_subset) > 0:
            # Create risk groups from predictions (LIPI-subset only)
            datasets_pred_lipi_subset = create_risk_groups(merged_lipi_subset, score_col='proba1')

            plt_pred_lipi_subset, pw_df_pred_lipi_subset = plot_km_combined(datasets_pred_lipi_subset, stats=True)

            if output_dir:
                pred_lipi_subset_path = os.path.join(output_dir, f'{base_name}-lipi_subset.png')
                plt_pred_lipi_subset.savefig(pred_lipi_subset_path, bbox_inches='tight', dpi=600)
                print(f"      Saved KM plot (predictions, LIPI-subset): {pred_lipi_subset_path}")

                csv_lipi_subset_path = os.path.join(output_dir, f'{base_name}-lipi_subset_pairwise.csv')
                pw_df_pred_lipi_subset.to_csv(csv_lipi_subset_path, index=False)
                print(f"      Saved pairwise comparisons (predictions, LIPI-subset): {csv_lipi_subset_path}")

            plt_pred_lipi_subset.close()

    # Optionally plot LIPI comparison
    pw_df_lipi = None
    if plot_lipi and 'LIPI' in merged.columns:
        datasets_lipi = create_lipi_groups(merged)

        if len(datasets_lipi) > 0:
            plt_lipi, pw_df_lipi = plot_km_combined(datasets_lipi, stats=True)

            if output_dir:
                lipi_path = os.path.join(output_dir, f'{base_name}-lipi.png')
                plt_lipi.savefig(lipi_path, bbox_inches='tight', dpi=600)
                print(f"      Saved KM plot (LIPI): {lipi_path}")

                lipi_csv_path = os.path.join(output_dir, f'{base_name}-lipi_pairwise.csv')
                pw_df_lipi.to_csv(lipi_csv_path, index=False)
                print(f"      Saved LIPI pairwise comparisons: {lipi_csv_path}")

            plt_lipi.close()

    return pw_df_pred, pw_df_pred_lipi_subset


def plot_km_from_config(config_arg, annotations_data=None, rwd_data=None, 
                        base_dir=None, modalities=None, plot_lipi=True):
    """
    Generate KM plots for trained models based on config.

    Args:
        config_arg: Either a path to a config YAML file (str) or a config dictionary (dict)
        annotations_data: Path to CSV/Excel file with survival outcomes (default: data/annotations.csv)
        rwd_data: Path to CSV/Excel file with RWD data including LIPI (default: data/rwd.csv)
        base_dir: Base results directory (optional)
        modalities: List of modality strings to plot (None = all available)
        plot_lipi: Whether to generate LIPI comparison plots
    """
    # Default data paths
    if annotations_data is None:
        annotations_data = 'data/annotations.csv'
    if rwd_data is None:
        rwd_data = 'data/rwd.csv'
    
    # Verify annotations data file exists
    if not os.path.exists(annotations_data):
        print(f"❌ Error: Annotations data file not found: {annotations_data}")
        return
    
    # Check if RWD data exists (optional for LIPI comparison)
    rwd_available = os.path.exists(rwd_data)
    if plot_lipi and not rwd_available:
        print(f"⚠️  Warning: RWD data file not found: {rwd_data}")
        print("   LIPI comparison plots will be skipped")
        plot_lipi = False
    
    # Load config
    if isinstance(config_arg, str):
        with open(config_arg) as f:
            config = yaml.safe_load(f)
    else:
        config = config_arg
    
    # Define available modalities
    all_modalities = [
        "dp", "rwd", "radfm", "radpy", "radfm_dp", "radpy_dp",
        "rwd_dp", "rwd_radfm", "rwd_radpy", "rwd_radfm_dp", "rwd_radpy_dp",
        "rwd_radfm_dp_genomics", "rwd_radpy_dp_genomics"
    ]
    
    # Use specified modalities or all available
    modalities_to_plot = modalities if modalities else all_modalities
    
    # Parse path info using first modality
    path_info = build_path_from_config(config, modalities_to_plot[0], base_dir)
    
    task = path_info['task']
    training_type = path_info['training_type']
    outcomes = path_info['outcomes']
    
    # Verify this is a survival task
    if task != 'survival':
        print(f"⚠️  Warning: Task is '{task}', expected 'survival'")
        print("   KM plots are only applicable for survival analysis")
        return
    
    print(f"\n📊 Generating Kaplan-Meier plots")
    print(f"   Task: {task}")
    print(f"   Training type: {training_type}")
    print(f"   Outcomes: {', '.join(outcomes)}")
    print(f"   Annotations data: {annotations_data}")
    print(f"   RWD data: {rwd_data if rwd_available else 'Not available'}")
    print(f"   Modalities: {', '.join(modalities_to_plot)}")

    # Determine survival columns from outcome
    time_col = 'OS_MONTHS'  # Default
    event_col = 'DEATH_EVENT_OC'  # Default
    
    # Build naming parts from prefix_parts (skip leading "results")
    prefix_parts = path_info['prefix_parts']
    name_parts = list(prefix_parts[1:])  # e.g. ["C23"] or ["C23", "int"]

    for outcome in outcomes:
        print(f"\n   Processing outcome: {outcome}")

        # Build base path for this outcome
        base_path = os.path.join(
            path_info['base_dir'],
            *prefix_parts,
            outcome,
            task,
            training_type,
            path_info['data_type'],
            f"{path_info['source']}-{path_info['imp']}"
        )

        if not os.path.exists(base_path):
            print(f"      Path does not exist: {base_path}")
            continue

        # Process each modality
        for mod_string in modalities_to_plot:
            folder_path = os.path.join(base_path, mod_string)

            if not os.path.exists(folder_path):
                print(f"      Modality folder not found: {mod_string}")
                continue

            # Check for seed_0 folder
            seed_path = os.path.join(folder_path, 'seed_0')
            if not os.path.exists(seed_path):
                print(f"      seed_0 folder not found in: {folder_path}")
                continue

            # Look for test-set predictions in eval subfolder
            eval_path = os.path.join(seed_path, 'eval', '00000-mb_attention_mil')
            parquet_path = os.path.join(eval_path, 'predictions.parquet')
            if not os.path.exists(parquet_path):
                print(f"      Predictions parquet not found: {parquet_path}")
                continue

            print(f"      Processing modality: {mod_string}")

            # Create output directory in same location as predictions (seed_0 folder)
            output_dir = seed_path

            # Build save_prefix: km_plot-{COHORT}-{subanalysis}-{training_type}-{outcome}-{modality}
            save_prefix = "km_plot-" + "-".join(name_parts + [training_type, outcome, mod_string])

            try:
                # Generate KM plots
                pw_df_pred, pw_df_pred_lipi_subset = plot_km_from_predictions(
                    parquet_path=parquet_path,
                    annotations_data=annotations_data,
                    rwd_data=rwd_data if rwd_available else None,
                    time_col=time_col,
                    event_col=event_col,
                    output_dir=output_dir,
                    modality_name=mod_string,
                    training_type=training_type,
                    plot_lipi=plot_lipi,
                    save_prefix=save_prefix,
                )
                
                # Print pairwise comparisons
                if pw_df_pred is not None:
                    print(f"\n      Pairwise comparisons (predictions, all patients):")
                    print(pw_df_pred.to_string(index=False))
                
                if pw_df_pred_lipi_subset is not None:
                    print(f"\n      Pairwise comparisons (LIPI-subset):")
                    print(pw_df_pred_lipi_subset.to_string(index=False))
                    
            except Exception as e:
                print(f"      Error processing {mod_string}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n✅ All KM plots generated!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Kaplan-Meier plots from config file"
    )
    parser.add_argument("--config", required=True, 
                       help="Path to config YAML file")
    parser.add_argument("--annotations_data", default=None,
                       help="Path to CSV/Excel file with survival outcomes (default: data/annotations.csv)")
    parser.add_argument("--rwd_data", default=None,
                       help="Path to CSV/Excel file with RWD data including LIPI (default: data/rwd.csv)")
    parser.add_argument("--base_dir", default=None, 
                       help="Base results directory (optional)")
    parser.add_argument("--modalities", nargs='+', default=None,
                       help="Specific modalities to plot (default: all available)")
    parser.add_argument("--no_lipi", action='store_true',
                       help="Skip LIPI comparison plots")
    
    args = parser.parse_args()
    
    plot_km_from_config(
        config_arg=args.config,
        annotations_data=args.annotations_data,
        rwd_data=args.rwd_data,
        base_dir=args.base_dir,
        modalities=args.modalities,
        plot_lipi=not args.no_lipi
    )


if __name__ == "__main__":
    main()