import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ------------------------------------------------------------------------------
# Configuration for generating plots across experiments
# Specify: cohort filters, path structure, training type, task, and outcomes to analyze
BASE_DIR = Path(__file__).resolve().parents[4]  # Risali a Mil2/
RESULTS_DIR = BASE_DIR / "results"

training_type = 'cross_validation'
# training_type = 'standard'
sub0 = RESULTS_DIR
sub1 = 'cohort2'
sub2 = ''
task = 'classification'
# task = 'survival'
path_pre = f'mil' # new_path
path_suf = f'{task}/{training_type}/hypothesis_driven/pyrad-noimp'
# ------------------------------------------------------------------------------

if task == 'classification':
    outcomes = ['os_months_24']
elif task == 'survival':
    outcomes = ['OS_MONTHS']
else:
    raise ValueError(f"Unsupported task: {task}")

# modality dictionary
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

for outcome in outcomes:
    path = os.path.join(sub0, sub1, sub2, path_pre, outcome, path_suf)

    # Check if path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Get all folders in the path
    all_folders = [f.name for f in os.scandir(path) if f.is_dir()]

    # Check that all folder names are in the modality dictionary
    for folder in all_folders:
        if folder not in modality_dict:
            raise ValueError(f"Folder '{folder}' is not in the modality dictionary")

    # Determine if this is a survival outcome based on path_suf
    is_survival = path_suf.startswith('survival')

    # Collect data for plotting
    modalities = []
    modality_keys = []  # Keep track of keys for CSV
    score_values = []
    ci_lowers = []
    ci_uppers = []

    # Process folders in the order of the modality_dict
    for key, label in modality_dict.items():
        folder_path = os.path.join(path, key)

        if os.path.exists(folder_path):
            # Check for seed_0 folder
            seed_path = os.path.join(folder_path, 'seed_0')
            if not os.path.exists(seed_path):
                raise FileNotFoundError(f"seed_0 folder not found in: {folder_path}")

            if is_survival:
                # Use eval_cindex_ci.csv for survival outcomes
                csv_path = os.path.join(seed_path, 'eval_cindex_ci.csv')
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"eval_cindex_ci.csv not found in: {seed_path}")

                # Read the CSV file
                try:
                    df = pd.read_csv(csv_path)
                    score = df['cindex_mean'].iloc[0]
                    ci_lower = df['ci_lower'].iloc[0]
                    ci_upper = df['ci_upper'].iloc[0]

                    modalities.append(label)
                    modality_keys.append(key)
                    score_values.append(score)
                    ci_lowers.append(ci_lower)
                    ci_uppers.append(ci_upper)

                except Exception as e:
                    raise ValueError(f"Error reading CSV file {csv_path}: {e}")
            else:
                # Use eval_auc_ci.csv for non-survival outcomes
                csv_path = os.path.join(seed_path, 'eval_auc_ci.csv')
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"eval_auc_ci.csv not found in: {seed_path}")

                # Read the CSV file
                try:
                    df = pd.read_csv(csv_path)
                    score = df['auc'].iloc[0]
                    ci_lower = df[' ci_lower'].iloc[0]  # Note the space in column name
                    ci_upper = df[' ci_upper'].iloc[0]  # Note the space in column name

                    modalities.append(label)
                    modality_keys.append(key)
                    score_values.append(score)
                    ci_lowers.append(ci_lower)
                    ci_uppers.append(ci_upper)

                except Exception as e:
                    raise ValueError(f"Error reading CSV file {csv_path}: {e}")

    # Convert to numpy arrays for plotting
    score_values = np.array(score_values)
    ci_lowers = np.array(ci_lowers)
    ci_uppers = np.array(ci_uppers)

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Create title with sub1-sub2-training_type-outcome format
    title_parts = [sub1, sub2, training_type, outcome]
    # Filter out empty strings and join with '-'
    title_suffix = '-'.join(part for part in title_parts if part)

    if is_survival:
        metric_label = 'CV C-Index'
        plot_title = f'C-Index - {title_suffix}'
        y_label = 'C-Index'
    else:
        metric_label = 'CV AUC'
        plot_title = f'AUC - {title_suffix}'
        y_label = 'AUC'

    plt.plot(modalities, score_values, marker='o', linestyle='-', color='#1a80bb', label=metric_label)
    plt.fill_between(modalities, ci_lowers, ci_uppers, color='#8cc5e3', alpha=0.3, label='Confidence interval')

    for i, (val, lower, upper) in enumerate(zip(score_values, ci_lowers, ci_uppers)):
        ci_range = val - lower  # Since CI is symmetric, this equals upper - val
        plt.text(i, 0.02, f"{val:.2f} ± {ci_range:.2f}", fontsize=10, ha='center', color='#1a80bb')

    plt.title(plot_title, pad=20)
    plt.ylabel(y_label)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    path = os.path.join(sub0, sub1, path_pre)

    # Save as PNG
    os.makedirs(path, exist_ok=True)

    plt.savefig(os.path.join(path, f'line_plot-{sub2}-{training_type}-{outcome}-600.png'), dpi=600, bbox_inches='tight')
    print(f"Saved plot to {os.path.join(path, f'line_plot-{sub2}-{training_type}-{outcome}-600.png')}")

    # Save plot data as CSV
    plot_data = pd.DataFrame({
        'Modality': modality_keys,
        'Score': score_values,
        'CI_Lower': ci_lowers,
        'CI_Upper': ci_uppers
    })
    plot_data.to_csv(os.path.join(path, f'line_plot-{sub2}-{training_type}-{outcome}.csv'), index=False)
