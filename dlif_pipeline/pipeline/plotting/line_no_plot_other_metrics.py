import pandas as pd
import numpy as np
import os
from pathlib import Path

# ------------------------------------------------------------------------------
# Configuration for csv F1, Precision, Recall across experiments
# Specify: cohort filters, path structure, training type, task, and outcomes to analyze

BASE_DIR = Path(__file__).resolve().parents[3]  # Risali a Mil2/
RESULTS_DIR = BASE_DIR / "dlif_pipeline/results"


training_type = 'standard'
sub1 = RESULTS_DIR
sub2 = 'C23'
path_pre = f'' # new_path
path_suf = f'classification/{training_type}/hypothesis_driven/pyrad-noimp'
outcomes = ['os_months_24']
# ------------------------------------------------------------------------------

# modality dictionary 
modality_dict = {
    "rwd": "RWD",
    "rwd_dp": "RWD\nDP",
    "rwd_radfm": "RWD\nFM RAD",
    "rwd_radpy": "RWD\nPYRAD",
    "rwd_radfm_dp": "RWD\nDP\nFM RAD",
    "rwd_radpy_dp": "RWD\nDP\nPYRAD",
    "rwd_radfm_dp_genomics": "RWD\nDP\nFM RAD\nGenomics",
    "rwd_radpy_dp_genomics": "RWD\nDP\nPYRAD\nGenomics",
}

for outcome in outcomes:
    path = os.path.join(sub1, sub2, path_pre, outcome, path_suf)
    print(f"Processing path: {path}")

    # Check if path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Get all folders in the path
    all_folders = [f.name for f in os.scandir(path) if f.is_dir()]

    # Check that all folder names are in the modality dictionary
    for folder in all_folders:
        if folder not in modality_dict:
            raise ValueError(f"Folder '{folder}' is not in the modality dictionary")

    # Collect data for output
    modalities = []
    f1_values = []
    specificity_values = []
    sensitivity_values = []

    # Process folders in the order of the modality_dict
    for key, label in modality_dict.items():
        folder_path = os.path.join(path, key)
        
        if os.path.exists(folder_path):
            # Check for seed_0 folder
            seed_path = os.path.join(folder_path, 'seed_0')
            if not os.path.exists(seed_path):
                raise FileNotFoundError(f"seed_0 folder not found in: {folder_path}")
            
            # Use eval_classification_metrics.csv for classification metrics
            csv_path = os.path.join(seed_path, 'eval_classification_metrics.csv')
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"eval_classification_metrics.csv not found in: {seed_path}")
            
            # Read the CSV file
            try:
                df = pd.read_csv(csv_path)
                f1 = df['f1'].iloc[0]
                specificity = df['specificity'].iloc[0]
                sensitivity = df['sensitivity'].iloc[0]
                
                modalities.append(key)
                f1_values.append(f1)
                specificity_values.append(specificity)
                sensitivity_values.append(sensitivity)
                
            except Exception as e:
                raise ValueError(f"Error reading CSV file {csv_path}: {e}")

    # Save classification metrics data as CSV
    metrics_data = pd.DataFrame({
        'Modality': modalities,
        'F1': f1_values,
        'Specificity': specificity_values,
        'Sensitivity': sensitivity_values
    })
    metrics_data.to_csv(os.path.join(path, f'classification_metrics-{sub2}-{training_type}-{outcome}.csv'), index=False)
    
    print(f"Classification metrics saved for {outcome}")
