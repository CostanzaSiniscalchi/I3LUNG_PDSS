import numpy as np
import pandas as pd
from scipy import stats
import os
import glob
from sklearn.metrics import f1_score, confusion_matrix
from pathlib import Path

# ------------------------------------------------------------------------------
# Configuration for computing F1, Precision, Recall across experiments
# Specify: training type, cohort filters, path structure, and outcomes to analyze
BASE_DIR = Path(__file__).resolve().parents[4]  # Risali a Mil2/
RESULTS_DIR = BASE_DIR / "results"


training_type = 'cross_validation'
sub1 = RESULTS_DIR
sub2 = 'cohort2'
path_pre = 'mil' # new_path
path_suf = f'classification/{training_type}/hypothesis_driven/pyrad-noimp'
outcomes = ['os_months_24', 'os_months_6', 'DCR', 'ORR']
# ------------------------------------------------------------------------------

def find_latest_mb_attention_dir(eval_path):
    """Find the latest (highest numbered) mb_attention_mil directory in the eval path."""
    pattern = os.path.join(eval_path, '*-mb_attention_mil')
    matching_dirs = glob.glob(pattern)
    if not matching_dirs:
        raise FileNotFoundError(f"No directories matching '*-mb_attention_mil' found in {eval_path}")
    
    # Sort directories to get the last one (highest numbered)
    matching_dirs.sort()
    latest_dir = os.path.basename(matching_dirs[-1])
    return latest_dir

def compute_classification_metrics(y_true, y_pred, threshold=0.5):
    """
    Compute F1, specificity, and sensitivity from predictions.
    
    Args:
        y_true: array of true labels (0/1)
        y_pred: array of prediction probabilities
        threshold: threshold for binary classification
    
    Returns:
        dict with f1, specificity, sensitivity
    """
    y_true = np.array(y_true).astype(int)
    y_pred_binary = (np.array(y_pred) >= threshold).astype(int)
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    # Compute metrics
    f1 = f1_score(y_true, y_pred_binary)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        'f1': f1,
        'specificity': specificity,
        'sensitivity': sensitivity
    }

def compute_weighted_metrics(metrics_list, weights):
    """
    Compute weighted average of metrics across folds.
    
    Args:
        metrics_list: list of metric dictionaries
        weights: list of weights (sample sizes)
    
    Returns:
        dict with weighted averages
    """
    if len(metrics_list) == 0:
        return {'f1': 0.0, 'specificity': 0.0, 'sensitivity': 0.0}
    
    weights = np.array(weights)
    total_weight = weights.sum()
    
    if total_weight == 0:
        return {'f1': 0.0, 'specificity': 0.0, 'sensitivity': 0.0}
    
    weighted_f1 = sum(m['f1'] * w for m, w in zip(metrics_list, weights)) / total_weight
    weighted_spec = sum(m['specificity'] * w for m, w in zip(metrics_list, weights)) / total_weight
    weighted_sens = sum(m['sensitivity'] * w for m, w in zip(metrics_list, weights)) / total_weight
    
    return {
        'f1': weighted_f1,
        'specificity': weighted_spec,
        'sensitivity': weighted_sens
    }

# ------------------------------------------------------------------------------

for outcome in outcomes:
    base_path = os.path.join(sub1, sub2, path_pre, outcome, path_suf)

    # Find all subdirectories ending with 'seed_0'
    seed_directories = []
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name == 'seed_0':
                seed_directories.append(os.path.join(root, dir_name))

    # Process each seed directory
    for path in seed_directories:
        print(f"Processing: {path}")
        
        # Check if path does not contain folder called eval (cross-validation case)
        if not os.path.isdir(os.path.join(path, 'eval')):
            # Cross-validation: compute weighted average across folds
            folders = [f.name for f in os.scandir(path) if f.is_dir()]
            
            fold_metrics = []
            fold_weights = []
            
            for folder in folders:
                predictions_file = os.path.join(path, folder, 'predictions.parquet')
                if os.path.exists(predictions_file):
                    # Read predictions
                    predictions = pd.read_parquet(predictions_file)
                    
                    # Compute softmax predictions
                    predictions['pred'] = np.exp(predictions['y_pred1']) / (np.exp(predictions['y_pred0']) + np.exp(predictions['y_pred1']))
                    
                    # Compute metrics for this fold
                    metrics = compute_classification_metrics(predictions['y_true'], predictions['pred'])
                    fold_metrics.append(metrics)
                    fold_weights.append(len(predictions))
                    
                    # Create eval directory structure for compatibility
                    os.makedirs(os.path.join(path, folder, 'eval', '00000-mb_attention_mil'), exist_ok=True)
            
            # Compute weighted average metrics
            final_metrics = compute_weighted_metrics(fold_metrics, fold_weights)
            
        elif os.path.isdir(os.path.join(path, 'eval')):
            # Standard training: use final result directly
            eval_path = os.path.join(path, 'eval')
            latest_mb_dir = find_latest_mb_attention_dir(eval_path)
            predictions_file = os.path.join(eval_path, latest_mb_dir, 'predictions.parquet')
            if os.path.exists(predictions_file):
                # Read predictions
                predictions = pd.read_parquet(predictions_file)
                
                # Compute softmax predictions
                predictions['pred'] = np.exp(predictions['y_pred1']) / (np.exp(predictions['y_pred0']) + np.exp(predictions['y_pred1']))
                
                # Compute metrics
                final_metrics = compute_classification_metrics(predictions['y_true'], predictions['pred'])
            else:
                print(f"Predictions file not found: {predictions_file}")
                continue
        else:
            print(f"No valid structure found in {path}")
            continue
        
        # Save results
        output_file = os.path.join(path, 'eval_classification_metrics.csv')
        with open(output_file, 'w') as f:
            f.write('f1,specificity,sensitivity\n')
            f.write(f'{final_metrics["f1"]:.6f},{final_metrics["specificity"]:.6f},{final_metrics["sensitivity"]:.6f}\n')
        
        print(f"Completed processing for {path}:")
        print(f"  F1 = {final_metrics['f1']:.4f}")
        print(f"  Specificity = {final_metrics['specificity']:.4f}")
        print(f"  Sensitivity = {final_metrics['sensitivity']:.4f}")
        print(f"  Results saved to: {output_file}")

