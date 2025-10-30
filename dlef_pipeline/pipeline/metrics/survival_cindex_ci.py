import numpy as np
import pandas as pd
from scipy import stats
import os
import glob
from sksurv.metrics import concordance_index_censored
from pathlib import Path
# ------------------------------------------------------------------------------
# Configuration for computing c-index across experiments
# Specify: training type, cohort filters, path structure, and outcomes to analyze

BASE_DIR = Path(__file__).resolve().parents[4]  # Risali a Mil2/
RESULTS_DIR = BASE_DIR / "results"

training_type = 'standard'
sub1 = RESULTS_DIR
sub2 = 'adeno'
path_pre = 'mil' # new_path
path_suf = f'survival/{training_type}/hypothesis_driven/pyrad-noimp'
outcomes = ['OS_MONTHS'] # can be extended for other survival outcomes

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

def compute_c_index_and_ci(event, time, risk_score, n_boot=1000, alpha=0.05, seed=42):
    """
    Bootstrap estimation of C-index and percentile confidence interval.

    Parameters:
        event, time, risk_score: arrays
        n_boot: number of bootstrap replicates
        alpha: significance level
        seed: reproducibility

    Returns:
        dict with:
            - c_index: original point estimate
            - ci: (lower, upper) confidence interval
            - bootstrap_distribution: list of all bootstrapped estimates
    """
    np.random.seed(seed)
    n = len(time)
    c_indexes = []

    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        c_idx = concordance_index_censored(event[idx], time[idx], risk_score[idx])[0]
        if not np.isnan(c_idx):
            c_indexes.append(c_idx)

    c_index_orig = concordance_index_censored(event, time, risk_score)[0]
    ci_lower = np.percentile(c_indexes, 100 * (alpha / 2))
    ci_upper = np.percentile(c_indexes, 100 * (1 - alpha / 2))

    return {
        "c_index": c_index_orig,
        "ci": (ci_lower, ci_upper),
        "bootstrap_distribution": c_indexes
    }

def compute_cindex_ci(c_index_values, fold_sizes, alpha=0.05):
    """
    Compute confidence interval for C-index using weighted average and t-distribution.
    
    Args:
        c_index_values: array of C-index values from each fold
        fold_sizes: array of test set sizes for each fold
        alpha: significance level (default 0.05 for 95% CI)
    
    Returns:
        score_mean: weighted mean C-index
        ci_lower: lower bound of confidence interval
        ci_upper: upper bound of confidence interval
    """
    c_index_cv = np.array(c_index_values)
    fold_sizes = np.array(fold_sizes)
    
    # Filter out NaN values
    valid = ~np.isnan(c_index_cv)
    c_index_cv = c_index_cv[valid]
    fold_sizes = fold_sizes[valid]
    
    if len(c_index_cv) == 0:
        raise ValueError("No valid C-index values found after filtering NaN values")
    
    # Compute weighted mean and variance
    score_mean = np.average(c_index_cv, weights=fold_sizes)
    var = np.average((c_index_cv - score_mean)**2, weights=fold_sizes)
    score_std = np.sqrt(var)
    
    # Compute effective sample size
    n_eff = np.sum(fold_sizes)**2 / np.sum(fold_sizes**2)
    
    # Compute confidence interval using t-distribution
    t_crit = stats.t.ppf(1 - alpha / 2, df=n_eff - 1)
    margin_error = t_crit * (score_std / np.sqrt(n_eff))
    
    ci_lower = score_mean - margin_error
    ci_upper = score_mean + margin_error
    
    return score_mean, ci_lower, ci_upper

# ------------------------------------------------------------------------------

for outcome in outcomes:
    base_path = os.path.join(sub1, sub2, path_pre, outcome, path_suf)
    print(f"Processing outcome: {outcome}")
    print(f"Base path: {base_path}")


    # Find all subdirectories ending with 'seed_0'
    seed_directories = []
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name == 'seed_0':
                seed_directories.append(os.path.join(root, dir_name))

    # Process each seed directory
    for seed_path in seed_directories:
        print(f"Processing: {seed_path}")
        
        if training_type == 'standard':
            # For standard training, eval data is directly in the folder
            if os.path.isdir(os.path.join(seed_path, 'eval')):
                # Read predictions from eval directory
                eval_path = os.path.join(seed_path, 'eval')
                latest_mb_dir = find_latest_mb_attention_dir(eval_path)
                predictions_file = os.path.join(eval_path, latest_mb_dir, 'predictions.parquet')
                
                if os.path.exists(predictions_file):
                    try:
                        # Read predictions
                        predictions = pd.read_parquet(predictions_file)
                        
                        # Check required columns exist in predictions
                        required_cols = ['y_true0', 'y_true1', 'y_pred0']
                        if not all(col in predictions.columns for col in required_cols):
                            raise ValueError(f"Required columns {required_cols} not found in predictions file")
                        
                        # Extract data directly from predictions file
                        # y_true0 = OS_MONTHS (survival time)
                        # y_true1 = death event
                        # y_pred0 = prediction score
                        time = predictions['y_true0'].values.astype(float)
                        event = predictions['y_true1'].values.astype(bool)
                        risk_score = -predictions['y_pred0'].values.astype(float)  # Negative for proper interpretation
                        
                        # Remove any NaN values
                        valid_mask = ~(np.isnan(event) | np.isnan(time) | np.isnan(risk_score))
                        event = event[valid_mask]
                        time = time[valid_mask]
                        risk_score = risk_score[valid_mask]
                        
                        if len(event) == 0:
                            raise ValueError("No valid samples remaining after removing NaN values")
                        
                        # Compute C-index with bootstrap CI
                        result = compute_c_index_and_ci(event, time, risk_score)
                        
                        # Save results
                        output_file = os.path.join(seed_path, 'eval_cindex_ci.csv')
                        with open(output_file, 'w') as f:
                            f.write('cindex_mean,ci_lower,ci_upper,n_samples,method\n')
                            f.write(f'{result["c_index"]},{result["ci"][0]},{result["ci"][1]},{len(event)},bootstrap\n')
                        
                        print(f"Completed processing for {seed_path}:")
                        print(f"  C-Index = {result['c_index']:.4f}")
                        print(f"  95% CI = [{result['ci'][0]:.4f}, {result['ci'][1]:.4f}]")
                        print(f"  Number of samples = {len(event)}")
                        print(f"  Results saved to: {output_file}")
                        
                    except Exception as e:
                        print(f"Error processing {predictions_file}: {str(e)}")
                        raise
                else:
                    print(f"Predictions file not found: {predictions_file}")
            else:
                print(f"Eval directory not found in {seed_path}")
        
        else:
            # For cross_validation training, process fold-based structure
            c_index_values = []
            fold_sizes = []
            fold_paths = []
            
            # Look for fold directories
            if os.path.isdir(seed_path):
                for item in os.listdir(seed_path):
                    item_path = os.path.join(seed_path, item)
                    if os.path.isdir(item_path) and item.startswith('fold_'):
                        eval_path = os.path.join(item_path, 'eval')
                        latest_mb_dir = find_latest_mb_attention_dir(eval_path)
                        scores_file = os.path.join(eval_path, latest_mb_dir, 'scores_test.csv')
                        
                        if os.path.exists(scores_file):
                            try:
                                # Read the scores CSV file
                                scores_df = pd.read_csv(scores_file)
                                
                                if len(scores_df) == 0:
                                    raise ValueError(f"Empty scores file: {scores_file}")
                                
                                # Extract C_INDEX_TEST and N_TEST
                                c_index_test = scores_df['C_INDEX_TEST'].iloc[0]
                                n_test = scores_df['N_TEST'].iloc[0]
                                
                                # Check for missing C_INDEX_TEST
                                if pd.isna(c_index_test) or c_index_test == '':
                                    raise ValueError(f"Missing C_INDEX_TEST in {scores_file}")
                                
                                # Check for missing N_TEST
                                if pd.isna(n_test) or n_test == '':
                                    raise ValueError(f"Missing N_TEST in {scores_file}")
                                
                                c_index_values.append(float(c_index_test))
                                fold_sizes.append(int(n_test))
                                fold_paths.append(item_path)
                                
                            except Exception as e:
                                print(f"Error processing {scores_file}: {str(e)}")
                                raise
            
            if len(c_index_values) == 0:
                print(f"No valid fold data found in {seed_path}")
                continue
            
            try:
                # Compute C-index confidence interval using weighted average
                score_mean, ci_lower, ci_upper = compute_cindex_ci(c_index_values, fold_sizes)
                
                # Save results
                output_file = os.path.join(seed_path, 'eval_cindex_ci.csv')
                with open(output_file, 'w') as f:
                    f.write('cindex_mean,ci_lower,ci_upper,n_folds,total_test_samples,method\n')
                    f.write(f'{score_mean},{ci_lower},{ci_upper},{len(c_index_values)},{sum(fold_sizes)},weighted_average\n')
                
                print(f"Completed processing for {seed_path}:")
                print(f"  C-Index Mean = {score_mean:.4f}")
                print(f"  95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
                print(f"  Number of folds = {len(c_index_values)}")
                print(f"  Total test samples = {sum(fold_sizes)}")
                print(f"  Results saved to: {output_file}")
                
            except Exception as e:
                print(f"Error computing CI for {seed_path}: {str(e)}")
                raise
