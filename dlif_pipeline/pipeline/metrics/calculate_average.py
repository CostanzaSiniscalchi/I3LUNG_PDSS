import os
import pandas as pd
import numpy as np
import glob
from scipy.stats import norm

def compute_weighted_average(base_path, task=None):
    """
    Processes cross-validation score files from a classification or survival task
    and computes weighted average and standard deviation statistics.

    Parameters:
    -----------
    base_path : str
        Root directory containing experiment results. The function will recursively
        search for score files inside this path.
        type here is like: base path given be like: hparam_*/seed_*/

    task : str or None
        Optional explicit task type ("classification" or "survival"). If None, auto-detects based on available files.

    Output:
    -------
    A CSV file named `average_std_train_test_scores.csv` will be saved in `base_path`
    with two rows:
        - Mean of selected metrics
        - Standard deviation of selected metrics
    """
    print(f"\n Processing scores in: {base_path}")
    score_files = find_all_score_files(base_path)

    if task is None:
        if score_files['scores']:
            task = "survival"
        elif score_files['scores_test']:
            task = "classification"
        else:
            print(" Could not determine task type.")
            return

    if task == "survival":
        print("🔬 Detected: Survival task")
        compute_and_save_stats(
            score_files['scores_test'],
            output_path=os.path.join(base_path, "average_std_train_test_scores.csv"),
            train_cols=["C_INDEX_TRAIN", "MEAN_AUC_TRAIN", "BRIER_SCORE_TRAIN"],
            test_cols=["C_INDEX_TEST", "MEAN_AUC_TEST", "BRIER_SCORE_TEST"]
        )

    elif task == "classification":
        print(" Detected: Classification task")
        compute_and_save_stats(
            score_files['scores_train'] + score_files['scores_test'],
            output_path=os.path.join(base_path, "average_std_train_test_scores.csv"),
            train_cols=["F1", "AUC", "Sensitivity", "Specificity"],
            test_cols=["F1", "AUC", "Sensitivity", "Specificity"],
            train_weight_col="N",
            test_weight_col="N"
        )
    else:
        print(" Unsupported task:", task)

def find_all_score_files(base_path):
    """Finds all relevant score files in base_path."""
    return {
        "scores": glob.glob(os.path.join(base_path, "**", "scores.csv"), recursive=True),
        "scores_test": glob.glob(os.path.join(base_path, "**", "scores_test.csv"), recursive=True),
        "scores_train": glob.glob(os.path.join(base_path, "**", "scores_train.csv"), recursive=True),
    }

def compute_and_save_stats(
    files,
    output_path,
    train_cols=None,
    test_cols=None,
    train_weight_col="N_TRAIN",
    test_weight_col="N_TEST",
    index_labels=None
):
    all_cols = list((train_cols or []) + (test_cols or []))
    local_labels = list(index_labels) if index_labels else []
    mean_row = {}
    se_row = {}
    ci_lower_row = {}
    ci_upper_row = {}


    values_dict = {col: [] for col in all_cols}
    weights_dict = {col: [] for col in all_cols}

    for file in files:
        try:
            df = pd.read_csv(file)
            df = df.dropna(how="all")
            if df.shape[0] < 1:
                continue

            for col in all_cols:
                if col in df.columns:
                    val = df[col].iloc[0]
                    if pd.isna(val):
                        print(f" Skipping NaN in {file} for column {col}")
                        continue
                    weight_col = (
                        train_weight_col if col in (train_cols or []) else test_weight_col
                    )
                    weight = df[weight_col].iloc[0] if weight_col in df.columns else 1
                    values_dict[col].append(val)
                    weights_dict[col].append(weight)
                else:
                    print(f" Column {col} not found in {file}")
        except Exception as e:
            print(f" Error processing {file}: {e}")

    if not any(len(v) > 0 for v in values_dict.values()):
        print(" No valid values found to compute statistics.")
        return
    # Compute weighted mean, SE and 95% CI
    z = norm.ppf(0.975)  # 1.96-ish for 95% two-sided
    for col in all_cols:
        p = np.array(values_dict[col])
        w = np.array(weights_dict[col])
        if len(p) == 0:
            continue

        W    = w.sum()
        mean = (w * p).sum() / W
        var  = (w**2 * p * (1 - p)).sum() / W**2
        se   = np.sqrt(var)

        mean_row[col]     = round(mean,       4)
        se_row[col]       = round(se,         4)
        ci_lower_row[col] = round(mean - z*se, 4)
        ci_upper_row[col] = round(mean + z*se, 4)

    if not mean_row:
        print(" No valid values found to compute statistics.")
        return

    df_out = pd.DataFrame([
        mean_row,
        se_row,
        ci_lower_row,
        ci_upper_row
    ], index=[
        "Weighted Mean",
        "Standard Error",
        "95% CI Lower",
        "95% CI Upper"
    ])

    df_out.to_csv(output_path)
    print(f" Saved CLT‐based stats to {output_path}")

