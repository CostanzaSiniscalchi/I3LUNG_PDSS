import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from DeLong_test import auc_roc_ci, delong_roc_variance, fastDeLong_no_weights, compute_ground_truth_statistics
from scipy import stats
from typing import List, Dict

def add_single_biomarkers(outcome: str, df: pd.DataFrame, to_show: list[str]) -> tuple[pd.DataFrame, list[str]]: 
        """
        Create the dataframe with the results of the biomarkers.
        
        Args:
            outcome (str): The outcome to use for the prediction.
            df (pd.DataFrame): The dataframe with biomarkers data.
            to_show (list[str]): The list of biomarkers to show.
        
        Returns:
            tuple[pd.DataFrame, list[str]]: The dataframe with the scores of the desired biomarkers.
        """       
        
        biomarkers = pd.DataFrame(index=df.index)
        if 'BIO_PDL1_50' in to_show:
            biomarkers['BIO_PDL1_50'] = df.apply(lambda x: 
                'TP' if (x[outcome] == 1 and x['PDL1 CATEGORY'] >= 2) else
                'FP' if (x[outcome] == 0 and x['PDL1 CATEGORY'] >= 2) else
                'FN' if (x[outcome] == 1 and x['PDL1 CATEGORY'] < 2) else
                'TN' if (x[outcome] == 0 and x['PDL1 CATEGORY'] < 2) else
                np.nan, axis=1
            )
        if 'BIO_PDL1_1' in to_show:
            biomarkers['BIO_PDL1_1'] = df.apply(lambda x: 
                'TP' if (x[outcome] == 1 and x['PDL1 CATEGORY'] >= 1) else 
                'FP' if (x[outcome] == 0 and x['PDL1 CATEGORY'] >= 1) else
                'FN' if (x[outcome] == 1 and x['PDL1 CATEGORY'] < 1) else
                'TN' if (x[outcome] == 0 and x['PDL1 CATEGORY'] < 1) else
                np.nan, axis=1
            )
        if 'BIO_NLR' in to_show:
            biomarkers['BIO_NLR'] = df.apply(lambda x: 
                'TP' if (x[outcome] == 1 and x['NLR'] <= 4) else 
                'FP' if (x[outcome] == 0 and x['NLR'] <= 4) else
                'FN' if (x[outcome] == 1 and x['NLR'] > 4) else
                'TN' if (x[outcome] == 0 and x['NLR'] > 4) else
                np.nan, axis=1
            )
        if 'BIO_LDH' in to_show:
            biomarkers['BIO_LDH'] = df.apply(lambda x:
                'TP' if (x[outcome] == 1 and x['LDH'] <= 400) else
                'FP' if (x[outcome] == 0 and x['LDH'] <= 400) else
                'FN' if (x[outcome] == 1 and x['LDH'] > 400) else
                'TN' if (x[outcome] == 0 and x['LDH'] > 400) else
                np.nan, axis=1
            )
        if 'BIO_ECOG' in to_show:
            biomarkers['BIO_ECOG'] = df.apply(lambda x:
                'TP' if (x[outcome] == 1 and x['ECOG PS'] in [0, 1]) else
                'FP' if (x[outcome] == 0 and x['ECOG PS'] in [0, 1]) else
                'FN' if (x[outcome] == 1 and x['ECOG PS'] in [2, 3, 4]) else
                'TN' if (x[outcome] == 0 and x['ECOG PS'] in [2, 3, 4]) else
                np.nan, axis=1
            )
        
        
        df = pd.concat([df, biomarkers], axis=1)
        
        return df, biomarkers.columns.tolist()
def convert_to_binary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Convert the specified columns in the dataframe to binary values.
    
    Args:
        df (pd.DataFrame): The dataframe with the biomarkers data.
        columns (list[str]): The list of columns to convert.
    
    Returns:
        pd.DataFrame: The dataframe with the specified columns converted to binary values.
    """
    for col in columns:
        df[f'binary_{col}'] = df[col].map({'TP': 1, 'FP': 1, 'FN': 0, 'TN': 0})
    return df
#compute AUC, accuracy, sensitivity, specificity for each biomarker
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def compute_metrics_biomarker(y_true, y_pred):
    """
    Compute AUC, accuracy, sensitivity, and specificity.
    
    Args:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
    
    Returns:
        dict: Dictionary with AUC, accuracy, sensitivity, and specificity.
    """
    auc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'AUC': auc,
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    }

def compute_metrics(y_true, y_pred, y_proba):
    """
    Compute AUC, accuracy, sensitivity, and specificity.
    
    Args:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
        y_proba (pd.Series): Predicted probabilities.
    
    Returns:
        dict: Dictionary with AUC, accuracy, sensitivity, and specificity.
    """
    auc = roc_auc_score(y_true, y_proba)
    accuracy = accuracy_score(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    tn, fp, fn, tp = map(int, [tn, fp, fn, tp])
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'AUC': auc,
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    }


def delong_test_comparison(y_true, y_pred1, y_pred2, alpha=0.05):
    """
    Compare two ROC curves using DeLong's test for statistical significance.
    Compute p-value for the difference between two AUCs.
    
    Args:
        y_true: True binary labels
        y_pred1: Predictions from first model (e.g., biomarker)
        y_pred2: Predictions from second model (e.g., ML model)
        alpha: Significance level (default 0.05)
    
    Returns:
        dict: Contains AUCs, p-value, and significance test results
    """
    # Compute AUC and variance for each model
    auc1, var1 = delong_roc_variance(y_true, y_pred1)
    auc2, var2 = delong_roc_variance(y_true, y_pred2)
    
    # Compute covariance between the two AUCs
    # For this we need to combine predictions and compute joint variance
    predictions_combined = np.array([y_pred1, y_pred2])
    order, label_1_count = compute_ground_truth_statistics(y_true)
    predictions_sorted_transposed = predictions_combined[:, order]
    aucs, cov_matrix = fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    
    # Extract covariance
    cov_12 = cov_matrix[0, 1] if cov_matrix.ndim > 1 else 0
    
    # Compute test statistic
    auc_diff = auc1 - auc2
    var_diff = var1 + var2 - 2 * cov_12
    
    if var_diff <= 0:
        var_diff = 1e-10  # Avoid division by zero
    
    z_score = auc_diff / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
    
    # Get confidence intervals
    auc1_ci = auc_roc_ci(y_true, y_pred1, 0.95)[1]
    auc2_ci = auc_roc_ci(y_true, y_pred2, 0.95)[1]
    
    return {
        'auc1': auc1,
        'auc2': auc2,
        'auc1_ci': auc1_ci,
        'auc2_ci': auc2_ci,
        'auc_difference': auc_diff,
        'p_value': p_value,
        'z_score': z_score,
        'significant': p_value < alpha,
        'alpha': alpha
    }


def p_to_stars(p: float) -> str:
    """Converts a p-value to a significance star string."""
    if p <= 0.0001: return '****'
    if p <= 0.001: return '***'
    if p <= 0.01: return '**'
    if p <= 0.05: return '*'
    return 'ns'

def plot_radar_charts(
    biomarker_scores: Dict,
    model_scores: Dict,
    dl_scores: Dict,
    p_values_ml:  List[float],
    p_values_dl:  List[float],
    size: List[int],
    biomarkers: List[str],
    metrics: List[str],
    title_prefix: str = 'Comparison'
):
    """
    Generates and displays a series of radar charts comparing biomarker and model performance.

    Args:
        biomarker_scores (Dict): A dictionary containing metric scores for each biomarker.
        model_scores (Dict): A dictionary containing MLEF metric's scores.
        dl_scores (Dict, optional): A dictionary containing DLIF metric's scores.
        p_values_ml (List[float]): A list of MLEF p-values for each biomarker, used for significance annotations.
        p_values_dl (List[float], optional): A list of DLIF p-values for each biomarker, used for significance annotations.
        size (List[int]): A list of sample sizes corresponding to each biomarker.
        biomarkers (List[str]): A list of biomarker names for the chart axes (e.g., ['PDL1', 'PDL1_50']).
        metrics (List[str]): A list of performance metrics to plot (e.g., ['AUC', 'Accuracy']).
        title_prefix (str, optional): A prefix for the chart title. Defaults to 'Comparison'.
    """
    


    N = len(biomarkers)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1] # Close the loop
    fontsize = 12
    # --- Plotting Loop ---
    for metric in metrics:
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, projection='polar')
        
        vals_bio = [biomarker_scores[bio][metric] for bio in biomarkers] + [biomarker_scores[biomarkers[0]][metric]]
        vals_model = [model_scores[bio][metric] for bio in biomarkers] + [model_scores[biomarkers[0]][metric]]
        vals_dl = [dl_scores[bio][metric] for bio in biomarkers] + [dl_scores[biomarkers[0]][metric]] if dl_scores else None

        # Plotting the data
        ax.plot(angles, vals_bio, color="#811850", linewidth=2, label='Biomarkers')
        ax.fill(angles, vals_bio, color='#811850', alpha=0.5)
        ax.plot(angles, vals_model, color="#156ba9", linewidth=2, label='MLEF')
        ax.fill(angles, vals_model, color='#156ba9', alpha=0.5)
        if vals_dl is not None:
            ax.plot(angles, vals_dl, color="#477439", linewidth=2, label='DLIF')
            ax.fill(angles, vals_dl, color="#477439", alpha=0.5)

        
        # --- Labels and Ticks ---
        # if metric == 'AUC':
        #     tick_labels = [f"{b} {p_to_stars(p_values[i])}\nn={size[i]}" for i, b in enumerate(biomarkers)]
        # else:
        tick_labels = [f"{b}\nn={size[i]}" for i, b in enumerate(biomarkers)]
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tick_labels, fontsize=fontsize)
        ax.tick_params(axis='x', pad=42)
        
        ax.set_ylim(0, 1)
        ax.set_title(f'{title_prefix} - {metric} Comparison', pad=25, fontsize=14)
        
        # --- Annotations ---
        colors = ['#811850', '#156ba9', "#477439"]
        for i, angle in enumerate(angles[:-1]):
            if metric == 'AUC':
                bio_val_text = f"Biomarker: {vals_bio[i]:.2f}"
                model_val_text = f"MLEF: {vals_model[i]:.2f} {p_to_stars(p_values_ml[i])}"
                if vals_dl is not None:
                    dl_val_text = f"DLIF: {vals_dl[i]:.2f} {p_to_stars(p_values_dl[i])}"
            else:
                bio_val_text = f"Biomarker: {vals_bio[i]:.2f}"
                model_val_text = f"MLEF: {vals_model[i]:.2f}"
                if vals_dl is not None:
                    dl_val_text = f"DLIF: {vals_dl[i]:.2f}"
            angle_deg = np.rad2deg(angle)

            if 85 < angle_deg < 95: # Top
                ha = 'center'
                ax.text(angle, 1.14, model_val_text, color=colors[1], ha=ha, va='bottom', fontsize=fontsize)
                ax.text(angle, 1.0, bio_val_text, color=colors[0], ha=ha, va='bottom', fontsize=fontsize)
                if vals_dl is not None:
                    ax.text(angle, 1.07, dl_val_text, color=colors[2], ha=ha, va='bottom', fontsize=fontsize)
            elif 265 < angle_deg < 275: # Bottom
                ha = 'center'
                ax.text(angle, 1.01, model_val_text, color=colors[1], ha=ha, va='top', fontsize=fontsize)
                ax.text(angle, 1.16, bio_val_text, color=colors[0], ha=ha, va='top', fontsize=fontsize)
                if vals_dl is not None:
                    ax.text(angle, 1.08, dl_val_text, color=colors[2], ha=ha, va='top', fontsize=fontsize)
            elif angle_deg < 15 or angle_deg > 345: # Right
                ha = 'left'
                ax.text(angle - 0.16, 1.1, model_val_text, color=colors[1], ha=ha, va='bottom', fontsize=fontsize)
                ax.text(angle - 0.22, 1.1, bio_val_text, color=colors[0], ha=ha, va='top', fontsize=fontsize)
                if vals_dl is not None:
                    ax.text(angle - 0.16, 1.11, dl_val_text, color=colors[2], ha=ha, va='top', fontsize=fontsize)
            else: # Left
                ha = 'right'
                ax.text(angle + 0.16, 1.1, model_val_text, color=colors[1], ha=ha, va='bottom', fontsize=fontsize)
                ax.text(angle + 0.22, 1.1, bio_val_text, color=colors[0], ha=ha, va='top', fontsize=fontsize)
                if vals_dl is not None:
                    ax.text(angle + 0.16, 1.1, dl_val_text, color=colors[2], ha=ha, va='top', fontsize=fontsize)


         # --- ADD SIGNIFICANCE LEGEND ONLY FOR AUC PLOT ---
        if metric == 'AUC':
            significance_text = (
                'P-value significance:\n'
                r'ns: P > 0.05' '\n'
                r'*: P $\leq$ 0.05' '\n'
                r'**: P $\leq$ 0.01' '\n'
                r'***: P $\leq$ 0.001' '\n'
                r'****: P $\leq$ 0.0001'
            )
            # Add text to the figure, positioned at the bottom left
            plt.figtext(
                0.2, 0.9,
                significance_text,
                ha="right", va="top",
                fontsize=9,
                color="#474545",
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
            )

        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=fontsize)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.2)
        plt.show()