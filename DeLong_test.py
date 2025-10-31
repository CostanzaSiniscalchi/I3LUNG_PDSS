import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def compute_ground_truth_statistics(ground_truth):
    ground_truth = np.array(ground_truth)
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())

    return order, label_1_count


def delong_test_comparison(y_true, y_pred1, y_pred2, alpha=0.05):
    """
    Compare two ROC curves using DeLong's test for statistical significance.
    
    Args:
        y_true: True binary labels
        y_pred1: Predictions from first model (e.g., biomarker)
        y_pred2: Predictions from second model (e.g., ML model)
        alpha: Significance level (default 0.05)
    
    Returns:
        p-value
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
    auc1_ci = auc_roc_ci(y_true, y_pred1, alpha)[1]
    auc2_ci = auc_roc_ci(y_true, y_pred2, alpha)[1]
    
    return p_value

def p_to_stars(p: float) -> str:
    """Converts a p-value to a significance star string."""
    if p <= 0.0001: return '****'
    if p <= 0.001: return '***'
    if p <= 0.01: return '**'
    if p <= 0.05: return '*'
    return 'ns'

def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov



def auc_roc_ci(y_true, y_pred, alpha):
    """
    Return AUC and CL at alpha for vectors y_true, y_pred
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    auc, auc_cov = delong_roc_variance(
        y_true,
        y_pred)
    
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    
    ci = stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)
    
    ci[ci > 1] = 1
    
    return auc, ci

def compute_midrank(x):
  """Computes midranks.
  Args:
     x - a 1D numpy array
  Returns:
     array of midranks
  """
  J = np.argsort(x)
  Z = x[J]
  N = len(x)
  T = np.zeros(N, dtype=float)
  i = 0
  while i < N:
      j = i
      while j < N and Z[j] == Z[i]:
          j += 1
      T[i:j] = 0.5*(i + j - 1)
      i = j
  T2 = np.empty(N, dtype=float)
  # Note(kazeevn) +1 is due to Python using 0-based indexing
  # instead of 1-based in the AUC formula in the paper
  T2[J] = T + 1
  return T2


def compute_midrank_weight(x, sample_weight):
  """Computes midranks.
  Args:
     x - a 1D numpy array
  Returns:
     array of midranks
  """
  J = np.argsort(x)
  Z = x[J]
  cumulative_weight = np.cumsum(sample_weight[J])
  N = len(x)
  T = np.zeros(N, dtype=float)
  i = 0
  while i < N:
      j = i
      while j < N and Z[j] == Z[i]:
          j += 1
      T[i:j] = cumulative_weight[i:j].mean()
      i = j
  T2 = np.empty(N, dtype=float)
  T2[J] = T
  return T2


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating
              Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov