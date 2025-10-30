import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score

def softmax(x):
    """
    Numerically stable softmax function for logits.
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def calculate_classification_metrics(df, filename):
    """
    Compute classification metrics and save to CSV.
    """
    scores = {}
    y_true = df["y_true"].values
    pred_cols = [c for c in df.columns if c.startswith("y_pred")]
    y_pred_logits = df[pred_cols].values
    y_pred_probs = softmax(y_pred_logits)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)

    scores["F1"] = f1_score(y_true, y_pred_labels, average='macro')
    scores["AUC"] = roc_auc_score(y_true, y_pred_probs[:, 1], multi_class='ovr')
    scores["Sensitivity"] = recall_score(y_true, y_pred_labels, average='macro')
    scores["Specificity"] = precision_score(y_true, y_pred_labels, average='macro', zero_division=0)
    scores["N"] = len(df)

    pd.DataFrame([scores]).to_csv(filename, index=False)
    print(f"✅ Saved classification scores to {filename} (N={scores['N']})")
