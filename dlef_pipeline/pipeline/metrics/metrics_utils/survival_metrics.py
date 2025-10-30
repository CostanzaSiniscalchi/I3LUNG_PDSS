import pandas as pd
import numpy as np
import os
from sksurv.metrics import concordance_index_censored, integrated_brier_score, cumulative_dynamic_auc
from datetime import datetime

def calculate_survival_metrics(predictions_train_path, predictions_test_path):
    """
    Compute survival analysis metrics using precomputed predictions.
    Supports missing train path.
    """

    # Carica le predizioni di test (necessarie sempre)
    test_data = pd.read_parquet(predictions_test_path)
    test_data.rename(columns={"y_true0": "TIME", "y_true1": "EVENT", "y_pred0": "PRED"}, inplace=True)
    test_data["EVENT"] = test_data["EVENT"].astype(bool)
    y_test = np.array([(e, t) for e, t in zip(test_data["EVENT"], test_data["TIME"])],
                      dtype=[("EVENT", bool), ("TIME", float)])
    pred_test = -test_data["PRED"].values  # Invertiamo il segno


    ci_train = brier_train = auc_train = None  # Defaults if train not available

    # Se esiste il file train, lo carica e calcola le metriche
    if predictions_train_path and os.path.exists(predictions_train_path):
        train_data = pd.read_parquet(predictions_train_path)
        train_data.rename(columns={"y_true0": "TIME", "y_true1": "EVENT", "y_pred0": "PRED"}, inplace=True)
        train_data["EVENT"] = train_data["EVENT"].astype(bool)

        y_train = np.array([(e, t) for e, t in zip(train_data["EVENT"], train_data["TIME"])],
                           dtype=[("EVENT", bool), ("TIME", float)])
        pred_train = -train_data["PRED"].values

        # Concordance Index (train)
        ci_train = concordance_index_censored(train_data["EVENT"], train_data["TIME"], pred_train)[0]
    else:
        print("⚠️ Train file not found or path is None. Skipping train metrics.")

    # Concordance Index (test)
    ci_test = concordance_index_censored(test_data["EVENT"], test_data["TIME"], pred_test)[0]

    # Time points
    t_min, t_max = y_test["TIME"].min(), y_test["TIME"].max()
    allowed_t_max = min(t_max, 109.0013140604468)
    times = np.linspace(t_min + 1e-5, allowed_t_max - 1e-5, 30)

    # Sopravvivenza stimata per test (train se presente)
    baseline_hazard = 0.05
    surv_func_test = np.exp(-baseline_hazard * np.exp(pred_test)[:, None] * times)

    if predictions_train_path and os.path.exists(predictions_train_path):
        surv_func_train = np.exp(-baseline_hazard * np.exp(pred_train)[:, None] * times)
        brier_train = integrated_brier_score(y_train, y_train, surv_func_train, times)
        auc_train = cumulative_dynamic_auc(y_train, y_train, pred_train, times)[1]

    # Brier e AUC per test (usano sempre y_train se disponibile, altrimenti y_test come base)
    brier_test = integrated_brier_score(y_train if ci_train else y_test, y_test, surv_func_test, times)
    auc_test = cumulative_dynamic_auc(y_train if ci_train else y_test, y_test, pred_test, times)[1]

    return {
        "C_INDEX_TRAIN": ci_train,
        "C_INDEX_TEST": ci_test,
        "MEAN_AUC_TRAIN": auc_train,
        "MEAN_AUC_TEST": auc_test,
        "BRIER_SCORE_TRAIN": brier_train,
        "BRIER_SCORE_TEST": brier_test,
        "LAST_RUN": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
