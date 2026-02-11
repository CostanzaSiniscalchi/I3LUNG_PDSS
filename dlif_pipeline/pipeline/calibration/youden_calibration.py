import pandas as pd                                                                                                                                                                                       
import numpy as np
from pathlib import Path                                                                                                                                                                                  
from sklearn.metrics import roc_curve                                                                                                                                                                     

# Adjust this import to wherever delong_roc_variance lives in this project
from lung_helpers import delong_roc_variance

RESULTS_DIR = Path('/home/ludovica/I3LUNG_PDSS/dlif_pipeline/results')
SCORE_COL = 'y_pred1'
LABEL_COL = 'y_true'


def calibrate_cv_scores(df_cv, exclude_slides=None):
    """
    Calibrate combined CV (LOCO) predictions using Youden index + unit variance scaling.
    Threshold and std are calculated excluding train slides, but applied to all.
    """
    labels_all = df_cv[LABEL_COL].values
    scores_all = df_cv[SCORE_COL].values

    # Subset for threshold calculation (exclude train)
    if exclude_slides is not None:
        mask = ~df_cv['slide'].astype(str).isin(exclude_slides)
        labels_thr = df_cv.loc[mask, LABEL_COL].values
        scores_thr = df_cv.loc[mask, SCORE_COL].values
        print(f"  Threshold calculated on {mask.sum()}/{len(df_cv)} samples (excluded {(~mask).sum()} train)")
    else:
        labels_thr = labels_all
        scores_thr = scores_all

    # Youden threshold on non-train only
    fpr, tpr, thresholds = roc_curve(labels_thr, scores_thr)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]

    # Unit variance on non-train only
    score_std = np.std(scores_thr)
    threshold_scaled = optimal_threshold / score_std if score_std > 0 else optimal_threshold

    # Apply scaling to ALL samples
    scores_scaled = scores_all / score_std if score_std > 0 else scores_all

    # AUC with DeLong on all
    auc, auc_cov = delong_roc_variance(labels_all, scores_scaled)
    auc_std = np.sqrt(auc_cov)

    df_cv['score_calibrated'] = scores_scaled
    df_cv['prediction'] = (scores_scaled > threshold_scaled).astype(int)
    df_cv['youden_threshold'] = threshold_scaled

    calibration_params = {'threshold': optimal_threshold, 'score_std': score_std}
    metrics = {
        'auc': auc,
        'ci_lower': auc - 1.96 * auc_std,
        'ci_upper': auc + 1.96 * auc_std,
        'threshold': threshold_scaled
    }

    print(f"  CV: Youden={optimal_threshold:.3f}, scaled={threshold_scaled:.3f}, std={score_std:.3f}")
    print(f"  CV AUC: {metrics['auc']:.3f} (95% CI: {metrics['ci_lower']:.3f}-{metrics['ci_upper']:.3f})")

    return df_cv, metrics, calibration_params

def apply_calibration(df, calibration_params):
    """Apply CV-derived calibration to standard/evaluation predictions."""
    scores = df[SCORE_COL].values
    threshold = calibration_params['threshold']
    score_std = calibration_params['score_std']

    threshold_scaled = threshold / score_std if score_std > 0 else threshold
    scores_scaled = scores / score_std if score_std > 0 else scores

    df['score_calibrated'] = scores_scaled
    df['prediction'] = (scores_scaled > threshold_scaled).astype(int)
    df['youden_threshold'] = threshold_scaled

    labels = df[LABEL_COL].values
    auc, auc_cov = delong_roc_variance(labels, scores_scaled)
    auc_std = np.sqrt(auc_cov)
    metrics = {
        'auc': auc,
        'ci_lower': auc - 1.96 * auc_std,
        'ci_upper': auc + 1.96 * auc_std
    }

    return df, metrics

def calibrate_all():
    """
    Auto-discover all (cohort, outcome, extractor, modality) combinations.
    For each: combine CV folds → Youden threshold → apply to standard & evaluation.
    """
    cv_pattern = '*/*/classification/cross_validation/hypothesis_driven/*/*/seed_0'
    combinations = set()
    for path in RESULTS_DIR.glob(cv_pattern):
        parts = path.relative_to(RESULTS_DIR).parts
        # (cohort, outcome, 'classification', 'cross_validation', 'hypothesis_driven', extractor, modality, 'seed_0')
        combinations.add((parts[0], parts[1], parts[5], parts[6]))

    combinations = sorted(combinations)
    print(f"Found {len(combinations)} combinations\n")

    for cohort, outcome, extractor, modality in combinations:
        print(f"\n{'='*60}")
        print(f"{cohort} | {outcome} | {extractor} | {modality}")
        print(f"{'='*60}")

        base_path = RESULTS_DIR / cohort / outcome / 'classification'
        hd_path = 'hypothesis_driven' / Path(extractor) / modality / 'seed_0'

        # --- Step 1: Combine all CV fold predictions ---
        cv_base = base_path / 'cross_validation' / hd_path
        fold_dirs = sorted(cv_base.glob('fold_*'))

        if not fold_dirs:
            print("  No CV folds found, skipping")
            continue

        cv_dfs = []
        for fold_dir in fold_dirs:
            pred_file = fold_dir / 'predictions.parquet'
            if pred_file.exists():
                df_fold = pd.read_parquet(pred_file)
                df_fold['fold'] = fold_dir.name
                cv_dfs.append(df_fold)

        if not cv_dfs:
            print("  No predictions found in CV folds, skipping")
            continue

        df_cv = pd.concat(cv_dfs, ignore_index=True)
        print(f"  Combined {len(cv_dfs)} folds, {len(df_cv)} samples")

        # --- Step 2: Calibrate CV, get threshold ---
        df_cv_cal, cv_metrics, calibration_params = calibrate_cv_scores(df_cv)
        df_cv_cal.to_parquet(cv_base / 'predictions_calibrated.parquet')
        print(f"  Saved CV calibrated")

        # --- Step 3: Apply to standard ---
        std_pred = base_path / 'standard' / hd_path / 'predictions.parquet'
        if std_pred.exists():
            df_std = pd.read_parquet(std_pred)
            df_std_cal, std_metrics = apply_calibration(df_std, calibration_params)
            df_std_cal.to_parquet(std_pred.parent / 'predictions_calibrated.parquet')
            print(f"  Standard AUC: {std_metrics['auc']:.3f} "
                f"(95% CI: {std_metrics['ci_lower']:.3f}-{std_metrics['ci_upper']:.3f})")

        # --- Step 4: Apply to evaluation ---
        eval_base = base_path / 'evaluation' / hd_path / 'eval'
        if eval_base.exists():
            for eval_dir in sorted(eval_base.iterdir()):
                eval_pred = eval_dir / 'predictions.parquet'
                if eval_pred.exists():
                    df_eval = pd.read_parquet(eval_pred)
                    df_eval_cal, eval_metrics = apply_calibration(df_eval, calibration_params)
                    df_eval_cal.to_parquet(eval_dir / 'predictions_calibrated.parquet')
                    print(f"  Eval ({eval_dir.name}) AUC: {eval_metrics['auc']:.3f} "
                        f"(95% CI: {eval_metrics['ci_lower']:.3f}-{eval_metrics['ci_upper']:.3f})")