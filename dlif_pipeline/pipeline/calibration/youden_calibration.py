import sys                                                                                                                                                                                                
import numpy as np                                                                                                                                                                                        
import pandas as pd                                                                                                                                                                                       
from pathlib import Path
from sklearn.metrics import roc_curve, accuracy_score, recall_score, f1_score, confusion_matrix                                                                                                           
                                                                                                                                                                                                        
sys.path.append(str(Path(__file__).resolve().parents[2]))
from pipeline.metrics.delong_n import delong_roc_variance

BASE_DIR = Path(__file__).resolve().parents[3]
RESULTS_DIR = BASE_DIR / 'dlif_pipeline' / 'results'
ANNOTATIONS_PATH = BASE_DIR / 'data' / 'annotations.csv'
LABEL_COL = 'y_true'


def compute_score(df):
    """Compute softmax probability for positive class from logits."""
    df['score'] = np.exp(df['y_pred1']) / (np.exp(df['y_pred0']) + np.exp(df['y_pred1']))
    return df


def print_metrics(labels, predictions, prefix=""):
    """Print classification metrics for debugging."""
    acc = accuracy_score(labels, predictions)
    sens = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"  {prefix}Acc={acc:.3f} | Sens={sens:.3f} | Spec={spec:.3f} | F1={f1:.3f} | (TP={tp} FP={fp} FN={fn} TN={tn})")
    return {'accuracy': acc, 'sensitivity': sens, 'specificity': spec, 'f1': f1}


def calibrate_cv_scores(df_cv, exclude_slides=None):
    """
    Calibrate combined CV (LOCO) predictions using Youden index + unit variance scaling.
    Threshold and std calculated excluding test slides, applied to all.
    """
    labels_all = df_cv[LABEL_COL].values
    scores_all = df_cv['score'].values

    # Before calibration (median threshold)
    pred_before = (scores_all > np.median(scores_all)).astype(int)
    print(f"  [BEFORE] median threshold={np.median(scores_all):.3f}")
    print_metrics(labels_all, pred_before, prefix="[BEFORE] ")

    # Subset for threshold calculation (exclude test)
    if exclude_slides is not None:
        mask = ~df_cv['slide'].astype(str).isin(exclude_slides)
        labels_thr = df_cv.loc[mask, LABEL_COL].values
        scores_thr = df_cv.loc[mask, 'score'].values
        print(f"  Threshold on {mask.sum()}/{len(df_cv)} samples (excluded {(~mask).sum()} test)")
    else:
        labels_thr = labels_all
        scores_thr = scores_all

    # Youden on non-train only
    fpr, tpr, thresholds = roc_curve(labels_thr, scores_thr)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    print(f"  Youden index={youden_index[optimal_idx]:.3f} at threshold={optimal_threshold:.3f}")

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

    # After calibration (Youden threshold)
    print(f"  [AFTER] Youden={optimal_threshold:.3f}, scaled={threshold_scaled:.3f}, std={score_std:.3f}")
    cal_metrics = print_metrics(labels_all, df_cv['prediction'].values, prefix="[AFTER]  ")

    calibration_params = {'threshold': optimal_threshold, 'score_std': score_std}
    metrics = {
        'auc': auc,
        'ci_lower': auc - 1.96 * auc_std,
        'ci_upper': auc + 1.96 * auc_std,
        'threshold': threshold_scaled,
        **cal_metrics
    }

    print(f"  AUC: {metrics['auc']:.3f} (95% CI: {metrics['ci_lower']:.3f}-{metrics['ci_upper']:.3f})")
    print(f"  >> Calibrated: F1={cal_metrics['f1']:.3f} | Sens={cal_metrics['sensitivity']:.3f} | Spec={cal_metrics['specificity']:.3f}")

    return df_cv, metrics, calibration_params


def apply_calibration(df, calibration_params, set_name=""):
    """Apply CV-derived calibration to standard/evaluation predictions."""
    scores = df['score'].values
    labels = df[LABEL_COL].values
    threshold = calibration_params['threshold']
    score_std = calibration_params['score_std']

    # Before
    pred_before = (scores > np.median(scores)).astype(int)
    print(f"  [{set_name} BEFORE] median threshold={np.median(scores):.3f}")
    print_metrics(labels, pred_before, prefix=f"[{set_name} BEFORE] ")

    # Apply CV-derived scaling
    threshold_scaled = threshold / score_std if score_std > 0 else threshold
    scores_scaled = scores / score_std if score_std > 0 else scores

    df['score_calibrated'] = scores_scaled
    df['prediction'] = (scores_scaled > threshold_scaled).astype(int)
    df['youden_threshold'] = threshold_scaled

    # After
    print(f"  [{set_name} AFTER] CV threshold={threshold:.3f}, scaled={threshold_scaled:.3f}")
    cal_metrics = print_metrics(labels, df['prediction'].values, prefix=f"[{set_name} AFTER]  ")

    auc, auc_cov = delong_roc_variance(labels, scores_scaled)
    auc_std = np.sqrt(auc_cov)
    metrics = {
        'auc': auc,
        'ci_lower': auc - 1.96 * auc_std,
        'ci_upper': auc + 1.96 * auc_std,
        **cal_metrics
    }

    print(f"  [{set_name}] AUC: {metrics['auc']:.3f} (95% CI: {metrics['ci_lower']:.3f}-{metrics['ci_upper']:.3f})")
    print(f"  [{set_name}] >> Calibrated: F1={cal_metrics['f1']:.3f} | Sens={cal_metrics['sensitivity']:.3f} | Spec={cal_metrics['specificity']:.3f}")

    return df, metrics


def calibrate_all():
    """
    Auto-discover all combinations from CV directory structure.
    For each: combine CV folds -> Youden threshold -> apply to standard & evaluation.
    """
    # Load train slides to exclude from threshold calculation
    annotations = pd.read_csv(ANNOTATIONS_PATH)
    test_slides = set(annotations.loc[annotations['dataset'] == 'test', 'slide'].astype(str))
    print(f"Excluding {len(test_slides)} test slides from threshold calculation\n")

    # Auto-discover combinations (handles variable depth before 'classification')
    cv_pattern = '**/classification/cross_validation/hypothesis_driven/*/*/seed_0'
    combinations = set()
    for path in RESULTS_DIR.glob(cv_pattern):
        parts = path.relative_to(RESULTS_DIR).parts
        class_idx = parts.index('classification')
        prefix = str(Path(*parts[:class_idx]))
        extractor = parts[class_idx + 3]
        modality = parts[class_idx + 4]
        combinations.add((prefix, extractor, modality))

    combinations = sorted(combinations)
    print(f"Found {len(combinations)} combinations\n")

    for prefix, extractor, modality in combinations:
        print(f"\n{'='*60}")
        print(f"{prefix} | {extractor} | {modality}")
        print(f"{'='*60}")

        base_path = RESULTS_DIR / prefix / 'classification'
        hd_path = Path('hypothesis_driven') / extractor / modality / 'seed_0'

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
                print(f"  Read {fold_dir.name}: {len(df_fold)} samples")

        if not cv_dfs:
            print("  No predictions found, skipping")
            continue

        df_cv = compute_score(pd.concat(cv_dfs, ignore_index=True))
        print(f"  Combined {len(cv_dfs)} folds, {len(df_cv)} total samples")
        print(f"  Score range: [{df_cv['score'].min():.3f}, {df_cv['score'].max():.3f}]")
        print(f"  Label distribution: {dict(df_cv[LABEL_COL].value_counts().sort_index())}")

        # --- Step 2: Calibrate CV (excluding test) ---
        df_cv_cal, cv_metrics, calibration_params = calibrate_cv_scores(df_cv, exclude_slides=test_slides)
        df_cv_cal.to_parquet(cv_base / 'predictions_calibrated.parquet')
        print(f"  Saved CV calibrated")

        # --- Step 3: Apply to standard ---
        std_pred = base_path / 'standard' / hd_path / 'predictions.parquet'
        if std_pred.exists():
            df_std = compute_score(pd.read_parquet(std_pred))
            df_std_cal, std_metrics = apply_calibration(df_std, calibration_params, set_name="STANDARD")
            df_std_cal.to_parquet(std_pred.parent / 'predictions_calibrated.parquet')

        # --- Step 4: Apply to evaluation ---
        eval_base = base_path / 'evaluation' / hd_path / 'eval'
        if eval_base.exists():
            for eval_dir in sorted(eval_base.iterdir()):
                eval_pred = eval_dir / 'predictions.parquet'
                if eval_pred.exists():
                    df_eval = compute_score(pd.read_parquet(eval_pred))
                    df_eval_cal, eval_metrics = apply_calibration(df_eval, calibration_params, set_name=f"EVAL-{eval_dir.name}")
                    df_eval_cal.to_parquet(eval_dir / 'predictions_calibrated.parquet')


if __name__ == '__main__':
    calibrate_all()

