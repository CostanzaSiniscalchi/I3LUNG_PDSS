import pandas as pd
import numpy as np
from scipy import stats
from sksurv.metrics import concordance_index_censored



class Statistics:

    def elbow_selection(self, features: pd.DataFrame, min_coefs: int, max_coefs: int):
        """
        Select features using the elbow method.
        
        Parameters
        ----------
        features : pd.DataFrame
            Dataframe containing the feature importance values. It must have the column 'COEFFICIENT'.
        min_coefs : int
            Minimum number of features to select.
        max_coefs : int
            Maximum number of features to select.
        plot : bool, optional
            Plot the elbow method analysis, by default False.
            
        Returns
        -------
        pd.DataFrame
            Dataframe containing the selected features.
        """
        total_features = features.shape[0]
        target_features = (min_coefs + max_coefs) // 2
        uncertainty = (max_coefs - min_coefs) // 2
        
        if total_features <= min_coefs:
            return features
        
        features['ABS_COEFFICIENT'] = np.abs(features['COEFFICIENT'])
        features = features.sort_values(by='ABS_COEFFICIENT', ascending=False)
        coefs = features['ABS_COEFFICIENT'].values
        
        
        if coefs.max() > 0:
            y = coefs / coefs.max()
        else:
            y = coefs
        
        dy_dt = np.gradient(y)
        d2y_dt2 = np.gradient(dy_dt)
        curvature = np.abs(d2y_dt2) / (1 + dy_dt**2)**(3/2)
        
        elbow_idx = np.argmax(curvature[1:-1]) + 1
        
        distance_from_target = abs(elbow_idx - target_features)
        
        if distance_from_target > uncertainty:
            min_range = max(1, min_coefs)
            max_range = min(total_features - 1, max_coefs)
            
            local_range = range(min_range, max_range)
            if len(local_range) > 0:
                local_idx = min_range + np.argmax(curvature[min_range:max_range])
                
                global_curvature = curvature[elbow_idx]
                local_curvature = curvature[local_idx]
                
                if local_curvature > global_curvature * 0.5:
                    elbow_idx = local_idx
                else:
                    adj_factor = 0.7
                    elbow_idx = int(elbow_idx + adj_factor * (target_features - elbow_idx))
        
        if elbow_idx < len(coefs):
            threshold = coefs[elbow_idx]
        else:
            threshold = coefs[-1]
        
        selected_features = features[features['ABS_COEFFICIENT'] >= threshold]
        
        min_features = max(1, min_coefs)
        max_features = min(total_features, max_coefs)
        
        if len(selected_features) < min_features:
            selected_features = features.head(min_features)
        elif len(selected_features) > max_features:
            selected_features = features.head(max_features)
        
        return selected_features
    

    def _compute_ground_truth_statistics(self, ground_truth):
        ground_truth = np.array(ground_truth)
        if not np.array_equal(np.unique(ground_truth), [0, 1]):
            return None, None
        order = (-ground_truth).argsort()
        label_1_count = int(ground_truth.sum())

        return order, label_1_count


    def _delong_roc_variance(self, ground_truth, predictions):
        """
        Computes ROC AUC variance for a single set of predictions
        Args:
        ground_truth: np.array of 0 and 1
        predictions: np.array of floats of the probability of being class 1
        """
        order, label_1_count = self._compute_ground_truth_statistics(ground_truth)
        if order is None or label_1_count is None:
            return np.nan, np.nan
        
        predictions_sorted_transposed = predictions[np.newaxis, order]
        aucs, delongcov = self._fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
        assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"

        return aucs[0], delongcov



    def auc_roc_ci(self, y_true, y_pred, alpha):
        """
        Return AUC and CL at alpha for vectors y_true, y_pred
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        auc, auc_cov = self._delong_roc_variance(
            y_true,
            y_pred)

        if pd.isna(auc) or pd.isna(auc_cov):
            return np.nan, np.nan
        
        auc_std = np.sqrt(auc_cov)
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
        
        ci = stats.norm.ppf(
            lower_upper_q,
            loc=auc,
            scale=auc_std)
        
        ci[ci > 1] = 1
        
        return auc, ci
    

    def get_auc_ci(self, y_true, y_pred, alpha=0.95):
        auc, ci = self.auc_roc_ci(y_true, y_pred, alpha)

        return f'{auc:.3f} ± {((ci[1]-ci[0])/2):.3f}'
    

    def _compute_midrank(self, x):
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


    def _fastDeLong_no_weights(self, predictions_sorted_transposed, label_1_count):
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
            tx[r, :] = self._compute_midrank(positive_examples[r, :])
            ty[r, :] = self._compute_midrank(negative_examples[r, :])
            tz[r, :] = self._compute_midrank(predictions_sorted_transposed[r, :])
        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n

        return aucs, delongcov
    

    def compute_c_index_and_ci(self, event, time, risk_score, n_boot=1000, alpha=0.05, seed=42):
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
        c_indexes = []
        event = np.asarray(event, dtype=bool)
        time = np.asarray(time, dtype=float)
        risk_score = np.asarray(risk_score, dtype=float)
        valid_mask = ~np.isnan(risk_score)
        event = event[valid_mask]
        time = time[valid_mask]
        risk_score = risk_score[valid_mask]
        
        n = len(time)
        event = np.asarray(event, dtype=bool)
        time = np.asarray(time, dtype=float)
        risk_score = np.asarray(risk_score, dtype=float)
        valid_mask = ~np.isnan(risk_score)
        event = event[valid_mask]
        time = time[valid_mask]
        risk_score = risk_score[valid_mask]
        
        n = len(time)
        
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