import pandas as pd
import numpy as np



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