
def pick_best_hyperparams(base_results_path, config):
    """
    for each hyperparameter combination, concatenate predictions from all seeds
    and compute a single AUC/C-INDEX on the entire combined dataset.
    """
    import os
    import pandas as pd
    import numpy as np
    import re
    from metrics.delong_n import auc_roc_ci, find_latest_mb_attention_dir
    from metrics.survival_cindex_ci import compute_c_index_and_ci

    print(f"\n Searching best hyperparameters in: {base_results_path}")
    print(f"[DEBUG] Task: {config['task']}")

    task = config["task"]
    best_score = -1
    best_combo = None

    # STEP 1: iterates over each hyperparameter combination
    for combo_folder in os.listdir(base_results_path):
        combo_path = os.path.join(base_results_path, combo_folder)
        print(f"\n[DEBUG] Checking combo folder: {combo_folder}")
        
        if not combo_folder.startswith("hparam_") or not os.path.isdir(combo_path):
            print(f"[DEBUG] Skipping {combo_folder}")
            continue

        # STEP 2: gather all predictions from all seeds for this combo
        all_predictions = []
        
        for seed_folder in os.listdir(combo_path):
            seed_path = os.path.join(combo_path, seed_folder)
            print(f"[DEBUG]   Processing seed: {seed_folder}")
            
            if not os.path.isdir(seed_path):
                continue

            try:

                print(f"[DEBUG]     Cross-validation - reading from folds")
                folders = [f.name for f in os.scandir(seed_path) if f.is_dir()]
                
                for folder in folders:
                    fold_eval_path = os.path.join(seed_path, folder, 'eval')
                    if not os.path.isdir(fold_eval_path):
                        continue
                        
                    latest_mb_dir = find_latest_mb_attention_dir(fold_eval_path)
                    pred_path = os.path.join(fold_eval_path, latest_mb_dir, 'predictions.parquet')
                    
                    if os.path.exists(pred_path):
                        pred_df = pd.read_parquet(pred_path)
                        all_predictions.append(pred_df)
                        print(f"[DEBUG]     Loaded {len(pred_df)} predictions from {folder}")
            
            except Exception as e:
                print(f" Error processing {seed_path}: {e}")
                continue

        # STEP 4: If no predictions, skip this combo
        if not all_predictions:
            print(f" No predictions found for {combo_folder}")
            continue
        
        # STEP 5: concatenate all predictions
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        print(f"[DEBUG] Total predictions for {combo_folder}: {len(combined_predictions)}")
        
        # STEP 6: calculate AUC or C-INDEX on all predictions together
        try:
            if task == "classification":
                # Softmax on logits to get probabilities
                combined_predictions['pred'] = np.exp(combined_predictions['y_pred1']) / (
                    np.exp(combined_predictions['y_pred0']) + np.exp(combined_predictions['y_pred1'])
                )
                auc, ci = auc_roc_ci(
                    combined_predictions['y_true'], 
                    combined_predictions['pred'], 
                    0.95
                )
                score = auc
                print(f"📊 {combo_folder} → AUC: {score:.4f} (CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
                
            else:  # survival
                time = combined_predictions['y_true0'].values.astype(float)
                event = combined_predictions['y_true1'].values.astype(bool)
                risk_score = -combined_predictions['y_pred0'].values.astype(float)
                
                valid_mask = ~(np.isnan(event) | np.isnan(time) | np.isnan(risk_score))
                print(f"[DEBUG] Valid samples: {valid_mask.sum()}/{len(valid_mask)}")
                
                result = compute_c_index_and_ci(
                    event[valid_mask], 
                    time[valid_mask], 
                    risk_score[valid_mask]
                )
                score = result["c_index"]
                ci = result["ci"]
                print(f" {combo_folder} → C-INDEX: {score:.4f} (CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
            
            # STEP 7: update best combo if needed
            if score > best_score:
                best_score = score
                best_combo = combo_folder
                print(f"[DEBUG]  New best combo!")
                
        except Exception as e:
            print(f" Error computing metric for {combo_folder}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # STEP 8: Verifica che abbiamo trovato un combo valido
    if not best_combo:
        raise RuntimeError(" No valid hyperparameter combo found.")

    metric_name = "AUC" if task == "classification" else "C-INDEX"
    print(f"\n Best combo: {best_combo} with {metric_name} = {best_score:.4f}")

    # STEP 9: extract hyperparameter values from the combo string
    hyperparams = {}
    if "bat" in best_combo:
        hyperparams["batch_size"] = int(re.search(r"bat(\d+)", best_combo).group(1))
    if "rec" in best_combo:
        val = re.search(r"rec(\d+)", best_combo).group(1)
        hyperparams["reconstruction_weight"] = float(f"{val[0]}.{val[1:]}")
    if "n_l" in best_combo:
        hyperparams["n_layers"] = int(re.search(r"n_l(\d+)", best_combo).group(1))
    
    print(f"[DEBUG] Extracted hyperparams: {hyperparams}")
    return hyperparams


def pick_best_final_model(final_model_path, task):
    """
    calculate AUC/C-INDEX on all predictions from each seed_X folder
    and pick the best seed.
    """
    import os
    import pandas as pd
    import numpy as np
    from metrics.delong_n import auc_roc_ci, find_latest_mb_attention_dir
    from metrics.survival_cindex_ci import compute_c_index_and_ci

    best_score = -1
    best_seed = None
    best_path = None

    print(f"\n Searching best final model in: {final_model_path}")
    print(f"[DEBUG] Task: {task}")

    # STEP 1: iterates over each seed_X folder
    for folder in os.listdir(final_model_path):
        print(f"\n[DEBUG] Checking folder: {folder}")
        
        if not folder.startswith("seed_"):
            print(f"[DEBUG] Skipping {folder}")
            continue

        seed_root = os.path.join(final_model_path, folder)

        try:
            # STEP 2: gather all predictions from all folds (or direct eval/)
            all_predictions = []
            eval_dir_for_return = None  # to return the eval dir of the best seed

            # STEP 3: Check if it's cross-validation or standard
            eval_path = os.path.join(seed_root, "eval")

            
            if os.path.isdir(eval_path):
                print(f"[DEBUG]   Standard training - reading from eval/")
                latest_mb_dir = find_latest_mb_attention_dir(eval_path)
                pred_path = os.path.join(eval_path, latest_mb_dir, 'predictions.parquet')
                eval_dir_for_return = os.path.join(eval_path, latest_mb_dir)
                
                if os.path.exists(pred_path):
                    pred_df = pd.read_parquet(pred_path)
                    all_predictions.append(pred_df)
                    print(f"[DEBUG]   Loaded {len(pred_df)} predictions")
                else:
                    print(f" No predictions in {pred_path}")
                    continue
        

            # STEP 4: If there are no predictions, skip this seed
            if not all_predictions:
                print(f" No predictions found for {folder}")
                continue
            
            # STEP 5: concatenate all predictions
            combined_predictions = pd.concat(all_predictions, ignore_index=True)
            print(f"[DEBUG] Total predictions for {folder}: {len(combined_predictions)}")
            
            # STEP 6: calculate AUC or C-INDEX on all predictions together
            if task == "classification":
                combined_predictions['pred'] = np.exp(combined_predictions['y_pred1']) / (
                    np.exp(combined_predictions['y_pred0']) + np.exp(combined_predictions['y_pred1'])
                )
                auc, ci = auc_roc_ci(
                    combined_predictions['y_true'], 
                    combined_predictions['pred'], 
                    0.95
                )
                score = auc
                metric_name = "AUC"
                print(f"{folder} → AUC: {score:.4f} (CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
                
            else:  # survival
                time = combined_predictions['y_true0'].values.astype(float)
                event = combined_predictions['y_true1'].values.astype(bool)
                risk_score = -combined_predictions['y_pred0'].values.astype(float)
                
                valid_mask = ~(np.isnan(event) | np.isnan(time) | np.isnan(risk_score))
                print(f"[DEBUG] Valid samples: {valid_mask.sum()}/{len(valid_mask)}")
                
                result = compute_c_index_and_ci(
                    event[valid_mask], 
                    time[valid_mask], 
                    risk_score[valid_mask]
                )
                score = result["c_index"]
                ci = result["ci"]
                metric_name = "C-INDEX"
                print(f"{folder} → C-INDEX: {score:.4f} (CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
            
            # STEP 7: update best seed if needed
            if score > best_score:
                best_score = score
                best_seed = folder
                best_path = eval_dir_for_return
                print(f"[DEBUG] New best seed!")
                
        except Exception as e:
            print(f"Failed to process {seed_root}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # STEP 8: verify we found a valid seed
    if best_seed is None:
        raise RuntimeError("No valid final model found!")

    print(f"\n Best final model: {best_seed} with {metric_name} = {best_score:.4f}")
    print(f"[DEBUG] Best path: {best_path}")
    
    return {
        "seed": best_seed,
        "score": best_score,
        "metric": metric_name,
        "path": best_path
    }