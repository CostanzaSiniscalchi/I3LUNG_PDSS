from itertools import product
from ..training_loop import train_val  # or your wrapped trainer
from copy import deepcopy # to avoid modify nested structures
from ..utils.paths_utils import build_full_path

def stringify_hyperparams(combo):
    return "_".join([f"{k[:3]}{str(v).replace('.', '')}" for k, v in combo.items()])

def run_grid_search_cv(P, config, base_results_path, bag_path, folds, mods, fold, seed, training_type):
    """
    Runs cross-validation training over all hyperparameter combinations for a single fold and seed.
    Saves results separately per combo.
    """

    import os
    task_settings = config["task_settings"]
    hyper_grid = config["hyperparameters"]

    param_names = list(hyper_grid.keys())
    param_values = list(hyper_grid.values())
    combos = list(product(*param_values))


    for i, values in enumerate(combos):
        combo = dict(zip(param_names, values))
        print(f"🔧 Training combo {i+1}/{len(combos)}: {combo}")

        combo_str = stringify_hyperparams(combo)
        results_path = build_full_path(
            base_path=base_results_path,
            hyperparams=combo,
            fold=fold,
            seed=seed
        )

        # Update config temporarily for training
        config_copy = deepcopy(config)
        config_copy["hyper_combo"] = combo
        config_copy["results_path"] = results_path

        
        # Run training for this seed/fold
        train_val(P, config_copy, mods, fold, seed, results_path, bag_path, folds, training_type)




