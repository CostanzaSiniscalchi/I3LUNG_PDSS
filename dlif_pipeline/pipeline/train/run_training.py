from .training_loop import train_val
from .prepare_dataset import prepare_dataset
from .utils.paths_utils import build_base_path, build_full_path
from .utils.config_utils import get_folds
from MIL import Project
import os
from .hyperparameters_tuning.grid_runner import run_grid_search_cv
from .hyperparameters_tuning.hyperparam_tuning import pick_best_hyperparams, pick_best_final_model
import json
from metrics.compute_scores import compute_scores
from metrics.calculate_average import compute_weighted_average

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

def run_training(config, mods):
    """
    Executes the full training pipeline for a given modality combo.

    - If training_type == "hyperparameter_tuning":
        → Runs CV for all hyperparameter combos
        → Picks the best combo
        → Retrains on full data
        → Returns 'hyperparam_dir' and 'final_model_dir'

    - Else (standard, cross_validation, or cross_validation_stratified):
        → Trains based on training_type, seeds, and folds
        → Returns 'experiment_paths'
    """

    '''
     This prefix will be used to create unique paths for each experiment.
     The prefix is based on the filters that determine the subgroup analysis.
    '''
    prefix_parts = ["results"]
    if config.get("FILTER_SQUAMOUS") is not None:
        prefix_parts.append(f"squamous_{config['FILTER_SQUAMOUS']}")
    if config.get("FILTER_CHEMO_IMMUNO") is not None:
        prefix_parts.append(f"chemoio_{config['FILTER_CHEMO_IMMUNO']}")
    if config.get("USE_COHORT2_FILTER"):
        print("in run training")
        prefix_parts.append("C2")
    else:
        prefix_parts.append("C23")

    # intracenters analysis
    if config.get("FILTER_INT"):
        prefix_parts.append(f"int")
    if config.get("FILTER_GHD"):
        prefix_parts.append(f"ghd")
    if config.get("FILTER_SZMC"):
        prefix_parts.append(f"szmc")
    if config.get("FILTER_VHIO"):
        prefix_parts.append(f"vhio")
    if config.get("FILTER_MH"):
        prefix_parts.append(f"mh")
        
    if config.get("ADENO"):
        prefix_parts.append(f"adeno")

    if config.get("FILTER_PDL1"):
        prefix_parts.append(f"pdl1_{config['FILTER_PDL1']}")
    if config.get("FILTER_ALL_MODS"):
        prefix_parts.append("all_mods")

    # if len(prefix_parts) == 1:  # Solo "results", nessun filtro attivo
    #     prefix_parts.append("main_analysis")

    path_prefix = os.path.join(*prefix_parts)

    mod_string = "_".join([k for k, v in mods.items() if v])
    bag_path_mods = f"bags_{mod_string}"
    bag_path = os.path.join(ROOT, "bags", bag_path_mods)

  
    P = Project(ROOT)

    training_type = config["training_type"]
    folds = get_folds("cross_validation", config) if training_type == "hyperparameter_tuning" else get_folds(training_type, config)
    seeds = config["seed"]

    '''
    skip training if the flag is set
    '''
    if config.get("SKIP_TRAINING", False):
        print("-- skipping training as 'SKIP_TRAINING' is set to True")
        return {"status": "dataset_prepared", "bag_path": bag_path}

    '''
     the workflow split based on the training type
    '''



    '''
    -------
    -------
               hyperparameter tuning
    -------
    -------
    '''
    if training_type == "hyperparameter_tuning":
        base_path = build_base_path(config, mods)
        hyperparam_path = os.path.join(ROOT, path_prefix, "mil", base_path, "hyperparam")
        final_model_path = os.path.join(ROOT, path_prefix, "mil", base_path, "final_model")

        final_results = []

        for seed in seeds:
            for fold in folds:    
                          
                 run_grid_search_cv(
                    P,
                    config=config,
                    base_results_path=hyperparam_path,
                    bag_path=bag_path,
                    folds=folds,
                    mods=mods,
                    fold=fold,
                    seed=seed,
                    training_type="cross_validation", # always cross validation for hyperparam tuning
                )   
    
            ''' 
            Pick best hyperparameters for the current seed  
            '''          
            best_combo = pick_best_hyperparams(
                base_results_path=hyperparam_path,
                config=config
            )

            print(f"Best hyperparameters for seed {seed}: {best_combo} ")
            final_seed_path = os.path.join(final_model_path, f"seed_{seed}") 

            train_val(
                P,
                config={**config, "hyper_combo": best_combo},
                mods=mods,
                fold="ALL",
                seed=seed,
                results_path=os.path.abspath(final_seed_path),
                bag_path=bag_path,
                folds=folds,
                training_type="standard" # always standard for final model
            )
            print("DEBUG CONFIG at run_training: ", config)
            if(config.get("task") == "survival"):
                compute_scores(final_seed_path, config["task"])
                compute_weighted_average(final_seed_path, config["task"])

            final_results.append({
                "seed": seed,
                "path": final_seed_path,
                "combo": best_combo
            })

         # Pick best final model among all seeds
        best_model = pick_best_final_model(final_model_path, config["task"])
        best_seed = int(best_model["seed"].replace("seed_", ""))

        # retrieve the combo for the best seed
        combo_map = {item["seed"]: item["combo"] for item in final_results}
        print("combo map", combo_map)
        if best_seed not in combo_map:
            raise RuntimeError(f"Best seed {best_seed} not found in final_results!")

        best_combo = combo_map[best_seed]

        # save a summary file
        summary = {
            "seed": best_seed,
            "score": best_model["score"],
            "metric": best_model["metric"],
            "combo": best_combo
        }

        with open(os.path.join(final_model_path, "best_model.json"), "w") as f:
            json.dump(summary, f, indent=2)

        return {
            "hyperparam_dir": os.path.abspath(hyperparam_path),
            "final_model_dir": os.path.abspath(final_model_path),
            "best_models_per_seed": {best_seed: best_model["path"]},
            "best_combos_per_seed": {best_seed: best_combo},
        }
    

        '''
        -------
            standard / cross validation / evaluation
        -------
        -------
        '''
    else:
        experiment_paths = []
        base_path = os.path.join(ROOT, path_prefix, "mil", build_base_path(config, mods))
        print("PREFIX PARTS:", prefix_parts)

        for seed in seeds:
            # Handle the base seed path
            base_seed_path = os.path.join(base_path, f"seed_{seed}")
            experiment_paths.append(base_seed_path)

            for fold in folds:
                # Build the full training path
                full_path = build_full_path(
                    base_path=base_path,
                    fold=fold if fold != "ALL" else None,
                    seed=seed,
                )

                train_val(
                    P,
                    config,
                    mods,
                    fold,
                    seed,
                    results_path=full_path,
                    bag_path=bag_path,
                    folds=folds,
                    training_type=training_type
                )  

            if(config.get("task") == "survival"):
                compute_scores(full_path, config["task"])
            if config.get("training_type") == "cross_validation":
                compute_weighted_average(base_seed_path, config["task"])

                print(f"Training completed for seed {seed}, fold {fold} at {full_path}")

        return {"experiment_paths": experiment_paths}