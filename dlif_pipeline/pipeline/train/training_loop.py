
def train_val(P, config, mods, fold, seed, results_path, bag_path, folds, training_type):
    """
    Trains a MIL model for a specific fold, modality set, and seed.

    Supports:
    - Standard training (uses config["hyperparameters"])
    - Hyperparameter tuning (uses config["hyper_combo"])
    - Evaluation only (skips training, only evaluates on ext_val)

    Args:
        P: Slideflow Project instance
        config: Experiment config dict
        mods: List of modalities used in the training
        fold: Fold name or ID
        seed: Seed for reproducibility
        results_path: Where to save the training outputs
        bag_path: Path to MIL bag .npz files
        folds: List of folds for cross-validation
        training_type: Type of training ("standard", "cross_validation", "evaluation", etc.)
    """
    
    from MIL.mil import mil_config, eval_mil
    from .utils.config_utils import get_task_settings
    from .utils.dataset_utils import get_datasets, get_eval_dataset
    import torch, random, numpy as np
    import os

    # 1. Load task settings
    task_settings = get_task_settings(config["task"], config.get("task_settings"))
    outcome = task_settings["outcome"]
    save_monitor = task_settings.get("save_monitor", "valid_loss")  # default se non specificato
    events = task_settings.get("events")
    
    # 2. Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 3. Handle evaluation-only mode
    if training_type == "evaluation":
        print(f" Evaluation mode: skipping training, only evaluating on ext_val")
        
        # Select hyperparameters for config building
        combo = config.get("hyper_combo", config.get("hyperparameters_default", {}))
        epochs = combo.get("epochs", 25)
        batch_size = combo.get("batch_size", 64)
        bag_size = combo.get("bag_size", 32)
        recon_weight = combo.get("reconstruction_weight", 0.1)
        n_layers = combo.get("n_layers", 1)

        # Build MIL config for evaluation
        config_mil = mil_config(
            'mb_attention_mil',
            loss=task_settings["loss"],
            epochs=epochs,
            batch_size=batch_size,
            bag_size=bag_size,
            save_monitor=save_monitor,
            reconstruction_weight=recon_weight,
            model_kwargs={"n_layers": n_layers}
        )
        config_mil.mixed_bags = True

        # Evaluation on external validation set
        print(f" Evaluating on external validation set (ext_val)")
        standard_results_path = results_path.replace("/evaluation/", "/standard/")
        best_checkpoint = os.path.join(standard_results_path)
        
        if not os.path.exists(best_checkpoint):
            raise FileNotFoundError(f"best checkpoint not found at {best_checkpoint}")

        print(f"best model checkpoint found: {best_checkpoint}")

        # Filter for external validation set (with subanalysis filters)
        test_dataset = get_eval_dataset(P, config, outcome)
        outdir = os.path.join(results_path, "eval")
        os.makedirs(outdir, exist_ok=True)

        print(f"evaluation outputs will be saved to: {outdir}")

        # build evaluation kwargs for Slideflow
        eval_kwargs = {
            "weights": best_checkpoint,
            "config": config_mil,
            "outcomes": outcome,
            "dataset": test_dataset,
            "bags": bag_path,
            "outdir": os.path.abspath(outdir),
        }

        # Optional event-based evaluation (for survival)
        if events:
            eval_kwargs["events"] = events

        # run evaluation
        eval_mil(**eval_kwargs)
        print(f"evaluation done on external validation set!\n")
        return

    

    # 4. Prepare dataset for fold
    train_dataset, val_dataset = get_datasets(
        P, training_type, config, fold, folds, outcome
    )

    # Check if validation set has both classes
    val_labels = val_dataset.labels(outcome)
    print(f"val_labels type: {type(val_labels)}")
    print(f"val_labels length: {len(val_labels)}")
    print(f"val_labels first 5: {val_labels[:5]}")
    print(f"val_labels element types: {[type(x) for x in val_labels[:5]]}")
    # Estrai solo i valori stringa, ignorando dict
    val_labels_flat = []
    for x in val_labels:
        if isinstance(x, (list, np.ndarray)):
            val_labels_flat.extend([str(v) for v in x if not isinstance(v, dict)])
        elif not isinstance(x, dict):
            val_labels_flat.append(str(x))

    unique_val_labels = np.unique(val_labels_flat)
    if len(unique_val_labels) < 2:
        print(f"⚠️  Skipping fold {fold} - only one class ({unique_val_labels}) in validation set")
        return


    # 5. Select hyperparameters
    combo = config.get("hyper_combo", config.get("hyperparameters_default", {}))
    epochs = combo.get("epochs", 25)
    batch_size = combo.get("batch_size", 64)
    bag_size = combo.get("bag_size", 32)
    recon_weight = combo.get("reconstruction_weight", 0.1)
    n_layers = combo.get("n_layers", 1)
    lr = combo.get("lr", None)
    if lr is not None:
        lr = float(lr)
    fit_one_cycle = combo.get("fit_one_cycle", True)

    # 6. Build MIL config
    config_mil = mil_config(
        'mb_attention_mil',
        loss=task_settings["loss"],
        epochs=epochs,
        batch_size=batch_size,
        bag_size=bag_size,
        save_monitor=save_monitor,
        reconstruction_weight=recon_weight,
        lr=lr,
        fit_one_cycle=fit_one_cycle,
        model_kwargs={"n_layers": n_layers}
    )
    config_mil.mixed_bags = True

    # 7. Run training
    os.makedirs(results_path, exist_ok=True)
    train_kwargs = {
        "config": config_mil,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "outcomes": outcome,
        "bags": bag_path,
        "exp_label": os.path.abspath(results_path),
    }
    if events:
        train_kwargs["events"] = events

    P.train_mil(**train_kwargs)
    
    # 8. Evaluation on test set
    if config.get("use_early_stopping", True):
        print(f"\nearly stopping evaluation enabled!")

        best_checkpoint = os.path.join(results_path)
        if not os.path.exists(best_checkpoint):
            raise FileNotFoundError(f"best checkpoint not found at {best_checkpoint}")

        print(f"best model checkpoint found: {best_checkpoint}")

        # build correct filter based on training type
        if training_type == "cross_validation":
            if config.get("FILTER_INT"):
                test_filter = {f"INT_ONLY_FOLDS": [fold]}
                if outcome != "OS_MONTHS":
                    test_filter.update({f"{outcome}": ["0.0", "1.0"]})
                else:
                    pass
                    # new column for OS_MONTHS not implemented yet
            else:
                test_filter = {f"fold_{outcome}": [fold]}
            print(f"cross-validation: test set is fold {fold}")
        elif training_type == "standard":
            test_filter = {
                f"dataset_{outcome}": "test",
                f"early_stopping_{outcome}": "no"
            }
            print(f"standard training: test set = dataset=test AND early_stopping=no")
        else:
            raise ValueError(f"unknown training_type: {training_type}")

        # Evaluate model
        test_dataset = P.dataset(tile_px=256, tile_um=129, filters=test_filter)
        outdir = os.path.join(results_path, "eval")
        os.makedirs(outdir, exist_ok=True)

        print(f"evaluation outputs will be saved to: {outdir}")

        # build evaluation kwargs for Slideflow
        eval_kwargs = {
            "weights": best_checkpoint,
            "config": config_mil,
            "outcomes": outcome,
            "dataset": test_dataset,
            "bags": bag_path,
            "outdir": os.path.abspath(outdir),
        }

        # Optional event-based evaluation (for survival)
        if events:
            eval_kwargs["events"] = events

        # run evaluation
        eval_mil(**eval_kwargs)
        print(f"evaluation done on {'fold ' + str(fold) if training_type == 'cross_validation' else 'final test set'}!\n")
