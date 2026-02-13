def get_task_settings(task, config_overrides=None):
    """
    Returns default settings for a given task ('survival' or 'classification').
    Optionally applies overrides from the config if provided.
    """

    defaults = {
        "survival": {
            "loss": "mm_survival_loss",
            "save_monitor": "c_index",
            "outcome": "OS_MONTHS",
            "events": "DEATH_EVENT_OC"
        },
        "classification": {
            "loss": "mm_loss",
            "outcome": "DCR",
        }
    }

    if task not in defaults:
        raise ValueError(f"Unknown task: {task}")
    
    settings = defaults[task].copy()
    
    if config_overrides:
        settings.update(config_overrides)
    
    return settings

def get_folds(training_type, config=None):
    """
    Returns the correct folds list for the given training_type.
    If a 'folds' section exists in config and contains training_type, it overrides the defaults.
    """

    # 1) Look for a 'folds' mapping in the provided config
    if config is not None:
        if config.get("FILTER_INT"):
            return ["0", "1", "2", "3", "4"]
        folds_from_config = config.get("folds", {})

        if training_type in folds_from_config:
            print(f"Training type '{training_type}' in folds_from_config. Returning: {folds_from_config[training_type]!r}")
            return folds_from_config[training_type]
        else:
            print(f"Training type '{training_type}' Will fall back to defaults.")

    # Default mapping
    default_folds = {
        "cross_validation_stratified": [
            'GHD_2', 'INT_2', 'MH_2', 'SZMC_2', 'VHIO_2',
            'GHD_3', 'INT_3', 'MH_3', 'SZMC_3', 'VHIO_3'
        ],
        "cross_validation": [
            'GHD', 'INT', 'MH', 'SZMC', 'VHIO'
        ],
        "standard": ['ALL'],
        "evaluation": ['ALL']  # Evaluation mode doesn't actually use folds, but needs to be defined
    }

    if training_type not in default_folds:
        print(f"[ERROR] Unknown training_type: '{training_type}'. Raising ValueError.")
        raise ValueError(f"Unknown training_type: {training_type}")
    return default_folds[training_type]