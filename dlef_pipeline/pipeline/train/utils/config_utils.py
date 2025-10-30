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

    print(f"[DEBUG] Called get_folds() with training_type='{training_type}' and config={config!r}")

    # 1) Look for a 'folds' mapping in the provided config
    if config is not None:
        print("[DEBUG] Config is not None, attempting to retrieve 'folds' section...")
        folds_from_config = config.get("folds", {})
        print(f"[DEBUG] folds_from_config = {folds_from_config!r}")

        if training_type in folds_from_config:
            print(f"[DEBUG] Found '{training_type}' in folds_from_config. Returning: {folds_from_config[training_type]!r}")
            return folds_from_config[training_type]
        else:
            print(f"[DEBUG] '{training_type}' not in folds_from_config; will fall back to defaults.")

    else:
        print("[DEBUG] Config is None, skipping config-based folds.")

    # Default mapping
    default_folds = {
        "cross_validation_stratified": [
            'GHD_2', 'INT_2', 'MH_2', 'SZMC_2', 'VHIO_2',
            'GHD_3', 'INT_3', 'MH_3', 'SZMC_3', 'VHIO_3'
        ],
        "cross_validation": [
            'GHD', 'INT', 'MH', 'SZMC', 'VHIO'
        ],
        "standard": ['ALL']
    }
    print(f"[DEBUG] default_folds keys = {list(default_folds.keys())!r}")

    if training_type not in default_folds:
        print(f"[ERROR] Unknown training_type: '{training_type}'. Raising ValueError.")
        raise ValueError(f"Unknown training_type: {training_type}")

    print(f"[DEBUG] Returning default_folds['{training_type}'] = {default_folds[training_type]!r}")
    return default_folds[training_type]