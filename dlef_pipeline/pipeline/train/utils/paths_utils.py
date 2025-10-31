def build_experiment_path(config, mods, outcome_override=None):
    import os
    """
    Builds a base experiment path like:
    os_months_6/classification/standard/data_driven/foundation-noimp/rwd
    """
    task = config["task"]
    outcome = outcome_override or config["task_settings"]["outcome"]
    train_type = config["training_type"]
    data_type = config["data-type"]
    source = config["source"]
    imp = config["imp"]
    modalities = '_'.join([mod for mod, active in mods.items() if active])

    return os.path.join(
        outcome,
        task,
        train_type,
        data_type,
        f"{source}-{imp}",
        modalities
    )

def build_base_path(config, mods):
    import os
    """
    Builds a base path for the experiment based on the config and mods.
    Returns a string like:
        os_months_6/classification/standard/data_driven/foundation-noimp/rwd
    """
    # Extract relevant parts from the config
    task = config["task"]
    outcome = config["task_settings"]["outcome"]
    training_type = config["training_type"]  # should be "hyperparameter_tuning" if tuning!
    data_type = config["data-type"]
    source = config["source"]
    imp = config["imp"]
    mods_str = "_".join([k for k, v in mods.items() if v])

    print(f" -------- Building base path for task: {task}, outcome: {outcome}, training_type: {training_type}, data_type: {data_type}, source: {source}, imp: {imp}, mods: {mods_str} --------") 

    return os.path.join(
        outcome, task, training_type, data_type, f"{source}-{imp}", mods_str
    )

def build_full_path(base_path, hyperparams=None, fold=None, seed=None, final=False):
    import os
    path_parts = [base_path]

    if hyperparams:
        hparam_str = "hparam_" + "_".join(f"{k[:3]}{str(v).replace('.', '')}" for k, v in hyperparams.items())
        path_parts.append(hparam_str)

    if final:
        path_parts.append("final_model")

    if seed is not None:
        path_parts.append(f"seed_{seed}")

    if fold:
        path_parts.append(f"fold_{fold}")

    return os.path.join(*path_parts)
