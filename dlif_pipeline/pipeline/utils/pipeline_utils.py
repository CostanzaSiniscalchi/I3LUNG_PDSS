import os

def list_experiments(mods_path):
    """
    List valid experiment folders inside a modality path.
    Skips hidden files and non-directories.
    """
    return [
        name for name in os.listdir(mods_path)
        if not name.startswith(".") and os.path.isdir(os.path.join(mods_path, name))
    ]

def build_path(base_path, config, mods):
    """
    Build the path for the experiment based on the configuration and modalities.
    """
    source_imp = f"{config['source']}-{config['imp']}"
    modality_list = "_".join(mods)
    mods_path = os.path.join(base_path, config["task"], config["data-type"], config["training_type"], source_imp, modality_list)
    return mods_path
