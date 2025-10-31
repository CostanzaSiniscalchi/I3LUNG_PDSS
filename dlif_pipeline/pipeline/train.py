import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import sys
import copy

# Add parent directory to path BEFORE importing local modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from train.prepare_dataset import prepare_dataset
import argparse
import yaml
from train.run_training import run_training


def pipeline(config, mods):
    print(f"\n=== Training: outcome={config['task_settings']['outcome']}, mods={mods}, subanalysis={config.get('current_filter', 'none')} ===")
    experiments = run_training(config, mods)
    print("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base_dir", required=True)
    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    original_config = copy.deepcopy(config)

    try:
        # Dataset preparation if requested
        if config.get("prepare_dataset", False):
            for mods in config.get('mods', []):

                train_data = config.get("train_df")

                mod_string = "_".join([k for k, v in mods.items() if v])
                bag_path = f"bags_{mod_string}"
                bag_path_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), '../bags/', bag_path))
                prepare_dataset(train_data, config.get("annotation_file"), mods, bag_path_abs)

            print("dataset prepared")

        # Main processing: iterate over outcomes
        for outcome in original_config['task_settings'].get('outcomes', []):
            print(f"\n --- Processing outcome: {outcome}")
            # Reset full config for each outcome
            config = copy.deepcopy(original_config)
            config['task_settings']['outcome'] = outcome

            for mods in config.get('mods', []):
                print(f"\n --- Mods: {mods}")
                pipeline(config, mods)

    except Exception:
        raise


