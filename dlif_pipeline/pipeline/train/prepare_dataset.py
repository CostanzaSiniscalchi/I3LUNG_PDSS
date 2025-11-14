import pandas as pd
import shutil
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, project_root)

# Import direct depuis le projet
from MIL import Project
from MIL.util import prepare_multimodal_mixed_bags

def prepare_dataset(train_data: str, annotation_file: str, mods: dict, bag_path: str):
    """
    Prepares the dataset by:
    - Dropping unused modality columns
    - Setting up the annotation file path
    - Creating the Slideflow project and bags
    """

    # Identify which modalities are *not* active
    excluded_mods = [key for key in mods.keys() if not mods[key]]
    active_mods = [key for key in mods.keys() if mods[key]]

    print(f"\n[INFO] Preparing dataset with modalities: {active_mods}")

    if not excluded_mods:
        print("[INFO] Using all modalities.")
        if isinstance(train_data, list):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
            train_data_resolved = [os.path.normpath(os.path.join(project_root, td)) for td in train_data]
            dfs = [pd.read_parquet(td) for td in train_data_resolved]
            df = pd.concat(dfs, ignore_index=True)
            df.to_parquet('df.parquet', index=False)
        else:
            shutil.copyfile(train_data, 'df.parquet')
    else:
        # Map each modality name to its column name in the dataframe
        mod_mapping = {
            'rwd': 'mod1', 
            'dp': 'mod3',
            'genomics': 'mod4',
        }

        # Add the correct radiomics key
        if 'radpy' in mods:
            mod_mapping['radpy'] = 'mod2'
        if 'radfm' in mods:
            mod_mapping['radfm'] = 'mod2'  # same column name, different source

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        if isinstance(train_data, list):
            train_data_resolved = [os.path.normpath(os.path.join(project_root, td)) for td in train_data]
            dfs = [pd.read_parquet(td) for td in train_data_resolved]
            df = pd.concat(dfs, ignore_index=True)
        else:
            train_data_path = os.path.join(project_root, train_data.lstrip('../'))
            df = pd.read_parquet(train_data_path)
        
        # Drop columns of unused modalities (if they exist in the mapping)
        df = df.drop(columns=[mod_mapping[mod] for mod in excluded_mods if mod in mod_mapping], errors='ignore')

        # Ensure all missing values are properly handled
        df = df.applymap(lambda x: None if isinstance(x, float) and pd.isna(x) else x)
        df.dropna()  # This doesn't modify df in place

        df.to_parquet('df.parquet', index=False)

    # Create project + multimodal bags
    P = Project(os.getcwd(), create=True)
    prepare_multimodal_mixed_bags('df.parquet', bag_path)
