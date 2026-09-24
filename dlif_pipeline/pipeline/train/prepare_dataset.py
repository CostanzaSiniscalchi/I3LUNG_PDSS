import pandas as pd
import shutil
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, project_root)

# Import direct depuis le projet
from MIL import Project
from MIL.util import prepare_multimodal_mixed_bags

# Groups of mutually-exclusive `mods` flags that each select a different
# source variant feeding the *same* dataframe column (e.g. radpy vs radfm
# both feed mod2; dp vs dp-titan both feed mod3). `token=None` marks the
# legacy/default variant, whose parquet filenames predate this split and so
# carry no distinguishing token — it's selected by *excluding* every sibling
# variant's token rather than by matching one of its own.
#
# Add a tuple here when a new source-variant is introduced for a modality
# (e.g. a future dp-<other-model>): give it a unique filename token and it's
# automatically excluded from the legacy/default variant's selection too.
SOURCE_VARIANT_GROUPS = [
    [('radpy', 'radpy'), ('radfm', 'fmrad')],
    [('dp', None), ('dp-titan', 'dp-titan')],
]


def _select_source_variant_parquets(paths: list, active_mods: list) -> list:
    """Filter a list of resolved parquet paths down to the ones matching the
    active source-variant mods (radpy/radfm, dp/dp-titan, ...).

    A group is left unfiltered if zero or more than one of its mods are
    active (ambiguous), so callers relying on the historical "use everything"
    fallback still get that behavior.
    """
    selected = paths
    for group in SOURCE_VARIANT_GROUPS:
        active_in_group = [token for key, token in group if key in active_mods]
        if len(active_in_group) != 1:
            continue
        token = active_in_group[0]
        if token is not None:
            selected = [p for p in selected if token in os.path.basename(p)]
        else:
            sibling_tokens = [t for _, t in group if t is not None]
            selected = [p for p in selected if not any(t in os.path.basename(p) for t in sibling_tokens)]
    return selected


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
            selected = _select_source_variant_parquets(train_data_resolved, active_mods)
            dfs = [pd.read_parquet(p) for p in (selected or train_data_resolved)]
            df = pd.concat(dfs, ignore_index=True)
            df.to_parquet('df.parquet', index=False)
        else:
            shutil.copyfile(train_data, 'df.parquet')
    else:
        # Map each modality name to its column name in the dataframe
        mod_mapping = {
            'cb': 'mod1',
            'dp': 'mod3',
            'genomics': 'mod4',
        }

        # Add the correct radiomics key
        if 'radpy' in mods:
            mod_mapping['radpy'] = 'mod2'
        if 'radfm' in mods:
            mod_mapping['radfm'] = 'mod2'  # same column name, different source
        if 'dp-titan' in mods:
            mod_mapping['dp-titan'] = 'mod3'  # same column name as dp, different source

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        if isinstance(train_data, list):
            train_data_resolved = [os.path.normpath(os.path.join(project_root, td)) for td in train_data]
            # When multiple source variants of the same modality are listed
            # (radpy/radfm parquets with different mod2 dims, or dp/dp-titan
            # parquets with different mod3 dims), each subject appears once
            # per parquet. Select only the parquet(s) matching the active
            # source variant so that iloc[0] in prepare_multimodal_mixed_bags
            # gets the right features.
            selected = _select_source_variant_parquets(train_data_resolved, active_mods)
            dfs = [pd.read_parquet(p) for p in (selected or train_data_resolved)]
            df = pd.concat(dfs, ignore_index=True)
        else:
            train_data_path = os.path.join(project_root, train_data.lstrip('../'))
            df = pd.read_parquet(train_data_path)
        
        # Drop columns of unused modalities. Compared by column, not just by
        # flag name: dp and dp-titan (like radpy and radfm) share a column,
        # so an excluded sibling (e.g. dp: false) must not drop a column that
        # an active mod (e.g. dp-titan: true) still needs.
        active_cols = {mod_mapping[mod] for mod in active_mods if mod in mod_mapping}
        cols_to_drop = [mod_mapping[mod] for mod in excluded_mods if mod in mod_mapping and mod_mapping[mod] not in active_cols]
        df = df.drop(columns=cols_to_drop, errors='ignore')

        # Ensure all missing values are properly handled
        df = df.map(lambda x: None if isinstance(x, float) and pd.isna(x) else x)
        df.dropna()  # This doesn't modify df in place

        df.to_parquet('df.parquet', index=False)

    # Create project + multimodal bags
    P = Project(os.getcwd(), create=True)
    prepare_multimodal_mixed_bags('df.parquet', bag_path)
