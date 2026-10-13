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
# both feed mod2; dp vs dp-titan vs dp-titan-coral all feed mod3). `token=None`
# marks the legacy/default variant, whose parquet filenames predate this
# split and so carry no distinguishing token — it's selected by *excluding*
# every sibling variant's token rather than by matching one of its own.
#
# Add a tuple to a group's 'variants' list when a new source-variant is
# introduced for that modality (e.g. a future dp-<other-model>): give it a
# unique filename token (matching create_parquet.py's dp_output_suffix) and
# it's automatically excluded from the legacy/default variant's selection
# too, and automatically wired into mod_mapping below.
SOURCE_VARIANT_GROUPS = [
    {'column': 'mod2', 'variants': [('radpy', 'radpy'), ('radfm', 'fmrad')]},
    {'column': 'mod3', 'variants': [('dp', None), ('dp-titan', 'dp-titan'), ('dp-titan-coral', 'dp-titan-coral')]},
]


def _select_source_variant_parquets(paths: list, active_mods: list) -> list:
    """Filter a list of resolved parquet paths down to the ones matching the
    active source-variant mods (radpy/radfm, dp/dp-titan/dp-titan-coral, ...).

    A group is left unfiltered if none of its mods are active, so callers
    relying on the historical "use everything" fallback still get that
    behavior. If more than one mod in a group is active, that's ambiguous
    (two source variants can't both be selected for the same dataframe
    column) and raises rather than silently skipping the filter.
    """
    selected = paths
    for group in SOURCE_VARIANT_GROUPS:
        variants = group['variants']
        active_in_group = [(key, token) for key, token in variants if key in active_mods]
        if not active_in_group:
            continue
        if len(active_in_group) > 1:
            raise ValueError(
                f"Mutually-exclusive mods {[key for key, _ in active_in_group]} are all "
                f"active for column {group['column']!r}; enable exactly one."
            )
        token = active_in_group[0][1]
        if token is not None:
            selected = [p for p in selected if token in os.path.basename(p)]
        else:
            sibling_tokens = [t for _, t in variants if t is not None]
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
        # Map each modality name to its column name in the dataframe.
        # 'cb'/'genomics' are single-source; the source-variant modalities
        # (radpy/radfm, dp/dp-titan/dp-titan-coral, ...) are wired in from
        # SOURCE_VARIANT_GROUPS so a new variant only needs adding there.
        mod_mapping = {
            'cb': 'mod1',
            'genomics': 'mod4',
        }
        for group in SOURCE_VARIANT_GROUPS:
            for key, _token in group['variants']:
                if key in mods:
                    mod_mapping[key] = group['column']

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
            train_data_path = os.path.normpath(os.path.join(project_root, train_data))
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
