import argparse
import sys
from pathlib import Path

import pandas as pd

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent.parent / 'data' / 'prospective'

sys.path.insert(0, str(SCRIPT_DIR))
from create_parquet import RAD_OUTPUT_STEMS, dp_output_suffix  # noqa: E402


def _load_optional_modality(data_dir: Path, modality: str, cb_subjects: set, label: str) -> pd.DataFrame:
    """Load '<data_dir>/<modality>_processed.csv' if it exists, restricted to
    cb_subjects. Returns None (with a printed warning) if the file isn't
    there yet — a new cohort is often onboarded one modality at a time.
    """
    path = data_dir / f'{modality}_processed.csv'
    if not path.exists():
        print(f"WARNING: {path} not found; {label} features will be null for every subject")
        return None
    df = pd.read_csv(path, index_col='Subject')
    df = df[df.index.isin(cb_subjects)]
    return df.drop(columns=[c for c in ['SET', 'CENTER'] if c in df.columns])


def create_prospective_feature_dataset(
    data_dir: Path,
    rad_type: str = None,
    dp_type: str = None,
) -> pd.DataFrame:
    """Build the multimodal feature dataframe (Subject + mod1..mod4) for a new
    cohort (e.g. prospective) from already-processed modality CSVs.

    Only cb is mandatory. rad_type/dp_type are optional (pass None to skip
    that modality entirely, e.g. no radiomics available yet); genomics is
    included automatically if 'genomics_processed.csv' exists. Any modality
    that's skipped or whose processed CSV isn't found yet gets a null column
    (with a printed warning) instead of raising, since a new cohort is often
    onboarded one modality at a time.
    """
    data_dir = Path(data_dir)
    cb = pd.read_csv(data_dir / 'cb_processed.csv', index_col='Subject')
    cb_subjects = set(cb.index)
    cb_features = cb.drop(columns=[c for c in ['SET', 'CENTER'] if c in cb.columns])

    rad_features = _load_optional_modality(data_dir, rad_type, cb_subjects, 'rad') if rad_type else None
    if rad_type is None:
        print("WARNING: no --rad-type given; rad features will be null for every subject")
    dp_features = _load_optional_modality(data_dir, dp_type, cb_subjects, 'dp') if dp_type else None
    if dp_type is None:
        print("WARNING: no --dp-type given; dp features will be null for every subject")
    gen_features = _load_optional_modality(data_dir, 'genomics', cb_subjects, 'genomics')

    df = pd.DataFrame(index=cb.index)
    df['Subject'] = df.index
    df['mod1'] = cb_features.apply(lambda r: r.tolist(), axis=1)
    df['mod2'] = df['Subject'].apply(
        lambda s: rad_features.loc[s].tolist() if rad_features is not None and s in rad_features.index else None
    )
    df['mod3'] = df['Subject'].apply(
        lambda s: dp_features.loc[s].tolist() if dp_features is not None and s in dp_features.index else None
    )
    df['mod4'] = df['Subject'].apply(
        lambda s: gen_features.loc[s].tolist() if gen_features is not None and s in gen_features.index else None
    )

    # Null-out all-NaN lists
    for mod in ['mod1', 'mod2', 'mod3', 'mod4']:
        df[mod] = df[mod].apply(
            lambda lst: None if isinstance(lst, list) and all(pd.isna(x) for x in lst) else lst
        )

    print(f"Final shape: {df.shape}")
    for mod in ['mod1', 'mod2', 'mod3', 'mod4']:
        non_null = df[mod].notna().sum()
        print(f"{mod}: {non_null} non-null entries")

    return df.reset_index(drop=True)


def _default_output_name(rad_type: str, dp_type: str) -> str:
    rad_stem = RAD_OUTPUT_STEMS.get(rad_type, f'features_dataset_{rad_type}') if rad_type else 'features_dataset_cb'
    dp_suffix = dp_output_suffix(dp_type) if dp_type else ''
    return f'{rad_stem}{dp_suffix}.parquet'


def _parse_args():
    parser = argparse.ArgumentParser(description="Build the multimodal feature parquet for a new cohort (e.g. prospective), tolerating modalities not available yet.")
    parser.add_argument('--data-dir', default=str(DEFAULT_DATA_DIR), help=f"Directory containing <modality>_processed.csv and where the parquet is written (default: {DEFAULT_DATA_DIR})")
    parser.add_argument('--rad-type', default=None, help="Radiomics source to include (e.g. 'pyradiomics', 'fmrad'). Omit if not available yet for this cohort — mod2 will be null for every subject.")
    parser.add_argument('--dp-type', default=None, help="Digital pathology source to include, matching create_parquet.py's --dp-types (e.g. 'digital_pathology_titan_coral'). Omit if not available yet — mod3 will be null for every subject.")
    parser.add_argument('--output-name', default=None, help="Output parquet filename, written inside --data-dir (default: derived from --rad-type/--dp-type, e.g. 'features_dataset_cb_dp-titan-coral.parquet')")
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    data_dir = Path(args.data_dir)
    df = create_prospective_feature_dataset(data_dir, rad_type=args.rad_type, dp_type=args.dp_type)
    output_name = args.output_name or _default_output_name(args.rad_type, args.dp_type)
    output_path = data_dir / output_name
    df.to_parquet(output_path, index=False)
    print(f"Saved -> {output_path}")
