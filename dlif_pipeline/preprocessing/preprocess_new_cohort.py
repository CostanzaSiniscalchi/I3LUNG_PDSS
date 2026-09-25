"""Transform-only preprocessing for a new cohort (e.g. prospective), reusing the
imputers/scalers fit on the retrospective cohort.

Usage (defaults: looks for cb.csv/genomics.csv/digital_pathology.csv/pyradiomics.csv/
fmrad.csv in --raw-data-dir, skipping any that aren't present):
    python dlif_pipeline/preprocessing/preprocess_new_cohort.py \\
        --raw-data-dir data/prospective \\
        --artifacts-dir data \\
        --output-dir data/prospective

Only have some modalities so far (e.g. cb, genomics, and one or more digital
pathology variants, with radpy/radfm not available yet)? Just don't put those
files in --raw-data-dir (or don't pass a --dp-types matching them) — they're
skipped with a warning rather than erroring, and downstream columns are simply
left null, same as create_parquet.py/create_annotations.py already handle.

Process one or more digital pathology source variants (matching
create_parquet.py's --dp-types) instead of the legacy 'digital_pathology' default:
    python dlif_pipeline/preprocessing/preprocess_new_cohort.py \\
        --raw-data-dir data/prospective \\
        --artifacts-dir data \\
        --output-dir data/prospective \\
        --dp-types digital_pathology_titan digital_pathology_titan_coral

Override the file used for a specific fixed modality instead of the default
'<raw-data-dir>/<modality>.csv':
    python dlif_pipeline/preprocessing/preprocess_new_cohort.py \\
        --raw-data-dir data/prospective \\
        --artifacts-dir data \\
        --output-dir data/prospective \\
        --genomics-path data/prospective/genomics_v2.csv
"""
import argparse
import sys
from pathlib import Path

# Add the utils/ subdirectory itself (not this script's own directory) so the
# import below stays flat and never makes "utils" resolve to this folder for
# other code that expects the repo-root "utils" package.
sys.path.insert(0, str(Path(__file__).parent / 'utils'))
from transform_cohort import DEFAULT_FILENAMES, preprocess_new_cohort  # noqa: E402

# Fixed, non-dp modalities. Digital pathology is handled separately via
# --dp-types since it's an open-ended set of source variants, not a fixed list.
MODALITIES = list(DEFAULT_FILENAMES)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw-data-dir', required=True, help="Directory containing the new cohort's raw modality CSVs")
    parser.add_argument('--artifacts-dir', required=True, help="Retrospective data directory holding the fitted *_imputer.pkl / *_scaler.pkl / *_norm_config.json artifacts")
    parser.add_argument('--output-dir', required=True, help="Where to write <modality>_processed.csv")
    parser.add_argument('--modalities', nargs='+', default=MODALITIES, choices=MODALITIES, help="Which fixed (non-dp) modalities to process (default: all)")
    parser.add_argument('--dp-types', nargs='+', default=['digital_pathology'], help="Digital pathology source(s) to process, matching create_parquet.py's --dp-types. 'digital_pathology' (default) is the legacy/gigapath source; any 'digital_pathology_<token>' (e.g. 'digital_pathology_titan', 'digital_pathology_titan_coral') is read from '<raw-data-dir>/digital_pathology_<token>.csv' and transformed with the artifacts impute_and_normalize.py saved under that same name.")
    for modality in MODALITIES:
        parser.add_argument(
            f'--{modality.replace("_", "-")}-path',
            default=None,
            help=f"Override the raw CSV used for '{modality}' instead of the default '<raw-data-dir>/{DEFAULT_FILENAMES[modality]}'",
        )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    input_paths = {modality: getattr(args, f"{modality}_path") for modality in MODALITIES}

    preprocess_new_cohort(
        raw_data_dir=Path(args.raw_data_dir),
        artifacts_dir=Path(args.artifacts_dir),
        output_dir=Path(args.output_dir),
        modalities=args.modalities,
        dp_types=args.dp_types,
        input_paths=input_paths,
    )
