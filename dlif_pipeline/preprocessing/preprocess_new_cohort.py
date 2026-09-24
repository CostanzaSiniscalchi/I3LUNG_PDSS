"""Transform-only preprocessing for a new cohort (e.g. prospective), reusing the
imputers/scalers fit on the retrospective cohort.

Usage (defaults: looks for cb.csv/genomics.csv/digital_pathology.csv/pyradiomics.csv/
fmrad.csv in --raw-data-dir, skipping any that aren't present):
    python dlif_pipeline/preprocessing/preprocess_new_cohort.py \\
        --raw-data-dir data/prospective \\
        --artifacts-dir data \\
        --output-dir data/prospective_processed

Override the file used for a specific modality instead of the default
'<raw-data-dir>/<modality>.csv':
    python dlif_pipeline/preprocessing/preprocess_new_cohort.py \\
        --raw-data-dir data/prospective \\
        --artifacts-dir data \\
        --output-dir data/prospective_processed \\
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

MODALITIES = list(DEFAULT_FILENAMES)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw-data-dir', required=True, help="Directory containing the new cohort's raw modality CSVs")
    parser.add_argument('--artifacts-dir', required=True, help="Retrospective data directory holding the fitted *_imputer.pkl / *_scaler.pkl / *_norm_config.json artifacts")
    parser.add_argument('--output-dir', required=True, help="Where to write <modality>_processed.csv")
    parser.add_argument('--modalities', nargs='+', default=MODALITIES, choices=MODALITIES, help="Which modalities to process (default: all)")
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
        input_paths=input_paths,
    )
