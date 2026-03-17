#!/bin/bash
# Evaluate every modality model on the "all data available" patient population.
#
# For each modality directory under the given base path, this script:
#   1. Picks the correct subset flags based on the radiomics version in the name
#   2. Iterates over all outcomes (CBR, ORR, DCR, os_months_24, os_months_6)
#   3. Runs compute_subset_metrics.py (skipping if results already exist)
#   4. Collects per-modality CSVs into dlif_pipeline/results/models_on_all_data/
#
# Usage:
#   bash dlif_pipeline/scripts/run_all_models_on_full_population.sh <base_path>
#
# Example:
#   bash dlif_pipeline/scripts/run_all_models_on_full_population.sh \
#       dlif_pipeline/results/C2/CBR/classification/standard/hypothesis_driven/pyrad-noimp

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <base_dir_containing_modality_subdirs>"
    echo "Example: $0 dlif_pipeline/results/C2/CBR/classification/standard/hypothesis_driven/pyrad-noimp"
    exit 1
fi

BASE_DIR="${1%/}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
METRICS_SCRIPT="${SCRIPT_DIR}/pipeline/metrics/compute_subset_metrics.py"
COLLECT_SCRIPT="${SCRIPT_DIR}/sensitivity/collect_all_models_on_full_population.py"

OUTCOMES=(CBR ORR DCR os_months_24 os_months_6)

# ---------------------------------------------------------------------------
# Find which outcome token is in the base path so we can swap it later
# ---------------------------------------------------------------------------
ORIGINAL_OUTCOME=""
for oc in "${OUTCOMES[@]}"; do
    if [[ "$BASE_DIR" == *"/${oc}/"* || "$BASE_DIR" == *"/${oc}" ]]; then
        ORIGINAL_OUTCOME="$oc"
        break
    fi
done

if [ -z "$ORIGINAL_OUTCOME" ]; then
    echo "ERROR: Could not find any known outcome in the path: $BASE_DIR"
    echo "Expected one of: ${OUTCOMES[*]}"
    exit 1
fi

echo "Base directory : $BASE_DIR"
echo "Original outcome: $ORIGINAL_OUTCOME"
echo ""

# ---------------------------------------------------------------------------
# Iterate modality directories
# ---------------------------------------------------------------------------
for MOD_DIR in "$BASE_DIR"/*/; do
    MOD_NAME=$(basename "$MOD_DIR")

    # Only process directories that look like modality folders (start with rwd)
    if [[ "$MOD_NAME" != rwd* ]]; then
        continue
    fi

    # -----------------------------------------------------------------------
    # Determine subset flags and the expected output directory name
    # -----------------------------------------------------------------------
    if [[ "$MOD_NAME" == *"radpy"* ]]; then
        FLAGS="--has_radpy --has_dp --has_genomics"
        SUBSET_DIR_NAME="has_radpy_has_dp_has_genomics"
    elif [[ "$MOD_NAME" == *"radfm"* ]]; then
        FLAGS="--has_fmrad --has_dp --has_genomics"
        SUBSET_DIR_NAME="has_fmrad_has_dp_has_genomics"
    else
        FLAGS="--has_radpy --has_fmrad --has_dp --has_genomics"
        SUBSET_DIR_NAME="has_radpy_has_fmrad_has_dp_has_genomics"
    fi

    # -----------------------------------------------------------------------
    # Iterate over all outcomes
    # -----------------------------------------------------------------------
    for OUTCOME in "${OUTCOMES[@]}"; do
        # Swap the outcome token in the modality directory path
        PATH_FOR_OUTCOME="${MOD_DIR/$ORIGINAL_OUTCOME/$OUTCOME}"
        SEED_DIR="${PATH_FOR_OUTCOME}seed_0"

        if [ ! -d "$SEED_DIR" ]; then
            echo "SKIP: $SEED_DIR does not exist"
            continue
        fi

        # Find the latest mb_attention_mil eval directory
        EVAL_DIR="$SEED_DIR/eval"
        if [ ! -d "$EVAL_DIR" ]; then
            echo "SKIP: no eval/ directory in $SEED_DIR"
            continue
        fi

        MB_DIR=$(ls -d "$EVAL_DIR"/*mb_attention_mil 2>/dev/null | sort | tail -1)
        if [ -z "$MB_DIR" ]; then
            echo "SKIP: no mb_attention_mil directory in $EVAL_DIR"
            continue
        fi

        # Skip if the subset results already exist
        if [ -d "$MB_DIR/$SUBSET_DIR_NAME" ]; then
            echo "SKIP (exists): $MB_DIR/$SUBSET_DIR_NAME"
            continue
        fi

        echo ""
        echo "================================================================"
        echo "MODALITY: $MOD_NAME | OUTCOME: $OUTCOME"
        echo "FLAGS:    $FLAGS"
        echo "PATH:     $SEED_DIR"
        echo "================================================================"

        python "$METRICS_SCRIPT" --path "$SEED_DIR" $FLAGS
    done
done

# ---------------------------------------------------------------------------
# Collect results into models_on_all_data/
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Collecting results ..."
echo "================================================================"
python "$COLLECT_SCRIPT" "$BASE_DIR"

echo ""
echo "Done."
