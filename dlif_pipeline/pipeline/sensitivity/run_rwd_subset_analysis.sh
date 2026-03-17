#!/bin/bash
# Run subset analysis for all outcomes and modality combinations
#
# Usage:
#   bash dlif_pipeline/scripts/run_subset_analysis.sh <base_path>
#
# Example:
#   bash dlif_pipeline/scripts/run_subset_analysis.sh \
#       dlif_pipeline/results/C2/CBR/classification/standard/hypothesis_driven/pyrad-noimp/rwd/seed_0/eval/00000-mb_attention_mil
#
# The script replaces the outcome in the path to iterate over all outcomes.

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_any_outcome_eval_dir>"
    echo "Example: $0 dlif_pipeline/results/C2/CBR/classification/standard/hypothesis_driven/pyrad-noimp/rwd/seed_0/eval/00000-mb_attention_mil"
    exit 1
fi

TEMPLATE_PATH="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
METRICS_SCRIPT="${SCRIPT_DIR}/pipeline/metrics/compute_subset_metrics.py"
AGG_SCRIPT="${SCRIPT_DIR}/scripts/collect_subset_results.py"

OUTCOMES=(CBR ORR DCR os_months_24 os_months_6)

# Has all modalities

SUBSET_FLAGS=(
    "--has_dp --has_genomics --has_radpy"
    # "--has_dp --has_genomics --has_fmrad"
)

# SUBSET_FLAGS=(
#     "--has_radpy"
#     "--has_fmrad"
#     "--has_dp"
#     "--has_radpy --has_dp"
#     "--has_fmrad --has_dp"
#     "--has_genomics"
#     "--has_dp --has_genomics"
#     "--has_dp --has_genomics --has_radpy"
#     "--has_dp --has_genomics --has_fmrad"
#     "--no_radpy"
#     "--no_fmrad"
#     "--no_dp"
#     "--no_radpy --no_dp"
#     "--no_fmrad --no_dp"
#     "--no_genomics"
#     "--no_dp --no_genomics"
#     "--no_dp --no_genomics --no_radpy"
#     "--no_dp --no_genomics --no_fmrad"
# )

# Figure out which token in the path is the outcome so we can swap it
# Try each outcome to find which one is in the template path
ORIGINAL_OUTCOME=""
for oc in "${OUTCOMES[@]}"; do
    if [[ "$TEMPLATE_PATH" == *"/${oc}/"* ]]; then
        ORIGINAL_OUTCOME="$oc"
        break
    fi
done

if [ -z "$ORIGINAL_OUTCOME" ]; then
    echo "ERROR: Could not find any known outcome in the path: $TEMPLATE_PATH"
    exit 1
fi

for OUTCOME in "${OUTCOMES[@]}"; do
    PATH_FOR_OUTCOME="${TEMPLATE_PATH/$ORIGINAL_OUTCOME/$OUTCOME}"

    if [ ! -d "$PATH_FOR_OUTCOME" ]; then
        echo "SKIP: $PATH_FOR_OUTCOME does not exist"
        continue
    fi

    echo ""
    echo "================================================================"
    echo "OUTCOME: $OUTCOME"
    echo "PATH:    $PATH_FOR_OUTCOME"
    echo "================================================================"

    for FLAGS in "${SUBSET_FLAGS[@]}"; do
        echo ""
        echo "--- $FLAGS ---"
        python "$METRICS_SCRIPT" --path "$PATH_FOR_OUTCOME" $FLAGS
    done
done
python "$AGG_SCRIPT" "$PATH_FOR_OUTCOME"

echo ""
echo "Done."
