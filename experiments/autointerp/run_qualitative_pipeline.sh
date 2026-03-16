#!/usr/bin/env bash
# Qualitative autointerp pipeline (Stage 5) for representative prompts.
#
# Usage (from new-code/experiments/):
#   bash autointerp/run_qualitative_pipeline.sh <model> <task> <representative_ids...>
#
# Arguments:
#   model              TransformerLens model name (e.g. gpt2-small)
#   task               Task identifier (e.g. ioi-balanced, facts)
#   representative_ids Space-separated prompt IDs (e.g. 129 1613 1372)
#
# Two-phase design:
#   Phase 1 (automatic): Build Gemini JSONL + edge examples JSON
#   Phase 2 (manual):    User sends JSONL to Gemini batch API, downloads response
#   Phase 3 (automatic): Parse response, annotate graphs, build Bokeh HTML viewers
#                        Run with PARSE_RESPONSE=1 to execute Phase 3
#
# Example (IOI, GPT-2, BABA/ABBA/ALL representatives):
#   # Phase 1: build request
#   bash autointerp/run_qualitative_pipeline.sh gpt2-small ioi-balanced 129 1613 1372
#
#   # (send prompts_representatives_gpt2-small_ioi-balanced.jsonl to Gemini API)
#
#   # Phase 3: parse + visualize
#   PARSE_RESPONSE=1 bash autointerp/run_qualitative_pipeline.sh \
#       gpt2-small ioi-balanced 129 1613 1372
#
# Prerequisites:
#   - Stage 1 activations in: data/autointerp/activations_{layer}_{task}_{model_short}.h5
#   - Signals H5 in:          data/clustering/{model_short}/signals_balanced_*_not-norm.h5
#   - Traced GraphML in:      data/traced_graphs/{model_short}/{task}/*.graphml

set -euo pipefail

MODEL=${1:?"Usage: $0 <model> <task> <representative_ids...>"}
TASK=${2:?"Usage: $0 <model> <task> <representative_ids...>"}
shift 2
REPRESENTATIVES="$*"

if [ -z "$REPRESENTATIVES" ]; then
    echo "Error: at least one representative prompt ID is required." >&2
    exit 1
fi

MODEL_SHORT="${MODEL##*/}"
DATA_DIR="data/autointerp"
CLUSTERING_DIR="data/clustering"
GRAPHS_DIR="data/traced_graphs/${MODEL_SHORT}/${TASK}"
OUTPUT_DIR="${DATA_DIR}/graphs_interp"

JSONL_FILE="${DATA_DIR}/prompts_representatives_${MODEL_SHORT}_${TASK}.jsonl"
EXAMPLES_FILE="${DATA_DIR}/prompts_representatives_${MODEL_SHORT}_${TASK}_edge_examples.json"
RESPONSE_FILE="${DATA_DIR}/prompts_representatives_${MODEL_SHORT}_${TASK}_interpretation.jsonl"
INTERP_FILE="${DATA_DIR}/interpretations_representatives_${MODEL_SHORT}_${TASK}.json"
SIGNALS_FILE="${CLUSTERING_DIR}/${MODEL_SHORT}/signals_balanced_${MODEL_SHORT}_not-norm.h5"

echo "=== Qualitative autointerp pipeline ==="
echo "  model=${MODEL}  task=${TASK}  representatives=${REPRESENTATIVES}"
echo ""

# -----------------------------------------------------------------------
# Phase 1: Build Gemini batch JSONL + edge examples side output
# -----------------------------------------------------------------------
echo "--- Step 5a: Building Gemini batch request ---"
python autointerp/interpret_representatives.py build-request \
    --model "$MODEL" \
    --task "$TASK" \
    --representatives $REPRESENTATIVES \
    --signals_file "$SIGNALS_FILE" \
    --activations_dir "$DATA_DIR" \
    --output "$JSONL_FILE"

echo ""
echo "============================================================"
echo "MANUAL STEP:"
echo "  1. Send '${JSONL_FILE}' to the Gemini batch API"
echo "     (using your own API key — see Google AI Studio)"
echo "  2. Download the response to:"
echo "     '${RESPONSE_FILE}'"
echo "  3. Then re-run with PARSE_RESPONSE=1:"
echo "     PARSE_RESPONSE=1 bash autointerp/run_qualitative_pipeline.sh \\"
echo "         ${MODEL} ${TASK} ${REPRESENTATIVES}"
echo "============================================================"
echo ""

if [ "${PARSE_RESPONSE:-0}" != "1" ]; then
    exit 0
fi

# -----------------------------------------------------------------------
# Phase 3: Parse Gemini response
# -----------------------------------------------------------------------
echo "--- Step 5b: Parsing Gemini response ---"
python autointerp/interpret_representatives.py parse-response \
    --response_file "$RESPONSE_FILE" \
    --output_file "$INTERP_FILE"
echo ""

# -----------------------------------------------------------------------
# Phase 3: Annotate circuit graphs
# -----------------------------------------------------------------------
echo "--- Step 5c: Annotating circuit graphs ---"
mkdir -p "$OUTPUT_DIR"
python autointerp/annotate_graphs.py \
    --model "$MODEL" \
    --task "$TASK" \
    --representatives $REPRESENTATIVES \
    --interpretations_file "$INTERP_FILE" \
    --edge_examples_file "$EXAMPLES_FILE" \
    --signals_file "$SIGNALS_FILE" \
    --graphs_dir "$GRAPHS_DIR" \
    --output_dir "$OUTPUT_DIR"
echo ""

# -----------------------------------------------------------------------
# Phase 3: Build Bokeh HTML viewer for each representative
# -----------------------------------------------------------------------
echo "--- Step 5d: Building Bokeh HTML viewers ---"
for pid in $REPRESENTATIVES; do
    # Pattern: {model_short}_{task}_n{N}_{pid}_0_dynamic_ig.graphml
    # The n{N} part is variable (e.g. n3000 for balanced IOI), matched by glob.
    for graphml in "${OUTPUT_DIR}/${MODEL_SHORT}_${TASK}_n"*"_${pid}_0_dynamic_ig.graphml"; do
        if [ ! -f "$graphml" ]; then
            echo "  [warn] No GraphML found for prompt ${pid} (pattern: ${graphml})"
            continue
        fi
        stem="${graphml%.graphml}"
        tokens_file="${stem}-tokens.json"
        examples_json="${stem}_edge_examples.json"
        output_html="${stem}.html"

        extra_args=""
        if [ -f "$tokens_file" ]; then
            extra_args="--tokens ${tokens_file} --xaxis-labels tokens"
        fi
        if [ -f "$examples_json" ]; then
            extra_args="${extra_args} --examples ${examples_json}"
        fi

        echo "  Prompt ${pid}: ${graphml##*/} → ${output_html##*/}"
        # shellcheck disable=SC2086
        python autointerp/view_circuit.py "$graphml" "$output_html" $extra_args
    done
done
echo ""

echo "=== Done ==="
echo "  HTML viewers in: ${OUTPUT_DIR}/"
