#!/usr/bin/env bash
# Quantitative autointerp pipeline (Stages 0–4) for a single model.
#
# Usage (from new-code/experiments/):
#   bash autointerp/run_autointerp_pipeline.sh <model> [task] [device]
#
# Arguments:
#   model   TransformerLens model name (gpt2-small, EleutherAI/pythia-160m, gemma-2-2b)
#   task    Task identifier (default: ioi-balanced)
#   device  Torch device (default: cuda)
#
# Pipeline:
#   Stage 0: Collect Pile activations  (skipped if output already exists)
#   Stage 1: Extract top-K activations (skipped per layer if output already exists)
#   Stage 2: Generate interpretations  (one shard per layer, start=0 end=999999)
#   Stage 2b: Merge shards
#   Stage 3: Score interpretations
#   Stage 4: Compute metrics + generate Table 1 and Figure 32
#
# Prerequisites:
#   - Library installed: pip install -e .   (from new-code/)
#   - Signals H5 from Part 2 extract_signals.py in:
#       data/clustering/{model_short}/signals_balanced_{model_short}_not-norm.h5
#
# Note: Stages 0–3 are GPU-intensive. For large-scale runs on an HPC cluster,
# use the per-stage array job scripts in scc/ instead of this sequential script.

set -euo pipefail

MODEL=${1:?"Usage: $0 <model> [task] [device]"}
TASK=${2:-"ioi-balanced"}
DEVICE=${3:-"cuda"}
DATA_DIR="data/autointerp"
CLUSTERING_DIR="data/clustering"

# Determine number of layers per model
case "$MODEL" in
    gpt2-small|gpt2)
        N_LAYERS=12 ;;
    EleutherAI/pythia-160m|pythia-160m)
        N_LAYERS=12 ;;
    google/gemma-2-2b|gemma-2-2b)
        N_LAYERS=26 ;;
    *)
        N_LAYERS=${N_LAYERS:?"Unknown model '${MODEL}'. Set N_LAYERS=<n> before running."}
        ;;
esac

# Derive short name for file paths (strip org prefix e.g. EleutherAI/)
MODEL_SHORT="${MODEL##*/}"

SIGNALS_FILE="${CLUSTERING_DIR}/${MODEL_SHORT}/signals_balanced_${MODEL_SHORT}_not-norm.h5"
PILE_FILE="${DATA_DIR}/pile_activations_${MODEL_SHORT}.h5"
SHARDS_DIR="${DATA_DIR}/shards"
RESULTS_DIR="${DATA_DIR}/results"

echo "=== Autointerp quantitative pipeline ==="
echo "  model=${MODEL}  task=${TASK}  device=${DEVICE}  N_LAYERS=${N_LAYERS}"
echo ""

# -----------------------------------------------------------------------
# Stage 0: Collect Pile activations
# -----------------------------------------------------------------------
if [ -f "$PILE_FILE" ]; then
    echo "--- Stage 0: SKIPPED (${PILE_FILE} already exists) ---"
else
    echo "--- Stage 0: Collecting Pile activations ---"
    python autointerp/collect_pile_activations.py \
        --model "$MODEL" \
        --device "$DEVICE" \
        --output_dir "$DATA_DIR"
fi
echo ""

# -----------------------------------------------------------------------
# Stage 1: Extract top-K activations (per layer)
# -----------------------------------------------------------------------
echo "--- Stage 1: Extracting top-K activations (layers 1–${N_LAYERS}) ---"
for layer in $(seq 1 "$N_LAYERS"); do
    OUT_FILE="${DATA_DIR}/activations_${layer}_${TASK}_${MODEL_SHORT}.h5"
    if [ -f "$OUT_FILE" ]; then
        echo "  Layer ${layer}: skipped (${OUT_FILE} exists)"
    else
        echo "  Layer ${layer}..."
        python autointerp/extract_top_activations.py \
            --model "$MODEL" \
            --task "$TASK" \
            --layer "$layer" \
            --signals_file "$SIGNALS_FILE" \
            --pile_activations "$PILE_FILE" \
            --output_dir "$DATA_DIR" \
            --device "$DEVICE"
    fi
done
echo ""

# -----------------------------------------------------------------------
# Stage 2: Generate interpretations (per layer, single shard)
# -----------------------------------------------------------------------
echo "--- Stage 2: Generating interpretations (layers 1–${N_LAYERS}) ---"
mkdir -p "$SHARDS_DIR"
for layer in $(seq 1 "$N_LAYERS"); do
    echo "  Layer ${layer}..."
    python autointerp/interpret_signals.py \
        --model "$MODEL" \
        --layer "$layer" \
        --start 0 \
        --end 999999 \
        --activations_file "${DATA_DIR}/activations_${layer}_${TASK}_${MODEL_SHORT}.h5" \
        --output_dir "$SHARDS_DIR" \
        --device "$DEVICE"
done
echo ""

# -----------------------------------------------------------------------
# Stage 2b: Merge shards
# -----------------------------------------------------------------------
echo "--- Stage 2b: Merging shards ---"
python autointerp/merge_shards.py \
    --model "$MODEL" \
    --task "$TASK" \
    --layers $(seq 1 "$N_LAYERS") \
    --activations_dir "$DATA_DIR" \
    --shards_dir "$SHARDS_DIR"
echo ""

# -----------------------------------------------------------------------
# Stage 3: Score interpretations (per layer)
# -----------------------------------------------------------------------
echo "--- Stage 3: Scoring interpretations (layers 1–${N_LAYERS}) ---"
for layer in $(seq 1 "$N_LAYERS"); do
    echo "  Layer ${layer}..."
    python autointerp/score_interpretations.py \
        --model "$MODEL" \
        --layer "$layer" \
        --activations_file "${DATA_DIR}/activations_${layer}_${TASK}_${MODEL_SHORT}.h5" \
        --device "$DEVICE"
done
echo ""

# -----------------------------------------------------------------------
# Stage 4: Compute metrics + generate Table 1 and Figure 32
# -----------------------------------------------------------------------
echo "--- Stage 4: Computing metrics ---"
mkdir -p "$RESULTS_DIR"
python autointerp/compute_metrics.py \
    --models "$MODEL" \
    --task "$TASK" \
    --activations_dir "$DATA_DIR" \
    --output_dir "$RESULTS_DIR"
echo ""

echo "=== Done ==="
echo "  Table 1:   ${RESULTS_DIR}/table1.tex"
echo "  Figure 32: ${RESULTS_DIR}/figure32_fdr_significance.pdf"
echo "  Metrics:   ${RESULTS_DIR}/metrics.csv"
