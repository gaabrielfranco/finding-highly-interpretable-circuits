#!/usr/bin/env bash
# Master pipeline for Section 3 / Appendix F: Clustering and Signal Analysis.
#
# Orchestrates Steps 2–7 of the clustering pipeline. Step 1 (tracing) runs
# separately via run_tracing_balanced.sh since it's expensive.
#
# Prerequisites:
#   - Traced graphs in data/traced_graphs/{model}/ioi-balanced/ (from Step 1)
#   - Library installed: pip install -e .  (from new-code/)
#
# Usage:
#   bash experiments/run_clustering_pipeline.sh [DEVICE]
#
# DEVICE defaults to "mps".  Use "cuda" for GPU or "cpu" for CPU.
# Steps 2–4 and 7 run on CPU only. Steps 5–6 load models (GPU recommended).

set -euo pipefail

DEVICE="${1:-mps}"
MODELS=("gpt2-small" "pythia-160m" "gemma-2-2b")

GRAPH_BASE="data/traced_graphs"
DATA_BASE="data/clustering"
FIG_DIR="figures/clustering"

# ---------------------------------------------------------------------------
# Helper: extract representative prompt ID from JSON
#   Usage: get_repr_id <json_file> <component_type> <template_key>
#   Example: get_repr_id ".../gpt2-small_representatives.json" "sv_as_component" "BABA"
# ---------------------------------------------------------------------------
get_repr_id() {
    python3 -c "
import json
with open('$1') as f:
    d = json.load(f)
print(d['$2']['$3']['id'])
"
}

# TransformerLens model names (extract_signals.py uses TL names)
# Note: uses a function instead of declare -A for macOS bash 3.2 compatibility.
tl_name() {
    case "$1" in
        pythia-160m) echo "EleutherAI/pythia-160m" ;;
        *) echo "$1" ;;
    esac
}

echo "=== Clustering Pipeline (Steps 2–7, device=$DEVICE) ==="
echo ""

# # -----------------------------------------------------------------------
# # Step 2: Extract component vectors from traced graphs
# # -----------------------------------------------------------------------
echo ">>> Step 2: process_graphs.py"
for model in "${MODELS[@]}"; do
    echo "  $model"
    python experiments/process_graphs.py -m "$model"
done
echo ""

# # -----------------------------------------------------------------------
# # Step 3: Clustermaps and distributions
# # -----------------------------------------------------------------------
echo ">>> Step 3: plot_clustering.py"
for model in "${MODELS[@]}"; do
    echo "  $model (per-model)"
    python experiments/plot_clustering.py -m "$model"
done
echo "  combined (3-model)"
python experiments/plot_clustering.py --combined
echo ""

# # -----------------------------------------------------------------------
# # Step 4: Find representative prompts per template
# # -----------------------------------------------------------------------
echo ">>> Step 4: find_representatives.py"
for model in "${MODELS[@]}"; do
    echo "  $model"
    python experiments/find_representatives.py -m "$model"
done
echo ""

# -----------------------------------------------------------------------
# Step 5: Extract signals (requires model loading — GPU recommended)
#
# Skips models whose H5 file already exists. Delete the H5 to re-extract.
# -----------------------------------------------------------------------
echo ">>> Step 5: extract_signals.py (device=$DEVICE)"
for model in "${MODELS[@]}"; do
    H5_FILE="$DATA_BASE/$model/signals_balanced_${model}_not-norm.h5"
    if [ -f "$H5_FILE" ]; then
        echo "  $model: SKIPPED ($H5_FILE exists)"
    else
        echo "  $model"
        python experiments/extract_signals.py \
            -m "$(tl_name "$model")" -d "$DEVICE" --batch_size 8
    fi
done
echo ""

# -----------------------------------------------------------------------
# Step 6: Signal comparison heatmaps
#
# Representative prompt IDs are read from Step 4's JSON output.
# JSON keys are 0-indexed ("Template 0"–"Template 14").
# Labels use 1-indexed convention (paper's T1–T15).
# Mapping: label "Template N" → JSON key "Template {N-1}".
# -----------------------------------------------------------------------
echo ">>> Step 6: plot_signals.py"

COMP_TYPE="sv_as_component"

# --- Figure 2 pairs (required for combined figure) ---

# GPT-2: BABA vs ABBA
JSON_FILE="$FIG_DIR/gpt2-small_representatives.json"
P1=$(get_repr_id "$JSON_FILE" "$COMP_TYPE" "BABA")
P2=$(get_repr_id "$JSON_FILE" "$COMP_TYPE" "ABBA")
echo "  gpt2-small: BABA (id=$P1) vs ABBA (id=$P2)"
python experiments/plot_signals.py -m gpt2-small \
    -p1 "$P1" -p2 "$P2" -l1 "BABA" -l2 "ABBA"

# Pythia: Template 9 vs 10 (1-indexed → JSON keys "Template 8", "Template 9")
JSON_FILE="$FIG_DIR/pythia-160m_representatives.json"
P1=$(get_repr_id "$JSON_FILE" "$COMP_TYPE" "Template 8")
P2=$(get_repr_id "$JSON_FILE" "$COMP_TYPE" "Template 9")
echo "  pythia-160m: T9 (id=$P1) vs T10 (id=$P2)"
python experiments/plot_signals.py -m pythia-160m \
    -p1 "$P1" -p2 "$P2" -l1 "Template 9" -l2 "Template 10"

# Gemma: Template 14 vs 15 (1-indexed → JSON keys "Template 13", "Template 14")
JSON_FILE="$FIG_DIR/gemma-2-2b_representatives.json"
P1=$(get_repr_id "$JSON_FILE" "$COMP_TYPE" "Template 13")
P2=$(get_repr_id "$JSON_FILE" "$COMP_TYPE" "Template 14")
echo "  gemma-2-2b: T14 (id=$P1) vs T15 (id=$P2)"
python experiments/plot_signals.py -m gemma-2-2b \
    -p1 "$P1" -p2 "$P2" -l1 "Template 14" -l2 "Template 15"

# Combined Figure 2 (reads NPZ files produced above)
echo "  combined (Figure 2)"
python experiments/plot_signals.py --combined

echo ""

# --- Appendix F pairs (additional per-model heatmaps) ---

echo ">>> Step 6 (Appendix): additional signal heatmaps"

# Pythia additional pairs
JSON_FILE="$FIG_DIR/pythia-160m_representatives.json"

P1=$(get_repr_id "$JSON_FILE" "$COMP_TYPE" "Template 12")
P2=$(get_repr_id "$JSON_FILE" "$COMP_TYPE" "Template 4")
echo "  pythia-160m: T13 vs T5"
python experiments/plot_signals.py -m pythia-160m \
    -p1 "$P1" -p2 "$P2" -l1 "Template 13" -l2 "Template 5"

P1=$(get_repr_id "$JSON_FILE" "$COMP_TYPE" "Template 7")
P2=$(get_repr_id "$JSON_FILE" "$COMP_TYPE" "Template 6")
echo "  pythia-160m: T8 vs T7"
python experiments/plot_signals.py -m pythia-160m \
    -p1 "$P1" -p2 "$P2" -l1 "Template 8" -l2 "Template 7"

P1=$(get_repr_id "$JSON_FILE" "$COMP_TYPE" "Template 13")
P2=$(get_repr_id "$JSON_FILE" "$COMP_TYPE" "Template 14")
echo "  pythia-160m: T14 vs T15"
python experiments/plot_signals.py -m pythia-160m \
    -p1 "$P1" -p2 "$P2" -l1 "Template 14" -l2 "Template 15"

P1=$(get_repr_id "$JSON_FILE" "$COMP_TYPE" "Template 12")
P2=$(get_repr_id "$JSON_FILE" "$COMP_TYPE" "Template 9")
echo "  pythia-160m: T13 vs T10"
python experiments/plot_signals.py -m pythia-160m \
    -p1 "$P1" -p2 "$P2" -l1 "Template 13" -l2 "Template 10"

P1=$(get_repr_id "$JSON_FILE" "$COMP_TYPE" "Template 7")
P2=$(get_repr_id "$JSON_FILE" "$COMP_TYPE" "Template 5")
echo "  pythia-160m: T8 vs T6"
python experiments/plot_signals.py -m pythia-160m \
    -p1 "$P1" -p2 "$P2" -l1 "Template 8" -l2 "Template 6"

# Gemma additional pairs
JSON_FILE="$FIG_DIR/gemma-2-2b_representatives.json"

P1=$(get_repr_id "$JSON_FILE" "$COMP_TYPE" "Template 4")
P2=$(get_repr_id "$JSON_FILE" "$COMP_TYPE" "Template 12")
echo "  gemma-2-2b: T5 vs T13"
python experiments/plot_signals.py -m gemma-2-2b \
    -p1 "$P1" -p2 "$P2" -l1 "Template 5" -l2 "Template 13"

# Gemma: Cluster B vs Cluster A (manually selected IDs, not from JSON)
echo "  gemma-2-2b: Cluster B vs Cluster A"
python experiments/plot_signals.py -m gemma-2-2b \
    -p1 1786 -p2 2993 -l1 "Cluster B" -l2 "Cluster A"

echo ""

# -----------------------------------------------------------------------
# Step 7: LaTeX tables (Tables 3, 4, 5)
# -----------------------------------------------------------------------
echo ">>> Step 7: generate_tables.py"
python experiments/generate_tables.py -o "$FIG_DIR/tables.tex"
echo ""

echo "=== Pipeline complete ==="
echo ""
echo "Output:"
echo "  $DATA_BASE/{model}/                  (processed data + NPZ files)"
echo "  $FIG_DIR/                             (all figures)"
echo "  $FIG_DIR/tables.tex                   (Tables 3–5)"
