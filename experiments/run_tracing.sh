#!/usr/bin/env bash
# Trace per-prompt circuit graphs for all model/task combinations (Appendix E).
#
# Produces graphml files in data/traced_graphs/{model}/{task}/.
#
# Usage:
#   bash experiments/run_tracing.sh [DEVICE]
#
# DEVICE defaults to "mps".  Use "cuda" for GPU or "cpu" for CPU.

set -euo pipefail

DEVICE="${1:-mps}"
SCRIPT="experiments/trace.py"
OUT_BASE="data/traced_graphs"

echo "=== Tracing per-prompt circuit graphs (device=$DEVICE, output=$OUT_BASE/) ==="
echo ""

# --- GPT-2 small ---
echo ">>> gpt2-small / ioi  (n=256)"
python "$SCRIPT" -m gpt2-small -t ioi -n 256 -s 0 -at dynamic -d "$DEVICE" \
    --label_mode roles -o "$OUT_BASE/gpt2-small/ioi"

echo ">>> gpt2-small / gt  (n=256)"
python "$SCRIPT" -m gpt2-small -t gt -n 256 -s 0 -at dynamic -d "$DEVICE" \
    --label_mode roles -o "$OUT_BASE/gpt2-small/gt"

echo ">>> gpt2-small / gp  (n=100)"
python "$SCRIPT" -m gpt2-small -t gp -s 0 -at dynamic -d "$DEVICE" \
    --label_mode roles -o "$OUT_BASE/gpt2-small/gp"

# --- Pythia-160m ---
echo ">>> pythia-160m / ioi  (n=256)"
python "$SCRIPT" -m EleutherAI/pythia-160m -t ioi -n 256 -s 0 -at dynamic -d "$DEVICE" \
    --label_mode roles -o "$OUT_BASE/pythia-160m/ioi"

echo ">>> pythia-160m / gt  (n=256)"
python "$SCRIPT" -m EleutherAI/pythia-160m -t gt -n 256 -s 0 -at dynamic -d "$DEVICE" \
    --label_mode roles -o "$OUT_BASE/pythia-160m/gt"

echo ">>> pythia-160m / gp  (n=100)"
python "$SCRIPT" -m EleutherAI/pythia-160m -t gp -s 0 -at dynamic -d "$DEVICE" \
    --label_mode roles -o "$OUT_BASE/pythia-160m/gp"

# --- Gemma-2-2b ---
echo ">>> gemma-2-2b / ioi  (n=256)"
python "$SCRIPT" -m gemma-2-2b -t ioi -n 256 -s 0 -at dynamic -d "$DEVICE" \
    --label_mode roles -o "$OUT_BASE/gemma-2-2b/ioi"

echo ">>> gemma-2-2b / gp  (n=100)"
python "$SCRIPT" -m gemma-2-2b -t gp -s 0 -at dynamic -d "$DEVICE" \
    --label_mode roles -o "$OUT_BASE/gemma-2-2b/gp"

echo ""
echo "=== All tracing complete. Graphs saved to $OUT_BASE/ ==="
