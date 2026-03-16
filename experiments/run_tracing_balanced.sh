#!/usr/bin/env bash
# Trace per-prompt circuit graphs for balanced IOI (Section 3 / Appendix F).
#
# Balanced IOI always generates exactly 3,000 prompts (100 per 15 templates × 2).
# The -n flag is not needed — the dataset size is fixed.
#
# All models run in batches (3000 prompts is too large for a single forward pass).
#
# Produces graphml files in data/traced_graphs/{model}/ioi-balanced/.
#
# Usage:
#   bash experiments/run_tracing_balanced.sh [DEVICE]
#
# DEVICE defaults to "mps".  Use "cuda" for GPU or "cpu" for CPU.

set -euo pipefail

DEVICE="${1:-mps}"
SCRIPT="experiments/trace.py"
OUT_BASE="data/traced_graphs"

echo "=== Tracing balanced IOI circuits (device=$DEVICE, output=$OUT_BASE/) ==="
echo ""

# --- GPT-2 small (3000 prompts, batch_size=128) ---
echo ">>> gpt2-small / ioi-balanced (n=3000, batch_size=128)"
python "$SCRIPT" -m gpt2-small -t ioi-balanced -s 0 -at dynamic -d "$DEVICE" \
    --label_mode tokens --batch_size 128 -o "$OUT_BASE/gpt2-small/ioi-balanced"

echo ""

# --- Pythia-160m (3000 prompts, batch_size=128) ---
echo ">>> pythia-160m / ioi-balanced (n=3000, batch_size=128)"
python "$SCRIPT" -m EleutherAI/pythia-160m -t ioi-balanced -s 0 -at dynamic -d "$DEVICE" \
    --label_mode tokens --batch_size 128 -o "$OUT_BASE/pythia-160m/ioi-balanced"

echo ""

# --- Gemma-2-2b (3000 prompts, batch_size=128) ---
echo ">>> gemma-2-2b / ioi-balanced (n=3000, batch_size=128)"
python "$SCRIPT" -m gemma-2-2b -t ioi-balanced -s 0 -at dynamic -d "$DEVICE" \
    --label_mode tokens --batch_size 128 -o "$OUT_BASE/gemma-2-2b/ioi-balanced"

echo ""
echo "=== All balanced IOI tracing complete. Graphs saved to $OUT_BASE/ ==="
