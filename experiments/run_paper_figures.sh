#!/usr/bin/env bash
# Generate paper figures from Appendices B, D, and E.
#
# Prerequisites:
#   - Library installed: pip install -e .  (from new-code/)
#   - For Appendix E (ACC vs ACC++): pre-traced graphs in data/
#     (see README.md for download instructions)
#
# Usage:
#   bash experiments/run_paper_figures.sh
#
# Set DEVICE to "mps" (Apple Silicon), "cuda" (GPU), or "cpu".

set -euo pipefail

DEVICE="${1:-mps}"

echo "=== Generating paper figures (device=$DEVICE) ==="
echo ""

# -----------------------------------------------------------------------
# Appendix B — Condition number heatmaps (Figures 4–6)
#
# Produces 2 PDFs per model:
#   figures/condition-numbers/{model}_W_Q_condition_numbers.pdf
#   figures/condition-numbers/{model}_W_K_condition_numbers.pdf
# -----------------------------------------------------------------------
echo ">>> Appendix B: Condition number heatmaps"

echo "  gpt2-small"
python experiments/plot_condition_numbers.py -m gpt2-small -d "$DEVICE"

echo "  pythia-160m"
python experiments/plot_condition_numbers.py -m EleutherAI/pythia-160m -d "$DEVICE"

echo "  gemma-2-2b"
python experiments/plot_condition_numbers.py -m gemma-2-2b -d "$DEVICE"

echo ""

# -----------------------------------------------------------------------
# Appendix D — Finding τ: ECDF of attention_weight × context_size (Figures 8–10)
#
# Produces 1 PDF per model/task:
#   figures/attention-scores-distribution/{model}_{task}_{n}.pdf
#
# Note: GT is skipped for gemma-2-2b (not in the paper).
# -----------------------------------------------------------------------
echo ">>> Appendix D: Finding tau (ECDF plots)"

echo "  gpt2-small"
python experiments/plot_tau_ecdf.py -m gpt2-small -d "$DEVICE"

echo "  pythia-160m"
python experiments/plot_tau_ecdf.py -m EleutherAI/pythia-160m -d "$DEVICE"

echo "  gemma-2-2b"
python experiments/plot_tau_ecdf.py -m gemma-2-2b -d "$DEVICE"

echo ""

# -----------------------------------------------------------------------
# Appendix E — ACC vs ACC++ comparison (Figures 11–13)
#
# Produces 15 PDFs:
#   figures/acc_accpp_comparison/n_nodes_{task}.pdf         (3 plots)
#   figures/acc_accpp_comparison/n_edges_{task}.pdf         (3 plots)
#   figures/acc_accpp_comparison/{task}_ecdf_in-degree_{model}.pdf  (9 plots)
#
# Requires pre-traced graphs in data/traced_graphs_acc/ and
# data/traced_graphs_accpp/. See README.md for download instructions.
# -----------------------------------------------------------------------
ACC_DIR="data/traced_graphs_acc"
ACCPP_DIR="data/traced_graphs"

if [ -d "$ACC_DIR" ] && [ -d "$ACCPP_DIR" ]; then
    echo ">>> Appendix E: ACC vs ACC++ comparison"
    python experiments/compare_acc_accpp.py \
        --acc_dir "$ACC_DIR" \
        --accpp_dir "$ACCPP_DIR"
else
    echo ">>> Appendix E: SKIPPED (data not found)"
    echo "   Expected: $ACC_DIR/ and $ACCPP_DIR/"
    echo "   Download pre-traced graphs first (see README.md)."
fi

echo ""
echo "=== All figures complete ==="
echo ""
echo "Output directories:"
echo "  figures/condition-numbers/              (Appendix B)"
echo "  figures/attention-scores-distribution/  (Appendix D)"
echo "  figures/acc_accpp_comparison/           (Appendix E)"
