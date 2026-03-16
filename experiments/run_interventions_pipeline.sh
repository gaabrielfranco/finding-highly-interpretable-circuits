#!/usr/bin/env bash
# Appendix E — Causal Interventions Pipeline
#
# Runs the full pipeline from per-prompt graphs to circuit comparison plots.
# Each step depends on the outputs of the previous one.
#
# Prerequisites:
#   - Library installed: pip install -e .  (from new-code/)
#   - Per-prompt traced graphs in data/traced_graphs/{model}/{task}/
#     (produced by trace.py or downloaded from HuggingFace)
#
# Usage:
#   bash experiments/run_interventions_pipeline.sh [device]
#
# Set DEVICE to "mps" (Apple Silicon), "cuda" (GPU), or "cpu".

set -euo pipefail

DEVICE="${1:-mps}"

MODELS=("gpt2-small" "pythia-160m" "gemma-2-2b")
TASKS_IOI_GT_GP=("ioi" "gt" "gp")
# Gemma only has IOI and GP (no GT in the paper)
TASKS_GEMMA=("ioi" "gp")

# Number of prompts per task (Appendix E uses n=256 for IOI/GT, n=100 for GP)
# Note: Section 3 (main paper) uses a separate balanced IOI dataset with n=3000,
# but the interventions pipeline here uses the standard n=256.
N_IOI=256
N_GT=256
N_GP=100

get_n_prompts() {
    local task=$1
    case $task in
        ioi) echo $N_IOI ;;
        gt)  echo $N_GT ;;
        gp)  echo $N_GP ;;
    esac
}

get_tasks() {
    local model=$1
    case $model in
        gemma-2-2b) echo "${TASKS_GEMMA[@]}" ;;
        *)          echo "${TASKS_IOI_GT_GP[@]}" ;;
    esac
}

echo "=== Appendix E: Causal Interventions Pipeline (device=$DEVICE) ==="
echo ""

# -----------------------------------------------------------------------
# Step 2: Graph Unification — unify_graphs.py
#
# Combines per-prompt graphml files into unified graphs.
# Only threshold 0.01 is needed (used by Step 3b). The 0.0 (unthresholded)
# combined graph is always saved as a byproduct.
# Only runs for model/task combos used by Step 3b:
#   GPT-2 small (ioi, gt, gp) + Pythia (ioi).
#
# Input:  data/traced_graphs/{model}/{task}/*.graphml
# Output: data/combined_graphs/{model}/{task}/*_combined_{0.0,0.01}.graphml
# -----------------------------------------------------------------------
echo ">>> Step 2: Graph Unification (threshold=0.01 only)"

for task in ioi gt gp; do
    n=$(get_n_prompts "$task")
    echo "  gpt2-small / $task (n=$n)"
    python experiments/unify_graphs.py -m gpt2-small -t "$task" -n "$n" --thresholds 0.01
done

echo "  pythia-160m / ioi (n=$N_IOI)"
python experiments/unify_graphs.py -m pythia-160m -t ioi -n "$N_IOI" --thresholds 0.01

echo ""

# -----------------------------------------------------------------------
# Step 3: Causal Interventions — interventions.py
#
# 3a: Curated edges (default mode)
#   Runs local interventions on hardcoded curated edges per model/task.
#   Input:  data/traced_graphs/{model}/{task}/ + model weights
#   Output: data/intervention_data/{model}_{task}.parquet
#
# 3b: All edges (-ig flag)
#   Runs interventions on ALL edges from unified graphs.
#   Input:  data/combined_graphs/{model}/{task}/ + data/traced_graphs/ + model weights
#   Output: data/interventions_graph_{model}_{task}_n{N}_{seed}_combined_0.01.graphml
# -----------------------------------------------------------------------

# Model name mapping for CLI (pythia needs full name for HookedTransformer)
get_model_cli_name() {
    local model=$1
    case $model in
        pythia-160m) echo "EleutherAI/pythia-160m" ;;
        *)           echo "$model" ;;
    esac
}

# Step 3a runs only for IOI — the curated edges for GT/GP were from ACC and
# some do not exist in ACC++ graphs (see paper Fig. 19).
echo ">>> Step 3a: Curated Edge Interventions (IOI only)"

for model in "${MODELS[@]}"; do
    cli_name=$(get_model_cli_name "$model")
    n=$N_IOI
    echo "  $model / ioi (n=$n)"
    python experiments/interventions.py -m "$cli_name" -t ioi -n "$n" -d "$DEVICE"
done

echo ""

# Step 3b runs for: GPT-2 small (ioi, gt, gp) + Pythia (ioi only). No Gemma.
echo ">>> Step 3b: All Edge Interventions (-ig)"

for task in ioi gt gp; do
    n=$(get_n_prompts "$task")
    echo "  gpt2-small / $task (n=$n)"
    python experiments/interventions.py -m gpt2-small -t "$task" -n "$n" -d "$DEVICE" -ig
done

echo "  pythia-160m / ioi (n=$N_IOI)"
python experiments/interventions.py -m EleutherAI/pythia-160m -t ioi -n "$N_IOI" -d "$DEVICE" -ig

# -----------------------------------------------------------------------
# Step 4: Intervention Graph Pruning — prune_intervention_graphs.py
#
# Prunes intervention graphs by prompt count and effect size thresholds.
# Same model/task combos as Step 3b: GPT-2 small (ioi, gt, gp) + Pythia (ioi).
#
# Input:  data/interventions_graph_{model}_{task}_n{N}_{seed}_combined_0.01.graphml
# Output: data/combined_graphs_intervention/{model}/{task}/*.graphml
# -----------------------------------------------------------------------
echo ">>> Step 4: Intervention Graph Pruning"

for task in ioi gt gp; do
    n=$(get_n_prompts "$task")
    echo "  gpt2-small / $task (n=$n)"
    python experiments/prune_intervention_graphs.py -m gpt2-small -t "$task" -n "$n"
done

echo "  pythia-160m / ioi (n=$N_IOI)"
python experiments/prune_intervention_graphs.py -m pythia-160m -t ioi -n "$N_IOI"

echo ""

# -----------------------------------------------------------------------
# Step 5: Circuit Comparison — circuit_comparison.py
#
# Compares ACC++ circuits against published baselines (Path Patching,
# ACDC, EAP, EAP-IG, Edge Pruning). Computes precision/recall/F1.
# Uses -i flag to load intervention-pruned graphs from Step 4.
# -th is the threshold at which to extract the "best" circuit for
# heatmaps and barplot CSV. Set these per model/task from paper results.
#
# Input:  data/combined_graphs_intervention/{model}/{task}/ + hardcoded baselines
# Output: data/circuit_comparison/{model}_{task}_*.csv + line plots + heatmaps
# -----------------------------------------------------------------------
echo ">>> Step 5: Circuit Comparison"

# Best thresholds per model/task (from paper results — inspect line plots to choose)
TH_GPT2_IOI=0.1
TH_GPT2_GT=0.1
TH_GPT2_GP=0.1
TH_PYTHIA_IOI=0.1

echo "  gpt2-small / ioi (th=$TH_GPT2_IOI)"
python experiments/circuit_comparison.py -m gpt2-small -t ioi -th "$TH_GPT2_IOI" -i

echo "  gpt2-small / gt (th=$TH_GPT2_GT)"
python experiments/circuit_comparison.py -m gpt2-small -t gt -th "$TH_GPT2_GT" -i

echo "  gpt2-small / gp (th=$TH_GPT2_GP)"
python experiments/circuit_comparison.py -m gpt2-small -t gp -th "$TH_GPT2_GP" -i

echo "  pythia-160m / ioi (th=$TH_PYTHIA_IOI)"
python experiments/circuit_comparison.py -m pythia-160m -t ioi -th "$TH_PYTHIA_IOI" -i

echo ""

# -----------------------------------------------------------------------
# Step 6: Circuit Comparison Barplot — plot_circuit_comparison.py
#
# Summary barplot from circuit comparison CSVs produced by Step 5.
# Input:  data/circuit_comparison/*.csv
# Output: figures/circuit_comparison/circuit_comparison.pdf
# -----------------------------------------------------------------------
echo ">>> Step 6: Circuit Comparison Barplot"
python experiments/plot_circuit_comparison.py

echo ""

# -----------------------------------------------------------------------
# Step 7: Intervention Plots — plot_interventions.py
#
# Violin/bar plots from curated edge intervention data (Step 3a).
# Input:  data/intervention_data/*.parquet
# Output: figures/interventions/*.pdf
# -----------------------------------------------------------------------
echo ">>> Step 7: Intervention Plots"
python experiments/plot_interventions.py

echo ""
echo "=== Pipeline complete ==="
