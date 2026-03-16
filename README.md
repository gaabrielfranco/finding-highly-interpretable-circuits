# Finding Highly Interpretable Prompt-Specific Circuits in Language Models

Experiment suite for **ACC++**, a circuit tracing algorithm for mechanistic interpretability of transformer attention heads. ACC++ decomposes attention head firings into upstream contributions using SVD of the bilinear form $W_Q W_K^T$, producing per-prompt circuit graphs that reveal how information flows through the model.

Paper: *"Finding Highly Interpretable Prompt-Specific Circuits in Language Models"*

## Installation

Requires Python 3.10.

```bash
git clone https://github.com/gaabrielfranco/finding-highly-interpretable-circuits.git
cd finding-highly-interpretable-circuits

# Install everything (paper experiments + vendored accpp_tracer library)
pip install -e .
```

For exact numerical reproducibility of the paper results, use the pinned environment:

```bash
pip install -r requirements.txt
```

## ACC++ Tracing Library

The core tracing algorithm is available as a standalone library: **[accpp-tracer](https://github.com/gaabrielfranco/accpp-tracer)**. If you want to trace circuits on your own prompts without reproducing the paper, use the library directly.

A pinned version of `accpp-tracer` is vendored in `lib/accpp_tracer/` and installed automatically with `pip install -e .`.

## Data

Pre-traced circuit graphs and experiment data are hosted on HuggingFace:

```bash
# Download pre-traced graphs (instructions will be added when the HF dataset is published)
```

## Reproducing Paper Results

All experiment scripts are in `experiments/`. Run them from the repository root.

### Step 1: Trace circuits

Trace per-prompt circuit graphs for all model-task combinations:

```bash
# Run all paper configurations
bash experiments/run_tracing.sh

# Or trace individually:
python experiments/trace.py -m gpt2-small -t ioi -n 256 -s 0 -d mps
python experiments/trace.py -m gpt2-small -t gt -n 256 -s 0 -d mps
python experiments/trace.py -m gpt2-small -t gp -s 0 -d mps
```

For large datasets, use batched tracing:

```bash
# Process in batches of 32 (cluster-friendly)
python experiments/trace.py -m gpt2-small -t ioi -n 3000 --batch_size 32 -b 0 -d cuda
python experiments/trace.py -m gpt2-small -t ioi -n 3000 --batch_size 32 -b 1 -d cuda
# ...
```

### Step 2: Appendix B -- Condition number heatmaps

```bash
python experiments/plot_condition_numbers.py -m gpt2-small -d mps
python experiments/plot_condition_numbers.py -m EleutherAI/pythia-160m -d mps
python experiments/plot_condition_numbers.py -m gemma-2-2b -d mps
```

Output: `figures/condition-numbers/{model}_W_Q_condition_numbers.pdf` and `_W_K_condition_numbers.pdf`.

### Step 3: Appendix D -- Finding tau

```bash
python experiments/plot_tau_ecdf.py -m gpt2-small -d mps
python experiments/plot_tau_ecdf.py -m EleutherAI/pythia-160m -d mps
python experiments/plot_tau_ecdf.py -m gemma-2-2b -d mps
```

Output: `figures/attention-scores-distribution/{model}_{task}_{n}.pdf`.

### Step 4: Appendix E -- ACC vs ACC++ comparison

Requires pre-traced graphs from both ACC and ACC++ (see [Data](#data)).

```bash
python experiments/compare_acc_accpp.py \
    --acc_dir data/traced_graphs_acc \
    --accpp_dir data/traced_graphs_accpp
```

Output: `figures/acc_accpp_comparison/` (node/edge count plots + in-degree ECDF).

### Step 5: Appendix E -- Causal interventions pipeline

The full pipeline runs Steps 2-7 sequentially. It requires pre-traced per-prompt graphs from Step 1.

```bash
# Run the full pipeline (default device: mps)
bash experiments/run_interventions_pipeline.sh

# Or specify device
bash experiments/run_interventions_pipeline.sh cuda
```

You can also run individual steps:

```bash
# Step 2: Unify per-prompt graphs at threshold 0.01
python experiments/unify_graphs.py -m gpt2-small -t ioi -n 256 --thresholds 0.01

# Step 3a: Curated edge interventions (IOI only)
python experiments/interventions.py -m gpt2-small -t ioi -n 256 -d mps

# Step 3b: All edge interventions (requires unified graphs from Step 2)
python experiments/interventions.py -m gpt2-small -t ioi -n 256 -d mps -ig

# Step 4: Prune intervention graphs
python experiments/prune_intervention_graphs.py -m gpt2-small -t ioi -n 256

# Step 5: Circuit comparison (P/R/F1 vs baselines)
python experiments/circuit_comparison.py -m gpt2-small -t ioi -th 0.1 -i

# Step 6: Circuit comparison barplot
python experiments/plot_circuit_comparison.py

# Step 7: Intervention violin/bar plots
python experiments/plot_interventions.py
```

Output directories:
- `data/combined_graphs/` -- unified graphs (Step 2)
- `data/intervention_data/` -- curated edge intervention parquets (Step 3a)
- `data/combined_graphs_intervention/` -- pruned intervention graphs (Step 4)
- `data/circuit_comparison/` -- P/R/F1 CSVs (Step 5)
- `figures/circuit_comparison/` -- barplot PDF (Step 6)
- `figures/interventions/` -- violin/bar plot PDFs (Step 7)

### Section 3 / Appendix F -- Clustering and signal analysis

This pipeline analyzes 3,000 balanced IOI prompts per model to show that circuits
are prompt-specific and organize into template-based families.

**Full pipeline** (Steps 2-7, assuming traced graphs already exist):

```bash
bash experiments/run_clustering_pipeline.sh mps
```

**Step-by-step**:

```bash
# Step 1: Trace balanced IOI circuits (3,000 prompts x 3 models, expensive)
bash experiments/run_tracing_balanced.sh mps

# Step 2: Extract component vectors from traced graphs
python experiments/process_graphs.py -m gpt2-small
python experiments/process_graphs.py -m pythia-160m
python experiments/process_graphs.py -m gemma-2-2b

# Step 3: Clustermaps and distributions (Figures 1, 22-23)
python experiments/plot_clustering.py -m gpt2-small
python experiments/plot_clustering.py -m pythia-160m
python experiments/plot_clustering.py -m gemma-2-2b
python experiments/plot_clustering.py --combined

# Step 4: Find representative prompts per template
python experiments/find_representatives.py -m gpt2-small
python experiments/find_representatives.py -m pythia-160m
python experiments/find_representatives.py -m gemma-2-2b

# Step 5: Extract per-edge signal vectors to H5 (GPU recommended)
python experiments/extract_signals.py -m gpt2-small -d mps --batch_size 8
python experiments/extract_signals.py -m EleutherAI/pythia-160m -d mps --batch_size 8
python experiments/extract_signals.py -m gemma-2-2b -d mps --batch_size 8

# Step 6: Signal comparison heatmaps (Figure 2)
python experiments/plot_signals.py -m gpt2-small -p1 129 -p2 1613 -l1 BABA -l2 ABBA
python experiments/plot_signals.py --combined

# Step 7: LaTeX tables (Tables 3-5, printed to stdout)
python experiments/generate_tables.py
```

Note: Step 6 prompt IDs come from Step 4's JSON output. The master shell script
(`run_clustering_pipeline.sh`) reads IDs from JSON automatically. The example above
uses hardcoded IDs from the original traces.

Output directories:
- `data/clustering/{model}/` -- processed parquets, H5 signals, NPZ files
- `figures/clustering/` -- clustermaps, signal heatmaps, representative lists

### Autointerpretation pipeline

This pipeline takes the per-edge signal vectors produced in the clustering pipeline (Step 5 above)
and generates natural-language interpretations for each circuit edge, validates them
with a judge LLM, and computes quantitative metrics.

There are two tracks: a **quantitative track** (Stages 0-4, all signals) and a
**qualitative track** (Stage 5, representative prompts only).

**Prerequisites** for all stages:
- Signal H5 files from the clustering pipeline:
  `data/clustering/{model}/signals_balanced_{model}_not-norm.h5`
- GPU server with sufficient VRAM (Stages 0-3 require GPU; Stage 4 is CPU-only)
- Stage 2 and 3 require [vLLM](https://docs.vllm.ai) installed:
  `pip install vllm`

#### Quantitative track (Stages 0-4)

**Full pipeline** (one model at a time):

```bash
# Run from the repository root
bash experiments/autointerp/run_autointerp_pipeline.sh gpt2-small
bash experiments/autointerp/run_autointerp_pipeline.sh EleutherAI/pythia-160m
bash experiments/autointerp/run_autointerp_pipeline.sh gemma-2-2b
```

**Step-by-step**:

```bash
# Stage 0: Collect Pile activations (one-time per model, ~6h on GPU)
python experiments/autointerp/collect_pile_activations.py \
    --model gpt2-small --device cuda --output_dir data/autointerp

# Stage 1: Extract top-40 and random-40 activations per signal (per layer)
python experiments/autointerp/extract_top_activations.py \
    --model gpt2-small --task ioi-balanced --layer 5 \
    --signals_file data/clustering/gpt2-small/signals_balanced_gpt2-small_not-norm.h5 \
    --pile_activations data/autointerp/pile_activations_gpt2-small.h5 \
    --output_dir data/autointerp --device cuda

# Stage 2: Generate interpretations via DeepSeek-R1 (per layer, vLLM)
python experiments/autointerp/interpret_signals.py \
    --model gpt2-small --layer 5 \
    --start 0 --end 999999 \
    --activations_file data/autointerp/activations_5_ioi-balanced_gpt2-small.h5 \
    --output_dir data/autointerp/shards/ --device cuda

# Stage 2b: Merge shard files back into the main activation H5
python experiments/autointerp/merge_shards.py \
    --model gpt2-small --task ioi-balanced \
    --layers 1 2 3 4 5 6 7 8 9 10 11 \
    --activations_dir data/autointerp/ \
    --shards_dir data/autointerp/shards/

# Stage 3: Score interpretations via Gemma-3-27b judge (per layer, vLLM)
python experiments/autointerp/score_interpretations.py \
    --model gpt2-small --layer 5 \
    --activations_file data/autointerp/activations_5_ioi-balanced_gpt2-small.h5 \
    --device cuda

# Stage 4: Compute metrics + generate tables and figures (CPU)
python experiments/autointerp/compute_metrics.py \
    --models gpt2-small pythia-160m gemma-2-2b \
    --task ioi-balanced \
    --activations_dir data/autointerp/ \
    --output_dir data/autointerp/results/
```

Output:
- `data/autointerp/activations_{layer}_{task}_{model}.h5` -- top-K indices + interpretations + labels
- `data/autointerp/results/table_interpretability.tex` -- Table 1 (median + IQR)
- `data/autointerp/results/fraction_significant_fdr_5pct_{model}.pdf` -- per-layer FDR significance
- `data/autointerp/results/fraction_significant_fdr_per_model.pdf` -- aggregate FDR barplot
- `data/autointerp/results/median_metrics_by_layer_{model}.pdf` -- per-layer accuracy/precision/recall
- `data/autointerp/results/fdr_per_model.csv` -- per-model FDR stats with Wilson CIs
- `data/autointerp/results/metrics.csv` -- raw per-signal metrics

#### Qualitative track (Stage 5)

Produces annotated circuit viewers for a small set of representative prompts using
the Gemini batch API. Requires representative prompt IDs from the clustering pipeline
(`find_representatives.py`).

```bash
# Step 5a: Build Gemini batch request + edge examples side output
python experiments/autointerp/interpret_representatives.py build-request \
    --model gpt2-small --task ioi-balanced \
    --representatives 129 1613 1372 \
    --signals_file data/clustering/gpt2-small/signals_balanced_gpt2-small_not-norm.h5 \
    --activations_dir data/autointerp/ \
    --output data/autointerp/prompts_representatives.jsonl

# Step 5b: Submit to Gemini batch API
export GEMINI_API_KEY=your_key_here
python experiments/autointerp/interpret_representatives.py submit-batch \
    --request_file data/autointerp/prompts_representatives.jsonl

# Step 5c: Check status (repeat until SUCCEEDED)
python experiments/autointerp/interpret_representatives.py check-batch \
    --job_id batches/abc123...

# Step 5d: Download response
python experiments/autointerp/interpret_representatives.py download-batch \
    --job_id batches/abc123... \
    --output_file data/autointerp/prompts_representatives_response.jsonl

# Step 5e: Parse Gemini response
python experiments/autointerp/interpret_representatives.py parse-response \
    --response_file data/autointerp/prompts_representatives_response.jsonl \
    --output_file data/autointerp/interpretations_representatives.json

# Step 5f: Annotate circuit graphs + apply Cytoscape layout
python experiments/autointerp/annotate_graphs.py \
    --model gpt2-small --task ioi-balanced \
    --representatives 129 1613 1372 \
    --interpretations_file data/autointerp/interpretations_representatives.json \
    --edge_examples_file data/autointerp/prompts_representatives_edge_examples.json \
    --signals_file data/clustering/gpt2-small/signals_balanced_gpt2-small_not-norm.h5 \
    --graphs_dir data/traced_graphs/gpt2-small/ioi-balanced/ \
    --output_dir data/autointerp/graphs_interp/

# Step 5g: Build interactive Bokeh HTML viewer
python experiments/autointerp/view_circuit.py \
    data/autointerp/graphs_interp/gpt2-small_ioi-balanced_n3000_129_0_dynamic_ig.graphml \
    data/autointerp/graphs_interp/baba_representative.html \
    --tokens data/autointerp/graphs_interp/gpt2-small_ioi-balanced_n3000_129_0_dynamic_ig-tokens.json \
    --examples data/autointerp/graphs_interp/gpt2-small_ioi-balanced_n3000_129_0_dynamic_ig_edge_examples.json \
    --xaxis-labels tokens
```

## Supported Models

| Model | Tasks | Notes |
|-------|-------|-------|
| `gpt2-small` | IOI, GT, GP | Bias, no RoPE |
| `EleutherAI/pythia-160m` | IOI, GT, GP | RoPE, no bias |
| `gemma-2-2b` | IOI, GP | RoPE, GQA, softcapping |

## Project Structure

```
finding-highly-interpretable-circuits/
├── lib/accpp_tracer/             # Vendored ACC++ tracing library
│   └── src/accpp_tracer/
│       ├── circuit.py            # Tracer class (trace / trace_from_cache)
│       ├── tracing.py            # trace_firing (per-firing decomposition)
│       ├── decomposition.py      # Omega SVD + weight pseudoinverses
│       ├── attribution.py        # IG softmax attribution
│       ├── rope.py               # RoPE rotation matrices
│       ├── models.py             # Model configuration
│       ├── signals.py            # Per-edge signal extraction
│       ├── datasets/             # IOI, GT, GP, Facts datasets
│       └── graphs/               # Unification, pruning, visualization
├── experiments/                  # Paper reproduction scripts
│   ├── trace.py                  # Per-prompt circuit tracing
│   ├── plot_condition_numbers.py # Appendix B
│   ├── plot_tau_ecdf.py          # Appendix D
│   ├── compare_acc_accpp.py      # Appendix E: ACC vs ACC++
│   ├── unify_graphs.py           # Appendix E: graph unification
│   ├── interventions.py          # Appendix E: causal interventions
│   ├── prune_intervention_graphs.py
│   ├── circuit_comparison.py     # Appendix E: P/R/F1 vs baselines
│   ├── plot_circuit_comparison.py
│   ├── plot_interventions.py
│   ├── process_graphs.py         # Section 3: component extraction
│   ├── plot_clustering.py        # Section 3: clustermaps (Figure 1)
│   ├── find_representatives.py   # Section 3: template representatives
│   ├── extract_signals.py        # Section 3: signal extraction to H5
│   ├── plot_signals.py           # Section 3: signal heatmaps (Figure 2)
│   ├── generate_tables.py        # Section 3: LaTeX tables (Tables 3-5)
│   ├── run_*.sh                  # Pipeline orchestration scripts
│   └── autointerp/               # Autointerpretation pipeline
│       ├── collect_pile_activations.py  # Stage 0
│       ├── extract_top_activations.py   # Stage 1
│       ├── interpret_signals.py         # Stage 2
│       ├── merge_shards.py              # Stage 2b
│       ├── score_interpretations.py     # Stage 3
│       ├── compute_metrics.py           # Stage 4
│       ├── interpret_representatives.py # Stage 5a
│       ├── annotate_graphs.py           # Stage 5b
│       ├── view_circuit.py              # Stage 5c
│       ├── prompts.py                   # Shared: LLM prompt templates
│       ├── h5_utils.py                  # Shared: H5 I/O helpers
│       └── run_*.sh                     # Pipeline orchestration
├── pyproject.toml
├── requirements.txt              # Pinned env for reproducibility
└── README.md
```

## Citation

```bibtex
@article{franco2026finding,
  title={Finding Highly Interpretable Prompt-Specific Circuits in Language Models},
  author={Franco, Gabriel and Tassis, Lucas M and Rohr, Azalea and Crovella, Mark},
  journal={arXiv preprint arXiv:2602.13483},
  year={2026}
}
```
