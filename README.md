# ACC++ Circuit Tracer

A library and experiment suite for **ACC++**, a circuit tracing algorithm for mechanistic interpretability of transformer attention heads. ACC++ decomposes attention head firings into upstream contributions using SVD of the bilinear form W_Q W_K^T, producing per-prompt circuit graphs that reveal how information flows through the model.

From the paper: *"Finding Highly Interpretable Prompt-Specific Circuits in Language Models"*.

## Installation

Requires Python 3.10.

```bash
# Clone both repositories (library + paper experiments)
git clone https://github.com/<org>/accpp-tracer.git
git clone https://github.com/<org>/accpp-paper.git

# 1. Install the library
pip install -e accpp-tracer/

# 2. Install experiment dependencies
pip install -e accpp-paper/
```

For exact numerical reproducibility of the paper results, use the pinned environment:

```bash
# Install the library with pinned deps
pip install -r accpp-tracer/requirements.txt

# Install experiment-specific pinned deps
pip install -r accpp-paper/requirements.txt
```

## Quick Start

### Trace a single prompt (Level 3 API)

```python
import torch
from transformer_lens import HookedTransformer
from accpp_tracer import Tracer

torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
tracer = Tracer(model)

graph = tracer.trace(
    prompt="When Mary and John went to the store, John gave a drink to",
    answer_token=" Mary",
    wrong_token=" John",
)

print(f"Circuit: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
```

### Trace from a pre-computed cache (Level 2 API)

For batch processing (paper reproduction), use `trace_from_cache()`:

```python
from accpp_tracer import Tracer
from accpp_tracer.datasets import IOIDataset

model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
tracer = Tracer(model)

dataset = IOIDataset(
    model_family="gpt2", prompt_type="mixed", N=8,
    tokenizer=model.tokenizer, prepend_bos=False, seed=0, device="cpu",
)

logits, cache = model.run_with_cache(dataset.toks)

# Trace one prompt
prompt_id = 0
logit_dir = model.W_U[:, dataset.toks[prompt_id, dataset.word_idx["IO"][prompt_id]]] \
          - model.W_U[:, dataset.toks[prompt_id, dataset.word_idx["S1"][prompt_id]]]

graph = tracer.trace_from_cache(
    cache=cache,
    logit_direction=logit_dir,
    end_token_pos=dataset.word_idx["end"][prompt_id].item(),
    idx_to_token={i: model.tokenizer.decode(dataset.toks[prompt_id, i])
                  for i in range(dataset.word_idx["end"][prompt_id].item() + 1)},
    root_node=("IO-S direction", "to"),
    prompt_idx=prompt_id,
)
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
# Process in batches of 32 (qsub-friendly)
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

Requires pre-traced graphs from both ACC and ACC++ (see [Data](#data) below).

```bash
python experiments/compare_acc_accpp.py \
    --acc_dir data/traced_graphs_acc \
    --accpp_dir data/traced_graphs_accpp
```

Output: `figures/acc_accpp_comparison/` (node/edge count plots + in-degree ECDF).

### Step 5: Appendix E -- Causal interventions pipeline

The full pipeline runs Steps 2–7 sequentially. It requires pre-traced per-prompt graphs from Step 1.

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
- `data/combined_graphs/` — unified graphs (Step 2)
- `data/intervention_data/` — curated edge intervention parquets (Step 3a)
- `data/combined_graphs_intervention/` — pruned intervention graphs (Step 4)
- `data/circuit_comparison/` — P/R/F1 CSVs (Step 5)
- `figures/circuit_comparison/` — barplot PDF (Step 6)
- `figures/interventions/` — violin/bar plot PDFs (Step 7)

### Section 3 / Appendix F -- Clustering and signal analysis

This pipeline analyzes 3,000 balanced IOI prompts per model to show that circuits
are prompt-specific and organize into template-based families.

**Full pipeline** (Steps 2–7, assuming traced graphs already exist):

```bash
bash experiments/run_clustering_pipeline.sh mps
```

**Step-by-step**:

```bash
# Step 1: Trace balanced IOI circuits (3,000 prompts × 3 models, expensive)
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
- `data/clustering/{model}/` — processed parquets, H5 signals, NPZ files
- `figures/clustering/` — clustermaps, signal heatmaps, representative lists

### Part 3 -- Autointerpretation pipeline

This pipeline takes the per-edge signal vectors produced in Section 3 (Step 5 above)
and generates natural-language interpretations for each circuit edge, validates them
with a judge LLM, and computes quantitative metrics (Table 1, Figure 32).

There are two tracks: a **quantitative track** (Stages 0–4, all signals) and a
**qualitative track** (Stage 5, representative prompts only).

**Prerequisites** for all stages:
- Signal H5 files from the clustering pipeline:
  `data/clustering/{model}/signals_balanced_{model}_not-norm.h5`
- GPU server with sufficient VRAM (Stages 0–3 require GPU; Stage 4 is CPU-only)
- Stage 2 and 3 require [vLLM](https://docs.vllm.ai) installed:
  `pip install vllm`

#### Quantitative track (Stages 0–4)

**Full pipeline** (one model at a time):

```bash
# Run from new-code/experiments/
bash autointerp/run_autointerp_pipeline.sh gpt2-small
bash autointerp/run_autointerp_pipeline.sh EleutherAI/pythia-160m
bash autointerp/run_autointerp_pipeline.sh gemma-2-2b
```

**Step-by-step**:

```bash
# Stage 0: Collect Pile activations (one-time per model, ~6h on GPU)
# Output: data/autointerp/pile_activations_{model}.h5
python autointerp/collect_pile_activations.py \
    --model gpt2-small --device cuda --output_dir data/autointerp

# Stage 1: Extract top-40 and random-40 activations per signal (per layer)
# Output: data/autointerp/activations_{layer}_{task}_{model}.h5
python autointerp/extract_top_activations.py \
    --model gpt2-small --task ioi-balanced --layer 5 \
    --signals_file data/clustering/gpt2-small/signals_balanced_gpt2-small_not-norm.h5 \
    --pile_activations data/autointerp/pile_activations_gpt2-small.h5 \
    --output_dir data/autointerp --device cuda

# Stage 2: Generate interpretations via DeepSeek-R1 (per layer, vLLM)
# Output: shard file in data/autointerp/shards/
python autointerp/interpret_signals.py \
    --model gpt2-small --layer 5 \
    --start 0 --end 999999 \
    --activations_file data/autointerp/activations_5_ioi-balanced_gpt2-small.h5 \
    --output_dir data/autointerp/shards/ --device cuda

# Stage 2b: Merge shard files back into the main activation H5
python autointerp/merge_shards.py \
    --model gpt2-small --task ioi-balanced \
    --layers 1 2 3 4 5 6 7 8 9 10 11 \
    --activations_dir data/autointerp/ \
    --shards_dir data/autointerp/shards/

# Stage 3: Score interpretations via Gemma-3-27b judge (per layer, vLLM)
python autointerp/score_interpretations.py \
    --model gpt2-small --layer 5 \
    --activations_file data/autointerp/activations_5_ioi-balanced_gpt2-small.h5 \
    --device cuda

# Stage 4: Compute metrics + generate Table 1 and Figure 32 (CPU)
python autointerp/compute_metrics.py \
    --models gpt2-small pythia-160m gemma-2-2b \
    --task ioi-balanced \
    --activations_dir data/autointerp/ \
    --output_dir data/autointerp/results/
```

Output:
- `data/autointerp/activations_{layer}_{task}_{model}.h5` — top-K indices + interpretations + labels
- `data/autointerp/results/table1.tex` — Table 1
- `data/autointerp/results/figure32_fdr_significance.pdf` — Figure 32
- `data/autointerp/results/metrics.csv` — raw per-signal metrics

#### Qualitative track (Stage 5)

Produces annotated circuit viewers for a small set of representative prompts using
the Gemini batch API. Requires representative prompt IDs from the clustering pipeline
(`find_representatives.py`).

```bash
# Step 5a: Build Gemini batch request + edge examples side output
python autointerp/interpret_representatives.py build-request \
    --model gpt2-small --task ioi-balanced \
    --representatives 129 1613 1372 \
    --signals_file data/clustering/gpt2-small/signals_balanced_gpt2-small_not-norm.h5 \
    --activations_dir data/autointerp/ \
    --output data/autointerp/prompts_representatives.jsonl
# Also writes: data/autointerp/prompts_representatives_edge_examples.json

# (Send prompts_representatives.jsonl to the Gemini batch API manually)

# Step 5b: Parse Gemini response
python autointerp/interpret_representatives.py parse-response \
    --response_file data/autointerp/prompts_representatives_interpretation.jsonl \
    --output_file data/autointerp/interpretations_representatives.json

# Step 5c: Annotate circuit graphs + apply Cytoscape layout
python autointerp/annotate_graphs.py \
    --model gpt2-small --task ioi-balanced \
    --representatives 129 1613 1372 \
    --interpretations_file data/autointerp/interpretations_representatives.json \
    --edge_examples_file data/autointerp/prompts_representatives_edge_examples.json \
    --signals_file data/clustering/gpt2-small/signals_balanced_gpt2-small_not-norm.h5 \
    --graphs_dir data/traced_graphs/gpt2-small/ioi-balanced/ \
    --output_dir data/autointerp/graphs_interp/

# Step 5d: Build interactive Bokeh HTML viewer
python autointerp/view_circuit.py \
    data/autointerp/graphs_interp/gpt2-small_ioi-balanced_n3000_129_0_dynamic_ig.graphml \
    data/autointerp/graphs_interp/baba_representative.html \
    --tokens data/autointerp/graphs_interp/gpt2-small_ioi-balanced_n3000_129_0_dynamic_ig-tokens.json \
    --examples data/autointerp/graphs_interp/gpt2-small_ioi-balanced_n3000_129_0_dynamic_ig_edge_examples.json \
    --xaxis-labels tokens
```

Or run both tracks with the master scripts:

```bash
# Full quantitative pipeline for one model
bash autointerp/run_autointerp_pipeline.sh gpt2-small

# Qualitative pipeline (builds Gemini request, pauses for API call, then annotates)
bash autointerp/run_qualitative_pipeline.sh gpt2-small ioi-balanced 129 1613 1372
# After receiving Gemini response:
PARSE_RESPONSE=1 bash autointerp/run_qualitative_pipeline.sh gpt2-small ioi-balanced 129 1613 1372
```

## Data

Pre-traced circuit graphs and experiment data are hosted on HuggingFace:

```bash
# Download pre-traced graphs
# (instructions will be added when the HF dataset is published)
```

## Supported Models

| Model | Tasks | Notes |
|-------|-------|-------|
| `gpt2-small` | IOI, GT, GP | Bias, no RoPE |
| `EleutherAI/pythia-160m` | IOI, GT, GP | RoPE, no bias |
| `gemma-2-2b` | IOI, GP | RoPE, GQA, softcapping |

## Project Structure

```
new-code/
├── src/accpp_tracer/          # Pip-installable library
│   ├── circuit.py             # Tracer class (Level 2 & 3 API)
│   ├── tracing.py             # trace_firing (Level 1 API)
│   ├── decomposition.py       # Omega SVD + weight pseudoinverses
│   ├── attribution.py         # IG softmax attribution
│   ├── rope.py                # RoPE rotation matrices
│   ├── models.py              # Model configuration
│   ├── signals.py             # Per-edge signal extraction
│   ├── datasets/              # IOI, GT, GP, Facts datasets
│   └── graphs/                # Unification, pruning, visualization
├── experiments/               # Paper reproduction scripts
│   ├── trace.py               # Per-prompt circuit tracing
│   ├── plot_condition_numbers.py   # Appendix B: condition numbers
│   ├── plot_tau_ecdf.py            # Appendix D: finding tau
│   ├── compare_acc_accpp.py        # Appendix E: ACC vs ACC++
│   ├── unify_graphs.py             # Appendix E: graph unification
│   ├── interventions.py            # Appendix E: causal interventions
│   ├── prune_intervention_graphs.py # Appendix E: graph pruning
│   ├── circuit_comparison.py       # Appendix E: P/R/F1 vs baselines
│   ├── plot_circuit_comparison.py  # Appendix E: barplot
│   ├── plot_interventions.py       # Appendix E: violin/bar plots
│   ├── process_graphs.py           # Section 3: component extraction
│   ├── plot_clustering.py          # Section 3: clustermaps (Figure 1)
│   ├── find_representatives.py     # Section 3: template representatives
│   ├── extract_signals.py          # Section 3: signal extraction to H5
│   ├── plot_signals.py             # Section 3: signal heatmaps (Figure 2)
│   ├── generate_tables.py          # Section 3: LaTeX tables (Tables 3-5)
│   ├── run_tracing.sh              # Runs all tracing
│   ├── run_tracing_balanced.sh     # Runs balanced IOI tracing (3k prompts)
│   ├── run_paper_figures.sh        # Runs Appendix B, D, E comparison plots
│   ├── run_interventions_pipeline.sh # Runs interventions pipeline
│   ├── run_clustering_pipeline.sh  # Runs clustering pipeline (Section 3)
│   └── autointerp/                # Part 3: autointerpretation pipeline
│       ├── collect_pile_activations.py   # Stage 0: Pile residuals
│       ├── extract_top_activations.py    # Stage 1: top-K activations per signal
│       ├── interpret_signals.py          # Stage 2: explainer LLM (DeepSeek, vLLM)
│       ├── merge_shards.py               # Stage 2b: merge shard H5 files
│       ├── score_interpretations.py      # Stage 3: judge LLM (Gemma-3, vLLM)
│       ├── compute_metrics.py            # Stage 4: Table 1, Figure 32
│       ├── interpret_representatives.py  # Stage 5a: Gemini batch request/parse
│       ├── annotate_graphs.py            # Stage 5b: annotate GraphML edges
│       ├── view_circuit.py               # Stage 5c: interactive Bokeh HTML viewer
│       ├── prompts.py                    # Shared: LLM prompt templates
│       ├── h5_utils.py                   # Shared: H5 I/O helpers
│       ├── run_autointerp_pipeline.sh    # Master: quantitative track (Stages 0–4)
│       └── run_qualitative_pipeline.sh   # Master: qualitative track (Stage 5)
├── pyproject.toml
├── requirements.txt           # Pinned env for reproducibility
└── README.md
```

## Citation

```bibtex
@article{accpp2025,
  title={Finding Highly Interpretable Prompt-Specific Circuits in Language Models},
  author={TODO},
  year={2025}
}
```
