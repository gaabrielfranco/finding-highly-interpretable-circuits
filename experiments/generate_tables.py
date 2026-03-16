"""Generate LaTeX tables for the paper (Tables 3, 4, 5).

Produces three tables and saves them to a single .tex file:
- Table 3: Average node counts per template (from graphml files)
- Table 4: Average component counts per template (from parquet)
- Table 5: Normalized Jaccard distances per template (from parquet)

This is Step 7 of the Section 3 / Appendix F pipeline.

Usage:
    python experiments/generate_tables.py
    python experiments/generate_tables.py --data_dir data/clustering --graph_dir data/traced_graphs
    python experiments/generate_tables.py -o figures/clustering/tables.tex
"""

import argparse
import os

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import distance

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = ["gpt2-small", "pythia-160m", "gemma-2-2b"]
DISPLAY_NAMES = ["GPT-2 Small", "Pythia-160M", "Gemma-2 2B"]
NUM_TEMPLATES = 15


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def get_ci_string(values):
    """Format mean +/- 95% CI as LaTeX string.

    Args:
        values: Array-like of numeric values.

    Returns:
        String like "12.34 $\\pm$ 0.56".
    """
    avg = np.mean(values)
    sem = np.std(values, ddof=1) / np.sqrt(len(values))
    margin = 1.96 * sem
    return f"{avg:.2f} $\\pm$ {margin:.2f}"


# ---------------------------------------------------------------------------
# Table 3: Node counts
# ---------------------------------------------------------------------------


def generate_table_nodes(data_dir, graph_dir):
    """Table 3: Average node count per template from graphml files.

    Args:
        data_dir: Base directory for clustering data (parquet files).
        graph_dir: Base directory for traced graphs.

    Returns:
        LaTeX table string.
    """
    results_dict = {}
    for model in MODELS:
        results_dict[model] = {}

        df = pd.read_parquet(os.path.join(data_dir, model, "processed_components.parquet"))

        def _get_graph_stats(ids):
            sizes = []
            base = os.path.join(graph_dir, model, "ioi-balanced")
            for graph_id in ids:
                fname = f"{model}_ioi-balanced_n3000_{graph_id}_0_dynamic_ig.graphml"
                G = nx.read_graphml(os.path.join(base, fname))
                sizes.append(G.number_of_nodes())
            return get_ci_string(sizes)

        # ALL
        results_dict[model]["ALL"] = _get_graph_stats(df["id"].tolist())

        # ABBA / BABA
        for template in ["ABBA", "BABA"]:
            subset = df[df["high_level_template"] == template]
            results_dict[model][template] = _get_graph_stats(subset["id"].tolist())

        # Per low-level template (T1-T15 in paper, 0-indexed in code)
        for i in range(NUM_TEMPLATES):
            subset = df[df["low_level_template"] == i]
            results_dict[model][f"T{i + 1}"] = _get_graph_stats(subset["id"].tolist())

    df_results = pd.DataFrame(results_dict)
    return df_results.to_latex(
        escape=False,
        caption="Average node count for traced communication graphs across templates.",
        label="tab:app-graph-sizes",
        column_format="c" + "c" * len(MODELS),
        header=DISPLAY_NAMES,
    )


# ---------------------------------------------------------------------------
# Table 4: Component counts
# ---------------------------------------------------------------------------


def generate_table_components(data_dir):
    """Table 4: Average count of edge-SV pairs per template.

    Args:
        data_dir: Base directory for clustering data.

    Returns:
        LaTeX table string.
    """
    results_dict = {}
    for model in MODELS:
        results_dict[model] = {}
        df = pd.read_parquet(os.path.join(data_dir, model, "processed_components.parquet"))

        # ALL
        cmatrix = np.stack(df["sv_as_component"].to_numpy())
        results_dict[model]["ALL"] = get_ci_string(np.sum(cmatrix, axis=1))

        # ABBA / BABA
        for template in ["ABBA", "BABA"]:
            subset = df[df["high_level_template"] == template]
            cmatrix = np.stack(subset["sv_as_component"].to_numpy())
            results_dict[model][template] = get_ci_string(np.sum(cmatrix, axis=1))

        # Per low-level template
        for i in range(NUM_TEMPLATES):
            subset = df[df["low_level_template"] == i]
            cmatrix = np.stack(subset["sv_as_component"].to_numpy())
            results_dict[model][f"T{i + 1}"] = get_ci_string(np.sum(cmatrix, axis=1))

    df_results = pd.DataFrame(results_dict)
    return df_results.to_latex(
        escape=False,
        caption="Average count of edge-singular value pairs across templates.",
        label="tab:app-component-counts",
        column_format="c" + "c" * len(MODELS),
        header=DISPLAY_NAMES,
    )


# ---------------------------------------------------------------------------
# Table 5: Normalized Jaccard distances
# ---------------------------------------------------------------------------


def generate_table_distances(data_dir):
    """Table 5: Normalized mean Jaccard distance per template.

    Distances are normalized by the ALL-prompts mean distance per model.

    Args:
        data_dir: Base directory for clustering data.

    Returns:
        LaTeX table string.
    """
    results_dict = {}
    for model in MODELS:
        results_dict[model] = {}
        df = pd.read_parquet(os.path.join(data_dir, model, "processed_components.parquet"))

        # ALL
        cmatrix = np.stack(df["sv_as_component"].to_numpy())
        dmatrix = distance.pdist(cmatrix, metric="jaccard")
        all_mean = np.mean(dmatrix)
        results_dict[model]["ALL"] = all_mean / all_mean  # = 1.0

        # ABBA / BABA
        for template in ["ABBA", "BABA"]:
            subset = df[df["high_level_template"] == template]
            cmatrix = np.stack(subset["sv_as_component"].to_numpy())
            dmatrix = distance.pdist(cmatrix, metric="jaccard")
            results_dict[model][template] = np.mean(dmatrix) / all_mean

        # Per low-level template
        for i in range(NUM_TEMPLATES):
            subset = df[df["low_level_template"] == i]
            cmatrix = np.stack(subset["sv_as_component"].to_numpy())
            dmatrix = distance.pdist(cmatrix, metric="jaccard")
            results_dict[model][f"T{i + 1}"] = np.mean(dmatrix) / all_mean

    df_results = pd.DataFrame(results_dict)
    df_results.columns = DISPLAY_NAMES
    df_results.index.name = "Template"
    return df_results.to_latex(
        float_format="%.2f",
        caption="Normalized Jaccard Distances by Model and Template",
        label="tab:app-model_distances",
        column_format="c" + "c" * len(MODELS),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables (Tables 3, 4, 5)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/clustering",
        help="Base directory for processed parquet data. Default: data/clustering",
    )
    parser.add_argument(
        "--graph_dir", type=str, default="data/traced_graphs",
        help="Base directory for traced graphml files. Default: data/traced_graphs",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="figures/clustering/tables.tex",
        help="Output .tex file. Default: figures/clustering/tables.tex",
    )
    args = parser.parse_args()

    table_nodes = generate_table_nodes(args.data_dir, args.graph_dir)
    table_components = generate_table_components(args.data_dir)
    table_distances = generate_table_distances(args.data_dir)

    content = (
        "% Auto-generated by experiments/generate_tables.py\n"
        "% Tables 3, 4, 5 for Section 3 / Appendix F\n\n"
        "% Table 3: Node counts\n"
        f"{table_nodes}\n"
        "% Table 4: Component counts\n"
        f"{table_components}\n"
        "% Table 5: Normalized Jaccard distances\n"
        f"{table_distances}"
    )

    # Print to stdout
    print(content)

    # Save to file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(content)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
