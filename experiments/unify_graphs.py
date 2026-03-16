"""Combine per-prompt circuit graphs into unified graphs at multiple thresholds.

Reads all per-prompt graphml files for a model/task, combines them into a single
unified graph (accumulating edge weights, tracking prompt appearances), then saves
pruned versions at thresholds 0.0, 0.01, 0.05, 0.1, …, 0.9.

This is Step 2 of the Appendix E pipeline.

Usage:
    python experiments/unify_graphs.py -m gpt2-small -t ioi -n 256
    python experiments/unify_graphs.py -m gpt2-small -t ioi -n 256 \\
        --input_dir data/traced_graphs/gpt2-small/ioi \\
        --output_dir data/combined_graphs/gpt2-small/ioi

"""

import argparse
import glob
import os
import sys

import networkx as nx
import numpy as np

# Add src/ to path so accpp_tracer is importable without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from accpp_tracer.graphs.unification import combine_prompt_graphs, prune_by_frequency

SUPPORTED_MODELS = ["gpt2-small", "pythia-160m", "gemma-2-2b"]
SUPPORTED_TASKS = ["ioi", "gt", "gp"]

# Thresholds for pruning
THRESHOLDS = [0.01, 0.05] + list(np.arange(0.1, 1.0, 0.1))


def main():
    parser = argparse.ArgumentParser(
        description="Combine per-prompt graphs into unified graphs (Step 2)"
    )
    parser.add_argument(
        "-m", "--model_name", choices=SUPPORTED_MODELS, required=True,
    )
    parser.add_argument(
        "-t", "--task", choices=SUPPORTED_TASKS, required=True,
    )
    parser.add_argument(
        "-n", "--num_prompts", type=int, required=True,
        help="Total number of prompts used in tracing (e.g. 256 for IOI/GT, 100 for GP)",
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Directory with per-prompt graphml files (default: data/traced_graphs/{model}/{task})",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save combined graphs (default: data/combined_graphs/{model}/{task})",
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=None,
        help="Specific thresholds to compute (default: all). Example: --thresholds 0.01",
    )
    args = parser.parse_args()

    model_short = args.model_name
    input_dir = args.input_dir or f"data/traced_graphs/{model_short}/{args.task}"
    output_dir = args.output_dir or f"data/combined_graphs/{model_short}/{args.task}"

    # --- Load per-prompt graphs ---
    graphs_path = glob.glob(os.path.join(input_dir, "*.graphml"))
    if not graphs_path:
        print(f"No graphml files found in {input_dir}/")
        return

    print(f"Found {len(graphs_path)} graph files in {input_dir}/")

    # Build a list indexed by prompt_id
    n_prompts = args.num_prompts
    graphs = [None] * n_prompts
    for g_path in graphs_path:
        # Parse prompt_id from filename: {model}_{task}_n{N}_{prompt_id}_{seed}_{thresh}_{ordering}.graphml
        prompt_id = eval(g_path.split("/")[-1].split("_")[-4])
        graphs[prompt_id] = nx.read_graphml(g_path)

    # --- Combine graphs using the library ---
    G_combined = combine_prompt_graphs(graphs)
    n_graphs = len(graphs_path)  # number of non-None graphs

    print(f"Combined graph with {G_combined.number_of_nodes()} nodes and {G_combined.number_of_edges()} edges")

    # Sanity check: no edge appears in more prompts than graphs exist
    for _, _, data in G_combined.edges(data=True):
        assert len(data["prompts_appeared"]) <= n_graphs

    # Convert prompts_appeared lists to str for graphml serialization
    for edge in G_combined.edges:
        G_combined.edges[edge]["prompts_appeared"] = str(
            G_combined.edges[edge]["prompts_appeared"]
        )

    # --- Save combined graph and pruned versions ---
    os.makedirs(output_dir, exist_ok=True)

    # Construct base_name from first file
    base_name = "_".join(graphs_path[0].split("/")[-1].split("_")[:-2])

    # Save the combined graph with no threshold
    path_0 = os.path.join(output_dir, f"{base_name}_combined_0.0.graphml")
    nx.write_graphml(G_combined, path_0)

    # Prune at each threshold
    thresholds = args.thresholds if args.thresholds is not None else THRESHOLDS
    for thresh in thresholds:
        G_pruned = prune_by_frequency(G_combined, thresh, n_graphs)

        # Check for disconnected components
        n_components = len(list(nx.weakly_connected_components(G_pruned)))
        if n_components > 1:
            print(f"  Warning: {n_components} connected components at th={np.round(thresh, 2)}")

        # prompts_appeared is already str (inherited from G_combined copy)
        path = os.path.join(
            output_dir, f"{base_name}_combined_{np.round(thresh, 2)}.graphml"
        )
        nx.write_graphml(G_pruned, path)
        n_thresh = int(round(thresh * n_graphs, 0))
        print(
            f"Pruned graph (th={np.round(thresh, 2)}, n>={n_thresh}) with "
            f"{G_pruned.number_of_nodes()} nodes and {G_pruned.number_of_edges()} edges"
        )

    print(f"\nDone. Combined graphs saved to {output_dir}/")


if __name__ == "__main__":
    main()
