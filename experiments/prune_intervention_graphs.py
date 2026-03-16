"""Prune intervention graphs by prompt count and effect size thresholds.

Takes combined intervention graphs from Step 3b, prunes edges based on:
- Minimum prompt appearances (n_prompts_thresh)
- Minimum absolute mean logit_diff (thresholded from 0.0 to 1.0)

Then removes isolated nodes and disconnected components, normalizes edge
weights, and saves pruned graphs for circuit comparison (Step 5).

This is Step 4 of the Appendix E pipeline.

Usage:
    python experiments/prune_intervention_graphs.py -m gpt2-small -t ioi -n 256
    python experiments/prune_intervention_graphs.py -m pythia-160m -t ioi -n 256
"""

import argparse
import os
import sys

import networkx as nx
import numpy as np

# Add src/ to path so accpp_tracer is importable without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from accpp_tracer.graphs.pruning import remove_isolated_nodes, keep_connected_to_root


# Root node labels per task (must match Step 3b output).
ROOT_NODE_LABELS = {
    "ioi": "('IO-S direction', 'end')",
    "gp": "('Correct - Incorrect pronoum', 'end')",
    "gt": "('True YY - False YY', 'end')",
}

# Default n_prompts_thresh per (model, task)
N_PROMPTS_THRESH = {
    ("gpt2-small", "ioi"): 30,
    ("pythia-160m", "ioi"): 15,
    ("gpt2-small", "gp"): 15,
    ("gpt2-small", "gt"): 15,
}

# Logit diff thresholds for pruning
LOGIT_DIFF_THRESHOLDS = list(np.arange(0, 1.05, 0.05))


def main():
    parser = argparse.ArgumentParser(
        description="Prune intervention graphs by prompt count and effect size (Step 4)"
    )
    parser.add_argument(
        "-m", "--model_name", choices=["gpt2-small", "pythia-160m"], required=True,
    )
    parser.add_argument(
        "-t", "--task", choices=["ioi", "gt", "gp"], required=True,
    )
    parser.add_argument(
        "-n", "--num_prompts", type=int, required=True,
        help="Number of prompts used in tracing (e.g. 256 for IOI/GT, 100 for GP)",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=0,
    )
    parser.add_argument(
        "--n_prompts_thresh", type=int, default=None,
        help="Min prompt appearances for edge retention (default: from N_PROMPTS_THRESH dict)",
    )
    args = parser.parse_args()

    model = args.model_name
    task = args.task
    n_prompts = args.num_prompts
    seed = args.seed

    ROOT_NODE = ROOT_NODE_LABELS[task]
    n_prompts_thresh = args.n_prompts_thresh or N_PROMPTS_THRESH[(model, task)]

    # Load the intervention graph from Step 3b
    input_path = f"data/interventions_graph_{model}_{task}_n{n_prompts}_{seed}_combined_0.01.graphml"
    print(f"Loading intervention graph: {input_path}")
    G: nx.MultiDiGraph = nx.read_graphml(input_path)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Parse serialized edge attributes
    for edge in G.edges(keys=True):
        G.edges[edge]["prompts_appeared"] = eval(G.edges[edge]["prompts_appeared"])
        if ROOT_NODE not in edge:
            G.edges[edge]["upstream_node"] = eval(G.edges[edge]["upstream_node"])
            G.edges[edge]["downstream_node"] = eval(G.edges[edge]["downstream_node"])
            G.edges[edge]["logit_diff"] = eval(G.edges[edge]["logit_diff"])

    # Output directory
    output_dir = f"data/combined_graphs_intervention/{model}/{task}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Pruning with n_prompts_thresh={n_prompts_thresh}")

    # Prune at each logit_diff threshold
    for th in LOGIT_DIFF_THRESHOLDS:
        th_round = np.round(th, 2)

        # Select edges to keep
        edges_keep = []
        for edge in G.edges(keys=True):
            if ROOT_NODE in edge:
                edges_keep.append(edge)
                continue
            if len(G.edges[edge]["prompts_appeared"]) >= n_prompts_thresh:
                if np.abs(np.mean(G.edges[edge]["logit_diff"])) >= th:
                    edges_keep.append(edge)

        # Create pruned subgraph
        G_pruned = nx.MultiDiGraph(
            nx.subgraph_view(G, filter_edge=lambda x, y, z: (x, y, z) in edges_keep)
        )

        # Remove isolated nodes and disconnected components
        remove_isolated_nodes(G_pruned)
        keep_connected_to_root(G_pruned, ROOT_NODE)

        # Serialize edge attributes back to str for graphml
        for edge in G_pruned.edges(keys=True):
            G_pruned.edges[edge]["prompts_appeared"] = str(G_pruned.edges[edge]["prompts_appeared"])
            if ROOT_NODE not in edge:
                try:
                    G_pruned.edges[edge]["upstream_node"] = str(G_pruned.edges[edge]["upstream_node"])
                    G_pruned.edges[edge]["downstream_node"] = str(G_pruned.edges[edge]["downstream_node"])
                    G_pruned.edges[edge]["logit_diff"] = str(G_pruned.edges[edge]["logit_diff"])
                except:
                    print(edge)
                    print(G_pruned.edges[edge])
                    break

        # Normalize edge weights per node by type (d/s)
        for node in G_pruned.nodes():
            total_contribution = {"d": 0, "s": 0}
            for edge in G_pruned.in_edges(node, data=True):
                total_contribution[edge[2]["type"]] += edge[2]["weight"]

            for edge in G_pruned.in_edges(node, keys=True):
                G_pruned.edges[edge]["norm_weight"] = G_pruned.edges[edge]["weight"] / total_contribution[G_pruned.edges[edge]["type"]]

        # Add abs(avg(logit_diff)) metric for visualization
        for edge in G_pruned.edges(keys=True):
            if ROOT_NODE not in edge:
                G_pruned.edges[edge]["abs_avg_logit_diff"] = np.abs(
                    np.mean(eval(G_pruned.edges[edge]["logit_diff"]))
                )
            else:
                G_pruned.edges[edge]["abs_avg_logit_diff"] = 2.  # Large value to show up

        output_path = os.path.join(
            output_dir, f"{model}_{task}_n{n_prompts}_combined_{th_round}.graphml"
        )
        nx.write_graphml(G_pruned, output_path)
        print(
            f"  th={th_round}: {G_pruned.number_of_nodes()} nodes, "
            f"{G_pruned.number_of_edges()} edges"
        )

    print(f"\nDone. Pruned graphs saved to {output_dir}/")


if __name__ == "__main__":
    main()
