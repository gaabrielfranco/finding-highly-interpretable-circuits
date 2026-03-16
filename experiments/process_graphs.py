"""Extract component vectors from traced balanced IOI circuit graphs.

Reads per-prompt graphml files and extracts three component representations:
- head_as_component: attention heads + MLPs (one-hot encoded)
- edge_as_component: source→destination edges (one-hot encoded)
- sv_as_component: edge + singular vector pairs (one-hot encoded)

This is Step 2 of the Section 3 / Appendix F pipeline.

Usage:
    python experiments/process_graphs.py -m gpt2-small
    python experiments/process_graphs.py -m pythia-160m
    python experiments/process_graphs.py -m gemma-2-2b
"""

import argparse
import glob
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformer_lens import HookedTransformer

from accpp_tracer.datasets import IOIDataset

MODEL_CONFIGS = {
    "gpt2-small": {"tl_name": "gpt2-small", "family": "gpt2", "prepend_bos": False},
    "pythia-160m": {"tl_name": "EleutherAI/pythia-160m", "family": "pythia", "prepend_bos": False},
    "gemma-2-2b": {"tl_name": "gemma-2-2b", "family": "gemma", "prepend_bos": True},
}


def format_node_name(node, ios_node):
    """Format a graph node name into a standardized component label.

    Args:
        node: Raw node string from graphml (e.g., "('L0H3', 0, 3, 'token')").
        ios_node: The IO-S direction root node string.

    Returns:
        Formatted string: "(layer, head)" for AH, "(layer, 'MLP')" for MLP,
        or the raw node for the root.
    """
    if ios_node not in node:
        node_val = eval(node)
        if "MLP" in node:
            return f"({node_val[0]}, 'MLP')"
        else:
            return f"({node_val[0]}, {node_val[1]})"
    else:
        return node


def main():
    parser = argparse.ArgumentParser(description="Extract component vectors from traced graphs.")
    parser.add_argument(
        "-m", "--model", required=True, choices=list(MODEL_CONFIGS.keys()),
        help="Model name.",
    )
    parser.add_argument(
        "-i", "--input_dir", default=None,
        help="Input directory with graphml files. Default: data/traced_graphs/{model}/ioi-balanced/",
    )
    parser.add_argument(
        "-o", "--output_dir", default=None,
        help="Output directory for processed data. Default: data/clustering/{model}/",
    )
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    input_dir = args.input_dir or f"data/traced_graphs/{args.model}/ioi-balanced"
    output_dir = args.output_dir or f"data/clustering/{args.model}"
    os.makedirs(output_dir, exist_ok=True)

    # Get all graph files
    files = sorted(glob.glob(os.path.join(input_dir, "*.graphml")))
    if not files:
        print(f"No graphml files found in {input_dir}/")
        return
    print(f"Found {len(files)} graph files in {input_dir}/")

    # Load model (for tokenizer and cfg) and dataset
    print(f"Loading {cfg['tl_name']}...")
    model = HookedTransformer.from_pretrained(cfg["tl_name"], device="cpu")

    ioi_dataset = IOIDataset(
        model_family=cfg["family"],
        prompt_type="balanced",
        N=3000,
        tokenizer=model.tokenizer,
        prepend_bos=cfg["prepend_bos"],
        seed=0,
        device="cpu",
    )

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Build info dict
    info_dict = {
        "id": [],
        "text": [],
        "high_level_template": [],
        "low_level_template": [],
        "head_as_component": [],
        "edge_as_component": [],
        "sv_as_component": [],
    }

    for f in files:
        prompt_id = int(f.split("/")[-1].split(".")[0].split("_")[-4])
        G = nx.read_graphml(f, force_multigraph=True)
        ios_node = [n for n in G.nodes if "'IO-S direction'" in n][0]

        # Remove unnecessary components (Embedding, AH bias, AH offset)
        nodes_to_remove = []
        for node in list(G.nodes):
            if ("Embedding" in node) or ("AH bias" in node) or ("AH offset" in node):
                nodes_to_remove.append(node)
        G.remove_nodes_from(nodes_to_remove)

        # Heads as components
        head_G = G.copy()
        head_G.remove_node(ios_node)
        head_as_component = []
        for node in head_G.nodes:
            head_as_component.append(format_node_name(node, ios_node))

        # Edges as components and SVs as components
        edge_as_component = []
        sv_as_component = []
        for src, tgt, v in G.edges(data=True):
            src = format_node_name(src, ios_node)
            tgt = format_node_name(tgt, ios_node)
            # Edge as component
            if ios_node not in tgt:
                edge_as_component.append(str((src, tgt)))
            else:
                edge_as_component.append(str((src, "('IO-S direction', 'end')")))
            # SV as component
            if ios_node in tgt:
                sv_as_component.append(str((src, "('IO-S direction', 'end')", None)))
            else:
                for sv in eval(v["svs_used"]):
                    sv_as_component.append(str((src, tgt, sv)))
        edge_as_component = sorted(edge_as_component)
        sv_as_component = sorted(sv_as_component)

        # Save data to dict
        info_dict["id"].append(prompt_id)
        info_dict["text"].append(ioi_dataset.ioi_prompts[prompt_id]["text"])
        info_dict["high_level_template"].append(ioi_dataset.templates_by_prompt[prompt_id])
        info_dict["low_level_template"].append(ioi_dataset.ioi_prompts[prompt_id]["TEMPLATE_IDX"])
        info_dict["head_as_component"].append(head_as_component)
        info_dict["edge_as_component"].append(edge_as_component)
        info_dict["sv_as_component"].append(sv_as_component)

    print(f"Processed {len(info_dict['id'])} graphs.")

    # --- One-hot encode components ---

    # Heads: fixed vocabulary (all possible heads + MLPs)
    HEAD_COMPONENTS = (
        [f"({i}, 'MLP')" for i in range(n_layers)]
        + [f"({i}, {j})" for i in range(n_layers) for j in range(n_heads)]
    )
    one_hot = []
    for components in info_dict["head_as_component"]:
        one_hot.append(np.array([1 if c in components else 0 for c in HEAD_COMPONENTS]))
    info_dict["head_as_component"] = one_hot

    # Edges: data-driven vocabulary (only edges that appear in the dataset)
    edge_set = set()
    for components in info_dict["edge_as_component"]:
        edge_set.update(components)
    EDGE_COMPONENTS = sorted(list(edge_set), key=lambda x: ("IO-S direction" in str(x), x))
    one_hot = []
    for components in info_dict["edge_as_component"]:
        one_hot.append(np.array([1 if c in components else 0 for c in EDGE_COMPONENTS]))
    info_dict["edge_as_component"] = one_hot

    # SVs: data-driven vocabulary (only edge-SV pairs that appear in the dataset)
    sv_set = set()
    for components in info_dict["sv_as_component"]:
        sv_set.update(components)
    SV_COMPONENTS = sorted(list(sv_set), key=lambda x: ("IO-S direction" in str(x), x))
    one_hot = []
    for components in info_dict["sv_as_component"]:
        one_hot.append(np.array([1 if c in components else 0 for c in SV_COMPONENTS]))
    info_dict["sv_as_component"] = one_hot

    # --- Save ---
    info_df = pd.DataFrame(info_dict)
    info_df = info_df.sort_values(by="id").reset_index(drop=True)
    info_df.to_parquet(os.path.join(output_dir, "processed_components.parquet"))
    np.save(os.path.join(output_dir, "head_as_component_labels.npy"), np.array(HEAD_COMPONENTS))
    np.save(os.path.join(output_dir, "edge_as_component_labels.npy"), np.array(EDGE_COMPONENTS))
    np.save(os.path.join(output_dir, "sv_as_component_labels.npy"), np.array(SV_COMPONENTS))

    print(f"\nSaved to {output_dir}/:")
    print(f"  processed_components.parquet ({len(info_df)} prompts)")
    print(f"  head_as_component_labels.npy ({len(HEAD_COMPONENTS)} components)")
    print(f"  edge_as_component_labels.npy ({len(EDGE_COMPONENTS)} components)")
    print(f"  sv_as_component_labels.npy   ({len(SV_COMPONENTS)} components)")


if __name__ == "__main__":
    main()
