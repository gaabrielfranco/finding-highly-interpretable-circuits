"""Signal comparison heatmaps for representative prompt pairs.

Per-pair mode (Appendix figures): loads a model, dataset, traced graphs, and H5
signal cache, then computes cosine similarity matrices between signal vectors of
two prompts and plots destination/source heatmaps.

Combined mode (Figure 2): reads pre-computed NPZ files from per-pair runs and
plots a 2x3 grid (rows = dest/src, columns = models).

This is Step 6 of the Section 3 / Appendix F pipeline.

Usage:
    # Per-pair mode (one pair at a time)
    python experiments/plot_signals.py -m gpt2-small -p1 129 -p2 1613 -l1 BABA -l2 ABBA

    # Combined mode (reads NPZ files from per-pair runs)
    python experiments/plot_signals.py --combined
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch
from transformer_lens import HookedTransformer

# Add src/ to path so accpp_tracer is importable without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from accpp_tracer.datasets import IOIDataset

# ---------------------------------------------------------------------------
# Matplotlib defaults
# ---------------------------------------------------------------------------

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 8

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "gpt2-small": {
        "tl_name": "gpt2-small",
        "family": "gpt2",
        "prepend_bos": False,
        "display_name": "GPT-2 Small",
    },
    "pythia-160m": {
        "tl_name": "EleutherAI/pythia-160m",
        "family": "pythia",
        "prepend_bos": False,
        "display_name": "Pythia-160M",
    },
    "gemma-2-2b": {
        "tl_name": "gemma-2-2b",
        "family": "gemma",
        "prepend_bos": True,
        "display_name": "Gemma-2 2B",
    },
}

# NPZ files for the combined figure (Figure 2). These correspond to the
# per-pair runs produced by the shell script. The keys are model names,
# values are the NPZ filename (inside data/clustering/{model}/).
COMBINED_PAIRS = {
    "gpt2-small": "gpt2-small_signals_p1=BABA_p2=ABBA.npz",
    "pythia-160m": "pythia-160m_signals_p1=Template9_p2=Template10.npz",
    "gemma-2-2b": "gemma-2-2b_signals_p1=Template14_p2=Template15.npz",
}


# ---------------------------------------------------------------------------
# Graph parsing
# ---------------------------------------------------------------------------


def parse_graph(file, token_to_idx):
    """Parse a graphml file and extract AH components sorted by (layer, head, dest, src).

    Removes root (IO-S direction), Embedding, AH bias, AH offset, and MLP nodes.

    Args:
        file: Path to graphml file.
        token_to_idx: Dict mapping (prompt_id, token_label) -> position.

    Returns:
        Tuple of (components, labels) where components is a list of
        (layer, head, dest_idx, src_idx) tuples and labels is the corresponding
        raw node strings, both sorted by component tuple.
    """
    G = nx.read_graphml(file, force_multigraph=True)
    prompt_id = int(file.split("/")[-1].split("_")[-4])

    # Remove root node
    ios_node = [n for n in G.nodes if "'IO-S direction'" in n][0]
    G.remove_node(ios_node)

    # Remove non-AH nodes
    nodes_to_remove = []
    for node in list(G.nodes):
        if ("Embedding" in node) or ("AH bias" in node) or ("AH offset" in node) or ("MLP" in node):
            nodes_to_remove.append(node)
    G.remove_nodes_from(nodes_to_remove)

    # Extract AH components
    head_as_component = []
    head_labels = []
    for head in G.nodes:
        l, h, d, s = eval(head)
        if l != 0:
            d_idx = token_to_idx[(prompt_id, d)]
            src_idx = token_to_idx[(prompt_id, s)]
            head_as_component.append((l, h, d_idx, src_idx))
            head_labels.append(head)

    combined = sorted(zip(head_as_component, head_labels), key=lambda x: x[0])
    head_as_component, head_labels = map(list, zip(*combined))
    return head_as_component, head_labels


# ---------------------------------------------------------------------------
# Signal caching
# ---------------------------------------------------------------------------


def cache_all_signals(h5_path, device="cpu"):
    """Pre-load all signals from H5 and aggregate per (prompt, head, dest, src).

    For each unique (prompt_id, head, dest, src) group, sums destination signals
    (S_U for d-type edges) and source signals (S_V for s-type edges), then
    normalizes each to unit norm.

    Args:
        h5_path: Path to H5 file with per-edge signals.
        device: Torch device for intermediate computation.

    Returns:
        Dict mapping (sig_type, layer, head, dest, src, prompt_id) -> numpy array.
        sig_type is "U" (destination) or "V" (source).
    """
    cache = {}
    with h5py.File(h5_path, "r") as f:
        for layer_key in f.keys():
            layer_idx = int(layer_key.split("_")[1])
            grp = f[layer_key]
            S_U = torch.from_numpy(grp["S_U"][:]).to(device)
            S_V = torch.from_numpy(grp["S_V"][:]).to(device)
            metadata = grp["metadata"][:]
            edge_type = grp["edge_type"][:]
            d_edges = edge_type == b"d"
            s_edges = edge_type == b"s"

            unique_pairs = np.unique(metadata[:, :4], axis=0)
            for prompt_id, head, dest, src in unique_pairs:
                metadata_mask = (
                    (metadata[:, 0] == prompt_id)
                    & (metadata[:, 1] == head)
                    & (metadata[:, 2] == dest)
                    & (metadata[:, 3] == src)
                )
                d_mask = metadata_mask & d_edges
                s_mask = metadata_mask & s_edges

                # Destination signal
                u_sum = S_U[d_mask].sum(dim=0)
                u_norm = torch.norm(u_sum)
                cache[("U", layer_idx, int(head), int(dest), int(src), int(prompt_id))] = (
                    (u_sum / u_norm).numpy() if u_norm > 0 else u_sum.numpy()
                )

                # Source signal
                v_sum = S_V[s_mask].sum(dim=0)
                v_norm = torch.norm(v_sum)
                cache[("V", layer_idx, int(head), int(dest), int(src), int(prompt_id))] = (
                    (v_sum / v_norm).numpy() if v_norm > 0 else v_sum.numpy()
                )
    return cache


# ---------------------------------------------------------------------------
# Token-to-index mapping
# ---------------------------------------------------------------------------


def build_token_to_idx(model, dataset):
    """Build (prompt_id, token_label) -> position mapping for ALL prompts.

    Handles duplicate tokens with " (1)" suffix, matching the logic in
    trace.py:build_idx_to_token but inverted and keyed per prompt.

    Args:
        model: HookedTransformer model.
        dataset: IOIDataset instance.

    Returns:
        Dict mapping (prompt_id, token_label) -> position (int).
    """
    token_to_idx = {}
    for pid in range(len(dataset)):
        count_token_dict = defaultdict(int)
        end_pos = dataset.word_idx["end"][pid].item()
        for i in range(end_pos + 1):
            token_decoded = model.tokenizer.decode(dataset.toks[pid, i])
            count_token = count_token_dict[token_decoded]
            if count_token > 0:
                token_to_idx[(pid, f"{token_decoded} ({count_token})")] = i
            else:
                token_to_idx[(pid, token_decoded)] = i
            count_token_dict[token_decoded] += 1
    return token_to_idx


# ---------------------------------------------------------------------------
# Per-pair mode
# ---------------------------------------------------------------------------


def plot_pair(args):
    """Per-pair mode: compute and plot signal cosine similarity heatmaps."""
    cfg = MODEL_CONFIGS[args.model]

    graph_dir = args.graph_dir or f"data/traced_graphs/{args.model}/ioi-balanced"
    signals_dir = args.signals_dir or f"data/clustering/{args.model}"
    output_dir = args.output_dir or "figures/clustering"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(signals_dir, exist_ok=True)

    # Load model and dataset
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

    # Load signal cache
    h5_path = os.path.join(signals_dir, f"signals_balanced_{args.model}_not-norm.h5")
    print(f"Loading signal cache from {h5_path}...")
    signal_cache = cache_all_signals(h5_path)

    # Find graph files
    files = sorted(
        glob.glob(os.path.join(graph_dir, "*.graphml")),
        key=lambda x: int(x.split("/")[-1].split("_")[-4]),
    )
    indices = [int(x.split("/")[-1].split("_")[-4]) for x in files]

    # Build token-to-idx mapping
    print("Building token-to-idx mapping...")
    token_to_idx = build_token_to_idx(model, ioi_dataset)

    # Collect signals per prompt
    all_p_signals_U = []
    all_p_signals_V = []
    labels = []
    for i, file in enumerate(files):
        prompt_id = indices[i]
        components, head_labels = parse_graph(file, token_to_idx)

        sigs_u = [signal_cache.get(("U", l, h, d, s, prompt_id)) for l, h, d, s in components]
        sigs_v = [signal_cache.get(("V", l, h, d, s, prompt_id)) for l, h, d, s in components]

        if any(x is not None for x in sigs_u):
            all_p_signals_U.append(np.stack([sig for sig in sigs_u if sig is not None]))
            all_p_signals_V.append(np.stack([sig for sig in sigs_v if sig is not None]))
            labels.append([head_labels[j] for j, sig in enumerate(sigs_u) if sig is not None])
        else:
            all_p_signals_U.append(None)
            all_p_signals_V.append(None)
            labels.append(None)

    # Compute cosine similarity matrices
    i = indices.index(args.prompt_1)
    j = indices.index(args.prompt_2)
    u_i = all_p_signals_U[i]
    v_i = all_p_signals_V[i]
    u_j = all_p_signals_U[j]
    v_j = all_p_signals_V[j]
    dot_u = np.dot(u_i, u_j.T)
    dot_v = np.dot(v_i, v_j.T)

    # Build tick labels: (layer, head, token)
    dest_labels_i = [f"({eval(l)[0]}, {eval(l)[1]}, {eval(l)[2].strip()})" for l in labels[i]]
    dest_labels_j = [f"({eval(l)[0]}, {eval(l)[1]}, {eval(l)[2].strip()})" for l in labels[j]]
    src_labels_i = [f"({eval(l)[0]}, {eval(l)[1]}, {eval(l)[3].strip()})" for l in labels[i]]
    src_labels_j = [f"({eval(l)[0]}, {eval(l)[1]}, {eval(l)[3].strip()})" for l in labels[j]]

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.2))

    # Destination heatmap
    sns.heatmap(
        dot_u,
        ax=ax[0],
        xticklabels=dest_labels_j,
        yticklabels=dest_labels_i,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        cbar=False,
        rasterized=False,
    )
    ax[0].set_title("Destination")
    ax[0].tick_params(axis="both", which="major", width=0.2, length=1, pad=0.5, labelsize=4)

    # Source heatmap
    sns.heatmap(
        dot_v,
        ax=ax[1],
        xticklabels=src_labels_j,
        yticklabels=src_labels_i,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        cbar=False,
        rasterized=False,
    )
    ax[1].set_title("Source")
    ax[1].tick_params(axis="both", which="major", width=0.2, length=1, pad=0.5, labelsize=4)

    # Colorbar
    plt.subplots_adjust(left=0.0, right=0.87, top=0.85, bottom=0.25, wspace=0.25, hspace=0.0)
    mappable = ax[1].collections[0]
    cbar_ax = fig.add_axes([0.88, 0.30, 0.015, 0.45])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label("Cosine Similarity", fontsize=6)
    cbar.ax.tick_params(labelsize=6, width=0.2, length=1, pad=2)
    cbar.outline.set_visible(False)

    # Save PDF
    l1_clean = args.label_1.replace(" ", "")
    l2_clean = args.label_2.replace(" ", "")
    pdf_name = f"{args.model}_signals_p1={l1_clean}_p2={l2_clean}.pdf"
    pdf_path = os.path.join(output_dir, pdf_name)
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {pdf_path}")

    # Save NPZ
    npz_name = f"{args.model}_signals_p1={l1_clean}_p2={l2_clean}.npz"
    npz_path = os.path.join(signals_dir, npz_name)
    np.savez(
        npz_path,
        dest_map=dot_u,
        src_map=dot_v,
        dest_p1_label=dest_labels_i,
        dest_p2_label=dest_labels_j,
        src_p1_label=src_labels_i,
        src_p2_label=src_labels_j,
    )
    print(f"Saved {npz_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Combined mode (Figure 2)
# ---------------------------------------------------------------------------


def plot_combined(args):
    """Combined mode: 2x3 grid from pre-computed NPZ files."""
    mpl.rcParams["font.size"] = 3
    mpl.rcParams["axes.linewidth"] = 0.0

    signals_base = args.signals_dir or "data/clustering"
    output_dir = args.output_dir or "figures/clustering"
    os.makedirs(output_dir, exist_ok=True)

    models = [
        ("gpt2-small", MODEL_CONFIGS["gpt2-small"]["display_name"]),
        ("pythia-160m", MODEL_CONFIGS["pythia-160m"]["display_name"]),
        ("gemma-2-2b", MODEL_CONFIGS["gemma-2-2b"]["display_name"]),
    ]

    data_objects = []
    for model_name, _ in models:
        npz_file = os.path.join(signals_base, model_name, COMBINED_PAIRS[model_name])
        print(f"Loading {npz_file}...")
        data_objects.append(np.load(npz_file))

    vmin, vmax = -1, 1
    fig, axes = plt.subplots(2, 3, figsize=(6.4, 4))

    for col_idx, (data, (_, display_name)) in enumerate(zip(data_objects, models)):
        # Row 0: Destination map
        ax_dest = axes[0, col_idx]
        im = ax_dest.imshow(data["dest_map"], aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax_dest.set_title(display_name, fontsize=8)
        if col_idx == 0:
            ax_dest.set_ylabel("Destination", fontsize=8)
        ax_dest.set_yticks(np.arange(len(data["dest_p1_label"])))
        ax_dest.set_yticklabels(data["dest_p1_label"])
        ax_dest.set_xticks(np.arange(len(data["dest_p2_label"])))
        ax_dest.set_xticklabels(data["dest_p2_label"], rotation=90)
        ax_dest.tick_params(axis="both", which="major", width=0.2, length=1, pad=0.5)

        # Row 1: Source map
        ax_src = axes[1, col_idx]
        im = ax_src.imshow(data["src_map"], aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
        if col_idx == 0:
            ax_src.set_ylabel("Source", fontsize=8)
        ax_src.set_yticks(np.arange(len(data["src_p1_label"])))
        ax_src.set_yticklabels(data["src_p1_label"])
        ax_src.set_xticks(np.arange(len(data["src_p2_label"])))
        ax_src.set_xticklabels(data["src_p2_label"], rotation=90)
        ax_src.tick_params(axis="both", which="major", width=0.2, length=1, pad=0.5)

    # Colorbar
    plt.subplots_adjust(left=0.08, right=0.88, top=0.88, bottom=0.20, wspace=0.05, hspace=0.0)
    cbar_ax = fig.add_axes([1.01, 0.35, 0.01, 0.3])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Cosine Similarity", fontsize=6)
    cbar.ax.tick_params(labelsize=6, width=0.2, length=1, pad=2)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    output_path = os.path.join(output_dir, "model_comparison_heatmap.pdf")
    plt.savefig(output_path, bbox_inches="tight", dpi=1200)
    print(f"Saved {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Signal comparison heatmaps (per-pair or combined figure)"
    )

    # Per-pair arguments
    parser.add_argument(
        "-m", "--model",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model name (required for per-pair mode).",
    )
    parser.add_argument("-p1", "--prompt_1", type=int, help="Prompt ID for first prompt.")
    parser.add_argument("-p2", "--prompt_2", type=int, help="Prompt ID for second prompt.")
    parser.add_argument("-l1", "--label_1", help="Label for prompt 1.")
    parser.add_argument("-l2", "--label_2", help="Label for prompt 2.")

    # Path arguments
    parser.add_argument(
        "--graph_dir", type=str, default=None,
        help="Directory with graphml files. Default: data/traced_graphs/{model}/ioi-balanced",
    )
    parser.add_argument(
        "--signals_dir", type=str, default=None,
        help="Directory with H5 signal files. Default: data/clustering/{model}",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for figures. Default: figures/clustering",
    )

    # Combined mode
    parser.add_argument(
        "--combined", action="store_true",
        help="Generate the 2x3 combined paper figure (Figure 2) from NPZ files.",
    )

    args = parser.parse_args()

    if args.combined:
        plot_combined(args)
    else:
        # Validate per-pair arguments
        if args.model is None:
            parser.error("-m/--model is required for per-pair mode")
        if args.prompt_1 is None or args.prompt_2 is None:
            parser.error("-p1 and -p2 are required for per-pair mode")
        if args.label_1 is None or args.label_2 is None:
            parser.error("-l1 and -l2 are required for per-pair mode")
        plot_pair(args)


if __name__ == "__main__":
    main()
