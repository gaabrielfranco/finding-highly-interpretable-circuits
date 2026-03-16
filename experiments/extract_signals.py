"""Extract per-edge signal vectors from traced balanced IOI circuit graphs.

For each edge in the circuit graph, computes the destination (signal_u) and source
(signal_v) signal vectors using `tracer.extract_edge_signal()`, and saves results
to H5 format grouped by downstream layer.

This is Step 5b of the Section 3 / Appendix F pipeline.

Usage:
    python experiments/extract_signals.py -m gpt2-small -d mps
    python experiments/extract_signals.py -m EleutherAI/pythia-160m -d cpu
    python experiments/extract_signals.py -m gemma-2-2b -d cuda --batch_size 8
"""

import argparse
import gc
import glob
import os
import sys
from collections import defaultdict

import h5py
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

# Add src/ to path so accpp_tracer is importable without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from accpp_tracer import Tracer
from accpp_tracer.datasets import IOIDataset
from accpp_tracer.signals import component_label_to_id

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "gpt2-small": {"family": "gpt2", "prepend_bos": False, "use_numpy_svd": False},
    "EleutherAI/pythia-160m": {"family": "pythia", "prepend_bos": False, "use_numpy_svd": True},
    "gemma-2-2b": {"family": "gemma", "prepend_bos": True, "use_numpy_svd": False},
}

SUPPORTED_MODELS = list(MODEL_CONFIGS.keys())


# ---------------------------------------------------------------------------
# Position resolution: node labels -> token positions
# ---------------------------------------------------------------------------


def build_token_to_idx(model, dataset, prompt_id):
    """Build token label -> position mapping for a prompt.

    Decodes each token and handles duplicates with ``" (1)"`` suffix,
    matching the logic in ``trace.py:build_idx_to_token`` but inverted.

    Args:
        model: HookedTransformer model.
        dataset: Dataset instance (must have .toks and .word_idx).
        prompt_id: Index of the prompt in the dataset.

    Returns:
        Dict mapping token label (str) -> position (int).
    """
    token_to_idx: dict[str, int] = {}
    count_dict: dict[str, int] = defaultdict(int)
    end_pos = dataset.word_idx["end"][prompt_id].item()
    for i in range(end_pos + 1):
        tok_str = model.tokenizer.decode(dataset.toks[prompt_id, i])
        count = count_dict[tok_str]
        if count > 0:
            token_to_idx[f"{tok_str} ({count})"] = i
        else:
            token_to_idx[tok_str] = i
        count_dict[tok_str] += 1
    return token_to_idx


def build_gram_role_to_idx(dataset, prompt_id):
    """Build gram role label -> position mapping for a prompt (IOI only).

    Args:
        dataset: IOI dataset instance.
        prompt_id: Index of the prompt in the dataset.

    Returns:
        Dict mapping gram role label (str) -> position (int).
    """
    role_to_idx: dict[str, int] = {}
    for gram_role in dataset.word_idx.keys():
        pos = dataset.word_idx[gram_role][prompt_id].item()
        role_to_idx[gram_role] = pos
    return role_to_idx


# ---------------------------------------------------------------------------
# H5 I/O
# ---------------------------------------------------------------------------


def save_signals_to_h5(signals_dict, filename):
    """Save extracted signals to H5 file.

    Replicates the format from the original interpreting-signals/signals.py.

    Args:
        signals_dict: Dict mapping downstream_layer (int) -> dict with keys
            "u", "v" (lists of numpy arrays), "metadata" (list of 7-element
            lists), "edge_type" (list of str), "edge" (list of 3-tuples).
        filename: Output H5 file path.
    """
    with h5py.File(filename, "w") as f:
        for layer_idx, data in signals_dict.items():
            grp_name = f"layer_{layer_idx}"
            grp = f.create_group(grp_name)

            S_U = np.stack(data["u"]).astype(np.float32)
            S_V = np.stack(data["v"]).astype(np.float32)
            meta = np.array(data["metadata"], dtype=np.int32)
            types = np.array(data["edge_type"], dtype="S1")

            dt = np.dtype([
                ("u", h5py.string_dtype(encoding="utf-8")),
                ("v", h5py.string_dtype(encoding="utf-8")),
                ("key", "i4"),
            ])
            edges = np.array(data["edge"], dtype=dt)

            grp.create_dataset("S_U", data=S_U)
            grp.create_dataset("S_V", data=S_V)
            grp.create_dataset("metadata", data=meta, compression="lzf")
            grp.create_dataset("edge_type", data=types, compression="lzf")
            grp.create_dataset("edges", data=edges, compression="lzf")

            print(f"  Saved {grp_name}: {len(S_U)} signals")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-edge signal vectors from traced circuit graphs"
    )
    parser.add_argument(
        "-m", "--model_name",
        choices=SUPPORTED_MODELS,
        required=True,
    )
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--graph_dir", type=str, default=None,
        help="Directory with graphml files. Default: data/traced_graphs/{model_short}/ioi-balanced",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output H5 file. Default: data/clustering/{model_short}/signals_balanced_{model_short}_not-norm.h5",
    )
    parser.add_argument(
        "--label_mode", type=str, default="tokens",
        choices=["tokens", "roles"],
        help=(
            "'roles' = gram role labels (default, matching balanced IOI tracing), "
            "'tokens' = actual decoded tokens."
        ),
    )
    args = parser.parse_args()

    model_cfg = MODEL_CONFIGS[args.model_name]
    model_short = args.model_name.split("/")[-1]

    graph_dir = args.graph_dir or f"data/traced_graphs/{model_short}/ioi-balanced"
    output_path = args.output or (
        f"data/clustering/{model_short}/signals_balanced_{model_short}_not-norm.h5"
    )

    # --- Find graph files ---
    files = sorted(glob.glob(os.path.join(graph_dir, "*.graphml")))
    if not files:
        print(f"No graphml files found in {graph_dir}/")
        return
    print(f"Found {len(files)} graph files in {graph_dir}/")

    # --- Extract prompt IDs from filenames ---
    # Filename: {model}_{task}_n{N}_{prompt_id}_{seed}_{thresh}_ig.graphml
    prompt_ids_from_files = []
    file_by_prompt_id = {}
    for f in files:
        pid = int(f.split("/")[-1].split(".")[0].split("_")[-4])
        prompt_ids_from_files.append(pid)
        file_by_prompt_id[pid] = f
    prompt_ids_from_files = sorted(set(prompt_ids_from_files))
    print(f"  {len(prompt_ids_from_files)} unique prompt IDs")

    torch.set_grad_enabled(False)

    # --- Load model ---
    print(f"Loading {args.model_name}...")
    model = HookedTransformer.from_pretrained(args.model_name, device=args.device)
    n_heads = model.cfg.n_heads

    # --- Create dataset ---
    print("Creating balanced IOI dataset (N=3000)...")
    dataset = IOIDataset(
        model_family=model_cfg["family"],
        prompt_type="balanced",
        N=3000,
        tokenizer=model.tokenizer,
        prepend_bos=model_cfg["prepend_bos"],
        seed=0,
        device=args.device,
    )

    # --- Initialize tracer ---
    print("Initializing Tracer (precomputing Omega SVD + pseudoinverses)...")
    tracer = Tracer(
        model,
        device=args.device,
        use_numpy_svd=model_cfg["use_numpy_svd"],
    )

    # --- Build label-to-position mappings for all prompts ---
    print("Building label-to-position mappings...")
    label_to_pos = {}
    for pid in prompt_ids_from_files:
        if args.label_mode == "tokens":
            label_to_pos[pid] = build_token_to_idx(model, dataset, pid)
        else:
            label_to_pos[pid] = build_gram_role_to_idx(dataset, pid)

    # --- Process prompts in batches ---
    SIGNALS = defaultdict(
        lambda: {"u": [], "v": [], "metadata": [], "edge_type": [], "edge": []}
    )
    total_edges = 0

    n_batches = (len(prompt_ids_from_files) + args.batch_size - 1) // args.batch_size
    print(f"\nProcessing {len(prompt_ids_from_files)} prompts in {n_batches} batches "
          f"(batch_size={args.batch_size})...")

    for batch_idx in tqdm(range(n_batches), desc="Batches"):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(prompt_ids_from_files))
        batch_pids = prompt_ids_from_files[batch_start:batch_end]

        # Forward pass on this batch
        _, cache = model.run_with_cache(dataset.toks[batch_pids])

        for local_idx, prompt_id in enumerate(batch_pids):
            graph_file = file_by_prompt_id[prompt_id]
            G = nx.read_graphml(graph_file, force_multigraph=True)
            mapping = label_to_pos[prompt_id]

            for upstream_node, downstream_node, key, data in G.edges(keys=True, data=True):
                if "svs_used" not in data:
                    continue

                # Parse node tuples
                downstream_vals = eval(downstream_node)
                upstream_vals = eval(upstream_node)

                downstream_layer = downstream_vals[0]
                downstream_ah_idx = downstream_vals[1]
                downstream_dest_label = downstream_vals[2]
                downstream_src_label = downstream_vals[3]

                upstream_layer = upstream_vals[0]
                upstream_component = upstream_vals[1]
                upstream_dest_label = upstream_vals[2]
                upstream_src_label = upstream_vals[3]

                upstream_component_id = component_label_to_id(upstream_component, n_heads)
                edge_type = data["type"]
                svs_used = eval(data["svs_used"])

                # Resolve labels to positions
                downstream_dest_token = mapping[downstream_dest_label]
                downstream_src_token = mapping[downstream_src_label]
                upstream_dest_token = mapping[upstream_dest_label]
                upstream_src_token = mapping[upstream_src_label]

                # Extract signal pair
                signal_u, signal_v = tracer.extract_edge_signal(
                    cache=cache,
                    prompt_idx=local_idx,
                    downstream_layer=downstream_layer,
                    downstream_ah_idx=downstream_ah_idx,
                    downstream_dest_token=downstream_dest_token,
                    downstream_src_token=downstream_src_token,
                    upstream_layer=upstream_layer,
                    upstream_component_id=upstream_component_id,
                    upstream_dest_token=upstream_dest_token,
                    upstream_src_token=upstream_src_token,
                    edge_type=edge_type,
                    svs_used=svs_used,
                )

                # Accumulate
                metadata_row = [
                    prompt_id,
                    downstream_ah_idx,
                    downstream_dest_token,
                    downstream_src_token,
                    upstream_layer,
                    upstream_component_id,
                    upstream_src_token,
                ]
                SIGNALS[downstream_layer]["u"].append(signal_u.cpu().numpy())
                SIGNALS[downstream_layer]["v"].append(signal_v.cpu().numpy())
                SIGNALS[downstream_layer]["metadata"].append(metadata_row)
                SIGNALS[downstream_layer]["edge_type"].append(edge_type)
                SIGNALS[downstream_layer]["edge"].append(
                    (upstream_node, downstream_node, int(key))
                )
                total_edges += 1

        del cache
        gc.collect()

    print(f"\nTotal signals extracted: {total_edges}")

    # --- Save to H5 ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving to {output_path}...")
    save_signals_to_h5(SIGNALS, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
