"""Stage 5b: Annotate GraphML circuit graphs with Gemini interpretations.

For each representative prompt, loads the traced circuit GraphML, annotates
edges with ``interpretation`` and ``full_response`` attributes from the
Gemini API output (Stage 5a), applies Cytoscape layout for visualization,
and exports a tokens JSON file for Bokeh x-axis labels.

If ``--edge_examples_file`` is provided (the side output from Step 7's
``build-request``), remaps signal keys to graph edge keys
(``"{u}|||{v}|||{multigraph_key}"``) and writes a per-prompt
``*_edge_examples.json`` alongside the GraphML.

Usage:
    python annotate_graphs.py \\\\
        --model gpt2-small \\\\
        --task ioi-balanced \\\\
        --representatives 129 1613 1372 \\\\
        --interpretations_file interpretations_representatives.json \\\\
        --edge_examples_file prompts_representatives_edge_examples.json \\\\
        --signals_file data/clustering/gpt2-small/signals_balanced_gpt2-small_not-norm.h5 \\\\
        --graphs_dir data/traced_graphs/gpt2-small/ioi-balanced/ \\\\
        --output_dir data/autointerp/graphs_interp/ \\\\
        --root_node_label "IO-S direction"
"""

import argparse
import glob
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import networkx as nx
import numpy as np
import torch
from transformer_lens import HookedTransformer

from accpp_tracer.graphs.visualization import format_graph_cytoscape_by_token_pos

from h5_utils import parse_edge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_model_family(model_name: str) -> str:
    """Map a TransformerLens model name to the IOIDataset model_family string.

    Args:
        model_name: TransformerLens model name (e.g. ``"gpt2-small"``).

    Returns:
        One of ``"gpt2"``, ``"pythia"``, ``"gemma"``, or ``"llama2"``.

    Raises:
        ValueError: If the model family cannot be determined.
    """
    name_lower = model_name.lower()
    if "gpt2" in name_lower:
        return "gpt2"
    if "pythia" in name_lower:
        return "pythia"
    if "gemma" in name_lower:
        return "gemma"
    if "llama" in name_lower:
        return "llama2"
    raise ValueError(
        f"Cannot determine model_family for model '{model_name}'. "
        f"Expected one of: gpt2, pythia, gemma, llama2."
    )


def _load_dataset(
    model: HookedTransformer,
    task: str,
    n_prompts: int,
    device: str,
) -> Any:
    """Load the appropriate dataset for a given task.

    Args:
        model: Loaded TransformerLens model (used for tokenizer and
            ``FactsDataset`` token encoding).
        task: Task identifier (e.g. ``"ioi-balanced"``, ``"ioi"``,
            ``"facts"``).
        n_prompts: Number of prompts in the dataset (``N`` for IOI,
            ``n`` for Facts).
        device: Torch device string.

    Returns:
        Dataset object with ``toks`` and ``word_idx["end"]`` attributes.

    Raises:
        ValueError: If the task is not supported.
    """
    task_lower = task.lower()

    if "ioi" in task_lower:
        from accpp_tracer.datasets.ioi import IOIDataset
        prompt_type = "balanced" if "balanced" in task_lower else "mixed"
        model_family = _get_model_family(model.cfg.model_name)
        return IOIDataset(
            prompt_type=prompt_type,
            model_family=model_family,
            N=n_prompts,
            tokenizer=model.tokenizer,
            device=device,
        )

    if task_lower == "facts":
        from accpp_tracer.datasets.facts import FactsDataset
        return FactsDataset(model=model, device=device)

    raise ValueError(
        f"Unsupported task: '{task}'. "
        f"Supported tasks: ioi, ioi-balanced, facts."
    )


def _build_sentence_tokens(
    tokenizer: Any,
    toks: torch.Tensor,
    end_pos: int,
) -> list[str]:
    """Build the ordered sentence token list for Cytoscape layout.

    Decodes tokens from position 0 to ``end_pos`` (inclusive) and appends
    a count suffix to handle duplicate tokens (e.g. ``" Marco (1)"``).
    This matches the token labeling used by the tracer in circuit node IDs.

    Args:
        tokenizer: TransformerLens tokenizer.
        toks: Token ID tensor for one prompt, shape (seq_len,).
        end_pos: Index of the last token to include (inclusive).

    Returns:
        List of token strings of length ``end_pos + 1``.
    """
    tokens: list[str] = []
    count_token_dict: dict[str, int] = defaultdict(int)
    for i in range(end_pos + 1):
        token_decoded = tokenizer.decode(int(toks[i]))
        count = count_token_dict[token_decoded]
        if count > 0:
            tokens.append(f"{token_decoded} ({count})")
        else:
            tokens.append(token_decoded)
        count_token_dict[token_decoded] += 1
    return tokens


def _compute_node_names_mapping(
    G: nx.MultiDiGraph,
    root_node: str,
    sentence_tokens: list[str],
) -> dict[str, str]:
    """Compute the node relabeling produced by ``format_graph_cytoscape_by_token_pos``.

    This mirrors the ``node_names_mapping`` computed inside
    ``format_graph_cytoscape_by_token_pos`` (``graphs/visualization.py``)
    and must be kept in sync with that function.

    Used to remap edge example keys from original node labels (as stored in
    the signal H5 ``edges`` dataset) to the relabeled node labels that appear
    in the output Cytoscape GraphML.

    Args:
        G: Graph with original node labels (before relabeling).
        root_node: The root node label (e.g. ``"('IO-S direction', ' to')"``)
        sentence_tokens: List of token strings (added as bottom nodes by
            ``format_graph_cytoscape_by_token_pos``; not relabeled).

    Returns:
        Dict mapping original node label → new node label.
    """
    _, dest_token_root = eval(root_node)
    mapping: dict[str, str] = {root_node: f"Logit direction\n{dest_token_root}"}
    for node in G.nodes:
        if node == root_node or node in sentence_tokens:
            continue
        try:
            node_tuple = eval(node)
        except Exception:
            continue
        if node_tuple[1] in ["MLP", "AH bias", "Embedding", "AH offset"]:
            mapping[node] = f"{node_tuple[1]} {node_tuple[0]}\n{node_tuple[2]}"
        else:
            mapping[node] = (
                f"AH({node_tuple[0]},{node_tuple[1]})\n({node_tuple[2]},{node_tuple[3]})"
            )
    return mapping


def _prompt_id_from_filename(filename: str) -> int:
    """Extract the prompt ID from a traced GraphML filename.

    Follows the naming convention from ``trace.py``:
    ``{model}_{task}_n{N}_{prompt_id}_{seed}_{thresh}_{ordering}.graphml``

    The prompt ID is the 4th-from-last component when split by ``_``.

    Args:
        filename: Basename of the GraphML file.

    Returns:
        Integer prompt ID.

    Raises:
        ValueError: If the prompt ID cannot be parsed.
    """
    parts = Path(filename).stem.split("_")
    try:
        return int(parts[-4])
    except (IndexError, ValueError) as e:
        raise ValueError(
            f"Cannot parse prompt_id from filename '{filename}': {e}"
        ) from e


# ---------------------------------------------------------------------------
# Main annotation function
# ---------------------------------------------------------------------------

def annotate_graphs(
    model_name: str,
    task: str,
    representatives: list[int],
    interpretations_file: str,
    signals_file: str,
    graphs_dir: str,
    output_dir: str,
    root_node_label: str,
    edge_examples_file: str | None = None,
    n_prompts: int = 3000,
    device: str = "cpu",
    hf_cache_dir: str | None = None,
) -> None:
    """Annotate representative circuit graphs and apply Cytoscape layout.

    For each representative prompt:

    1. Loads the traced GraphML from ``graphs_dir``.
    2. For every layer group in ``signals_file``, filters signals to those
       whose ``prompt_id`` matches a representative, then sets
       ``interpretation`` and ``full_response`` edge attributes from the
       ``interpretations_file`` JSON (keyed by
       ``"layer_{L}_{global_signal_idx}"``).
    3. Builds the sentence token list from the dataset and applies the
       Cytoscape layout via
       ``accpp_tracer.graphs.visualization.format_graph_cytoscape_by_token_pos``.
    4. Writes the annotated + Cytoscape-layout GraphML and a tokens JSON
       to ``output_dir``.
    5. If ``edge_examples_file`` is provided, remaps signal keys to
       graph edge keys ``"{u}|||{v}|||{key}"`` and writes per-prompt
       ``*_edge_examples.json`` alongside the GraphML.

    Args:
        model_name: TransformerLens model name (e.g. ``"gpt2-small"``).
        task: Task identifier (e.g. ``"ioi-balanced"``).
        representatives: List of representative prompt IDs.
        interpretations_file: Path to the interpretations JSON produced
            by ``interpret_representatives.py parse-response``.
        signals_file: Path to the signal H5 file produced by
            ``extract_signals.py`` (Part 2, Step 5).
        graphs_dir: Directory containing the traced GraphML files (from
            ``trace.py`` or ``run_tracing_balanced.sh``).
        output_dir: Directory to write annotated + Cytoscape-layout
            GraphML files, tokens JSON, and (if requested) edge examples
            JSON.
        root_node_label: Prefix used to identify the root node in the
            graph (e.g. ``"IO-S direction"`` for IOI,
            ``"Correct answer"`` for facts).
        edge_examples_file: Optional path to the ``*_edge_examples.json``
            side output from ``interpret_representatives.py build-request``.
            If provided, remaps keys and writes per-prompt
            ``*_edge_examples.json`` outputs.
        n_prompts: Number of prompts in the dataset (passed to the
            dataset constructor as ``N`` for IOI or ``n`` for Facts).
            Default 3000.
        device: Torch device string. The model is loaded on CPU regardless;
            this is passed to the dataset constructor.
        hf_cache_dir: Optional HuggingFace cache directory. Sets
            ``HF_HOME`` if provided.
    """
    if hf_cache_dir is not None:
        os.environ["HF_HOME"] = hf_cache_dir

    log.info(f"Loading model: {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device="cpu")
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    log.info(f"Loading dataset for task: {task}")
    dataset = _load_dataset(model, task, n_prompts, device="cpu")

    log.info(f"Loading interpretations: {interpretations_file}")
    with open(interpretations_file, "r", encoding="utf-8") as f:
        interpretations: dict[str, dict[str, str]] = json.load(f)

    # Load optional edge examples
    edge_examples_input: dict[str, list[str]] | None = None
    if edge_examples_file is not None:
        log.info(f"Loading edge examples: {edge_examples_file}")
        with open(edge_examples_file, "r", encoding="utf-8") as f:
            edge_examples_input = json.load(f)

    # Discover GraphML files for representative prompts
    all_graphml = glob.glob(str(Path(graphs_dir) / "*.graphml"))
    representatives_set = set(representatives)

    GRAPHS: dict[int, nx.MultiDiGraph] = {}
    for filepath in all_graphml:
        try:
            pid = _prompt_id_from_filename(Path(filepath).name)
        except ValueError as e:
            log.warning(f"Skipping {filepath}: {e}")
            continue
        if pid in representatives_set:
            GRAPHS[pid] = nx.read_graphml(filepath, force_multigraph=True)
            log.info(f"Loaded graph for prompt {pid}: {GRAPHS[pid]}")

    missing = representatives_set - set(GRAPHS.keys())
    if missing:
        log.warning(f"No GraphML found for prompt IDs: {sorted(missing)}")

    # Enumerate all layer groups in the signals H5
    with h5py.File(signals_file, "r") as f:
        layer_names = sorted(
            [k for k in f.keys() if k.startswith("layer_")],
            key=lambda s: int(s.split("_")[1]),
        )
    log.info(f"Found {len(layer_names)} layers in {signals_file}")

    # ------------------------------------------------------------------
    # Annotation pass: set edge attributes on each graph
    # ------------------------------------------------------------------
    for layer_name in layer_names:
        with h5py.File(signals_file, "r") as f:
            grp = f[layer_name]
            metadata_all = grp["metadata"][:]  # (n_signals, 7), int32
            edges_all = grp["edges"][:]        # structured (u, v, key)

        prompt_ids = metadata_all[:, 0].astype(int)
        signal_ids = np.where(np.isin(prompt_ids, list(representatives_set)))[0]

        for signal_id in signal_ids:
            prompt_id = int(prompt_ids[signal_id])
            if prompt_id not in GRAPHS:
                continue

            try:
                edge = parse_edge(edges_all[signal_id])
            except Exception as e:
                log.warning(f"Layer {layer_name} signal {signal_id}: edge parse error: {e}")
                continue

            key = f"{layer_name}_{signal_id}"

            if key not in interpretations:
                log.debug(f"Key {key} not in interpretations, skipping")
                continue

            try:
                GRAPHS[prompt_id].edges[edge]["interpretation"] = (
                    interpretations[key]["interpretation"]
                )
                GRAPHS[prompt_id].edges[edge]["full_response"] = (
                    interpretations[key]["full_response"]
                )
            except KeyError:
                log.warning(
                    f"Edge {edge} not found in graph for prompt {prompt_id} "
                    f"(layer {layer_name}, signal {signal_id})"
                )

    # ------------------------------------------------------------------
    # Per-prompt: Cytoscape layout + output
    # ------------------------------------------------------------------
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for pid in sorted(GRAPHS.keys()):
        G = GRAPHS[pid]

        end_pos = int(dataset.word_idx["end"][pid])
        sentence_tokens = _build_sentence_tokens(model.tokenizer, dataset.toks[pid], end_pos)

        # Find root node by label prefix
        root_nodes = [node for node in G.nodes if root_node_label in node]
        if len(root_nodes) != 1:
            log.warning(
                f"Prompt {pid}: expected 1 root node with label '{root_node_label}', "
                f"found {len(root_nodes)}: {root_nodes}. Skipping Cytoscape layout."
            )
            continue
        root_node = root_nodes[0]

        # Compute node_names_mapping before relabeling (for edge examples remap)
        node_names_mapping: dict[str, str] = {}
        if edge_examples_input is not None:
            node_names_mapping = _compute_node_names_mapping(G, root_node, sentence_tokens)

        # Build per-prompt edge examples with relabeled keys
        prompt_edge_examples: dict[str, list[str]] = {}
        if edge_examples_input is not None:
            layer_names_set = set(layer_names)
            with h5py.File(signals_file, "r") as f:
                for layer_name in layer_names:
                    grp = f[layer_name]
                    metadata_all = grp["metadata"][:]
                    edges_all = grp["edges"][:]

                    prompt_ids = metadata_all[:, 0].astype(int)
                    signal_ids = np.where(prompt_ids == pid)[0]

                    for signal_id in signal_ids:
                        key = f"{layer_name}_{signal_id}"
                        if key not in edge_examples_input:
                            continue

                        try:
                            u, v, edge_key = parse_edge(edges_all[signal_id])
                        except Exception:
                            continue

                        new_u = node_names_mapping.get(u, u)
                        new_v = node_names_mapping.get(v, v)
                        new_edge_key = f"{new_u}|||{new_v}|||{edge_key}"
                        prompt_edge_examples[new_edge_key] = edge_examples_input[key]

        # Apply Cytoscape layout (relabels nodes)
        G = format_graph_cytoscape_by_token_pos(G, root_node, n_layers, n_heads, sentence_tokens)

        # Determine output filename (preserve input filename stem)
        input_files = [
            fp for fp in all_graphml
            if _safe_prompt_id(fp) == pid
        ]
        if input_files:
            out_stem = Path(input_files[0]).stem
        else:
            model_short = model_name.split("/")[-1]
            out_stem = f"{model_short}_{task}_n{n_prompts}_{pid}_0_dynamic_ig"

        out_graphml = output_path / f"{out_stem}.graphml"
        out_tokens_json = output_path / f"{out_stem}-tokens.json"

        nx.write_graphml(G, str(out_graphml))
        log.info(f"Prompt {pid}: wrote {out_graphml}")

        with open(out_tokens_json, "w", encoding="utf-8") as f:
            json.dump(sentence_tokens, f, ensure_ascii=False, indent=2)
        log.info(f"Prompt {pid}: wrote {out_tokens_json}")

        if edge_examples_input is not None and prompt_edge_examples:
            out_examples_json = output_path / f"{out_stem}_edge_examples.json"
            with open(out_examples_json, "w", encoding="utf-8") as f:
                json.dump(prompt_edge_examples, f, ensure_ascii=False, indent=2)
            log.info(
                f"Prompt {pid}: wrote {out_examples_json} "
                f"({len(prompt_edge_examples)} edge entries)"
            )


def _safe_prompt_id(filepath: str) -> int | None:
    """Return prompt ID from a filepath, or None if it cannot be parsed."""
    try:
        return _prompt_id_from_filename(Path(filepath).name)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 5b: Annotate GraphML circuit graphs with Gemini interpretations "
            "and apply Cytoscape layout."
        )
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="TransformerLens model name (e.g. gpt2-small).",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task identifier (e.g. ioi-balanced, facts).",
    )
    parser.add_argument(
        "--representatives",
        type=int,
        nargs="+",
        required=True,
        help="Representative prompt IDs (space-separated integers).",
    )
    parser.add_argument(
        "--interpretations_file",
        required=True,
        help=(
            "Path to the interpretations JSON from "
            "interpret_representatives.py parse-response."
        ),
    )
    parser.add_argument(
        "--signals_file",
        required=True,
        help=(
            "Path to the signal H5 file "
            "(e.g. data/clustering/gpt2-small/signals_balanced_gpt2-small_not-norm.h5)."
        ),
    )
    parser.add_argument(
        "--graphs_dir",
        required=True,
        help="Directory containing the traced GraphML files.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/autointerp/graphs_interp/",
        help=(
            "Output directory for annotated GraphML, tokens JSON, and "
            "edge examples JSON (default: data/autointerp/graphs_interp/)."
        ),
    )
    parser.add_argument(
        "--root_node_label",
        default="IO-S direction",
        help=(
            "Prefix used to identify the root node (default: 'IO-S direction'). "
            "Use 'Correct answer' for facts prompts."
        ),
    )
    parser.add_argument(
        "--edge_examples_file",
        default=None,
        help=(
            "Optional path to the edge examples JSON from "
            "interpret_representatives.py build-request. "
            "If provided, produces per-prompt *_edge_examples.json outputs."
        ),
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=3000,
        help=(
            "Number of prompts in the dataset (N for IOI, n for Facts). "
            "Default: 3000."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device string (default: cpu). Dataset is always loaded on CPU.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        default=None,
        help="Optional HuggingFace cache directory (sets HF_HOME).",
    )
    args = parser.parse_args()

    annotate_graphs(
        model_name=args.model,
        task=args.task,
        representatives=args.representatives,
        interpretations_file=args.interpretations_file,
        signals_file=args.signals_file,
        graphs_dir=args.graphs_dir,
        output_dir=args.output_dir,
        root_node_label=args.root_node_label,
        edge_examples_file=args.edge_examples_file,
        n_prompts=args.n_prompts,
        device=args.device,
        hf_cache_dir=args.hf_cache_dir,
    )


if __name__ == "__main__":
    main()
