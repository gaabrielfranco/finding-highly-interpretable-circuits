"""Reproduce ACC++ circuit tracing results from the paper.

Supports all model-task combinations:
  Models: gpt2-small, EleutherAI/pythia-160m, EleutherAI/pythia-160m-deduped, gemma-2-2b
  Tasks:  ioi, ioi-balanced, gt, gp, facts

Usage examples:
    python experiments/trace.py -m gpt2-small -t ioi -n 8 -d mps
    python experiments/trace.py -m gpt2-small -t ioi-balanced -d mps
    python experiments/trace.py -m gpt2-small -t gt -n 20 -d mps
    python experiments/trace.py -m gpt2-small -t gp -d mps
    python experiments/trace.py -m gpt2-small -t facts -d mps
    python experiments/trace.py -m gemma-2-2b -t ioi -n 8 -d mps

Batched tracing (qsub-friendly):
    # Process batch 0 (prompts 0-31) of 8 batches of size 32:
    python experiments/trace.py -m gpt2-small -t ioi -n 256 --batch_size 32 -b 0 -d mps
    # Process batch 3 (prompts 96-127):
    python experiments/trace.py -m gpt2-small -t ioi -n 256 --batch_size 32 -b 3 -d mps
    # Process all batches sequentially (useful locally):
    python experiments/trace.py -m gpt2-small -t ioi -n 256 --batch_size 32 -d mps

Equivalent old command (IOI, GPT-2):
    python tracing.py -m gpt2-small -t ioi -n 8 -d mps -s 0 -tt -at dynamic -o ig
"""

import argparse
import os
import sys
from collections import defaultdict
from copy import deepcopy

import networkx as nx
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm

# Add src/ to path so accpp_tracer is importable without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from accpp_tracer import Tracer
from accpp_tracer.datasets import (
    FactsDataset,
    GenderedPronoun,
    IOIDataset,
    YearDataset,
    get_valid_years,
)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "gpt2-small": {"family": "gpt2", "prepend_bos": False, "use_numpy_svd": False},
    "EleutherAI/pythia-160m": {"family": "pythia", "prepend_bos": False, "use_numpy_svd": True},
    "gemma-2-2b": {"family": "gemma", "prepend_bos": True, "use_numpy_svd": False},
}

SUPPORTED_MODELS = list(MODEL_CONFIGS.keys())
SUPPORTED_TASKS = ["ioi", "ioi-balanced", "gt", "gp", "facts"]

ROOT_NODE_LABELS = {
    "ioi": "IO-S direction",
    "ioi-balanced": "IO-S direction",
    "gt": "True YY - False YY",
    "gp": "Correct - Incorrect pronoun",
    "facts": "Correct answer",
}

# Typo ("pronoum") kept for backward-compat with existing traced graphs.
ROOT_NODE_LABELS_OLD_COMPAT = {
    "ioi": "IO-S direction",
    "ioi-balanced": "IO-S direction",
    "gt": "True YY - False YY",
    "gp": "Correct - Incorrect pronoum",
    "facts": "Correct answer",
}

# ---------------------------------------------------------------------------
# Task-specific: dataset creation
# ---------------------------------------------------------------------------


def create_dataset(task, model, model_cfg, num_prompts, seed, device):
    """Create the appropriate dataset for the given task.

    Args:
        task: Task name.
        model: HookedTransformer model.
        model_cfg: Entry from MODEL_CONFIGS.
        num_prompts: Number of prompts (ignored for gp and facts).
        seed: Random seed.
        device: Torch device.

    Returns:
        Dataset instance and the actual number of prompts.
    """
    family = model_cfg["family"]
    prepend_bos = model_cfg["prepend_bos"]

    if task == "ioi":
        dataset = IOIDataset(
            model_family=family,
            prompt_type="mixed",
            N=num_prompts,
            tokenizer=model.tokenizer,
            prepend_bos=prepend_bos,
            seed=seed,
            device=device,
        )
        return dataset, num_prompts

    elif task == "ioi-balanced":
        # Balanced IOI generates its own N (from gen_prompt_balanced).
        # The N parameter to IOIDataset is ignored when prompt_type="balanced".
        dataset = IOIDataset(
            model_family=family,
            prompt_type="balanced",
            tokenizer=model.tokenizer,
            prepend_bos=prepend_bos,
            seed=seed,
            device=device,
        )
        return dataset, dataset.N

    elif task == "gt":
        years_to_sample_from = get_valid_years(model.tokenizer, 1000, 1900)
        dataset = YearDataset(
            years_to_sample_from,
            num_prompts,
            model.tokenizer,
            balanced=True,
            device=device,
            eos=True,  # Following the original paper
            random_seed=seed,
        )
        return dataset, num_prompts

    elif task == "gp":
        # GenderedPronoun always generates exactly 100 examples.
        dataset = GenderedPronoun(
            model,
            model_family=family,
            device=device,
            prepend_bos=prepend_bos,
        )
        return dataset, 100

    elif task == "facts":
        # FactsDataset has exactly 6 fixed prompts.
        dataset = FactsDataset(
            model,
            prepend_bos=prepend_bos,
            device=device,
        )
        return dataset, 6

    else:
        raise ValueError(f"Unknown task: {task}")


# ---------------------------------------------------------------------------
# Task-specific: logit direction (f_W_U from original tracing.py)
# ---------------------------------------------------------------------------


def compute_logit_direction(task, prompt_id, model, dataset):
    """Compute the logit direction for a prompt.

    Extracted from the original tracing.py f_W_U() function.

    Args:
        task: Task name.
        prompt_id: Index of the prompt in the dataset.
        model: HookedTransformer model.
        dataset: Dataset instance.

    Returns:
        Logit direction tensor of shape (d_model,).
    """
    if task in ("ioi", "ioi-balanced"):
        # IO - S direction
        IO_token = dataset.toks[prompt_id, dataset.word_idx["IO"][prompt_id]]
        S_token = dataset.toks[prompt_id, dataset.word_idx["S1"][prompt_id]]
        return model.W_U[:, IO_token] - model.W_U[:, S_token]

    elif task == "gt":
        # Average(W_U for years > YY) - Average(W_U for years <= YY)
        YY_idx = dataset.possible_targets_toks.index(dataset.YY_toks[prompt_id])

        direction_neg = None
        direction_pos = None
        for i in range(len(dataset.possible_targets_toks)):
            target_tok = dataset.possible_targets_toks[i]
            if i <= YY_idx:
                if direction_neg is None:
                    direction_neg = deepcopy(-model.W_U[:, target_tok])
                else:
                    direction_neg -= model.W_U[:, target_tok]
            else:
                if direction_pos is None:
                    direction_pos = deepcopy(model.W_U[:, target_tok])
                else:
                    direction_pos += model.W_U[:, target_tok]

        direction_neg /= YY_idx + 1
        direction_pos /= len(dataset.possible_targets_toks) - YY_idx - 1
        return direction_pos + direction_neg

    elif task == "gp":
        # Correct pronoun - Incorrect pronoun
        correct_idx = dataset.answers[prompt_id]
        wrong_idx = dataset.wrongs[prompt_id]
        return model.W_U[:, correct_idx] - model.W_U[:, wrong_idx]

    elif task == "facts":
        # Correct answer direction (no contrastive token)
        correct_idx = dataset.answers_id[prompt_id]
        return model.W_U[:, correct_idx]

    else:
        raise ValueError(f"Unknown task: {task}")


# ---------------------------------------------------------------------------
# Generic: token position -> label mapping
# ---------------------------------------------------------------------------


def build_idx_to_token(prompt_id, model, dataset):
    """Build token position -> label mapping for a prompt.

    Uses actual tokens with duplicate numbering, matching the original
    tracing.py behavior with the -tt (trace_w_tokens) flag.

    Args:
        prompt_id: Index of the prompt in the dataset.
        model: HookedTransformer model.
        dataset: Dataset instance (must have .toks and .word_idx).

    Returns:
        Dict mapping token position (int) -> label (str).
    """
    idx_to_token: dict[int, str] = {}
    count_dict: dict[str, int] = defaultdict(int)
    end_pos = dataset.word_idx["end"][prompt_id].item()
    for i in range(end_pos + 1):
        tok_str = model.tokenizer.decode(dataset.toks[prompt_id, i])
        count = count_dict[tok_str]
        if count > 0:
            idx_to_token[i] = f"{tok_str} ({count})"
        else:
            idx_to_token[i] = tok_str
        count_dict[tok_str] += 1
    return idx_to_token


def build_idx_to_gram_roles(task, prompt_id, model, dataset):
    """Build token position -> gram role label mapping for a prompt.

    Matches the original tracing.py behavior WITHOUT the -tt flag:
    - IOI: Only grammatical role positions are mapped (IO, S1, S2, etc.).
      Unlabeled positions are excluded, so they won't be traced.
    - GT: All positions mapped to actual tokens, then gram role positions
      overwritten with role names.
    - GP: All positions mapped via gram roles (every position has a role).
    - facts: Not supported (requires -tt / actual token labels).

    Args:
        task: Task name.
        prompt_id: Index of the prompt in the dataset.
        model: HookedTransformer model.
        dataset: Dataset instance.

    Returns:
        Dict mapping token position (int) -> label (str).
    """
    if task in ("ioi", "ioi-balanced"):
        # Only map positions with grammatical roles
        idx_to_token: dict[int, str] = {}
        for gram_role in dataset.word_idx.keys():
            pos = dataset.word_idx[gram_role][prompt_id].item()
            idx_to_token[pos] = gram_role
        return idx_to_token

    elif task == "gt":
        # First map all positions to actual tokens
        idx_to_token = build_idx_to_token(prompt_id, model, dataset)
        # Then overwrite with gram role names
        for gram_role in dataset.word_idx.keys():
            pos = dataset.word_idx[gram_role][prompt_id].item()
            idx_to_token[pos] = gram_role
        return idx_to_token

    elif task == "gp":
        # Every position has a gram role
        idx_to_token = {}
        for gram_role in dataset.word_idx.keys():
            pos = dataset.word_idx[gram_role][prompt_id].item()
            idx_to_token[pos] = gram_role
        return idx_to_token

    else:
        raise ValueError(
            f"Gram role labels not supported for task '{task}'. Use --label_mode tokens."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce ACC++ circuit tracing from the paper"
    )
    parser.add_argument(
        "-m", "--model_name",
        choices=SUPPORTED_MODELS,
        required=True,
    )
    parser.add_argument(
        "-t", "--task",
        choices=SUPPORTED_TASKS,
        required=True,
    )
    parser.add_argument("-n", "--num_prompts", type=int, default=8)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument(
        "-at", "--attn_weight_thresh", type=str, default="dynamic",
        help="'dynamic' or a float in [0, 1]",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default=None,
        help="Directory to save graphs (default: traced_graphs/<model>/<task>)",
    )
    parser.add_argument(
        "--label_mode", type=str, default="tokens",
        choices=["tokens", "roles"],
        help=(
            "'tokens' = actual decoded tokens (equivalent to old -tt flag), "
            "'roles' = grammatical role labels (equivalent to old default). "
            "Use 'roles' for backward-compat comparison with old traced_graphs."
        ),
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help=(
            "Number of prompts per batch for the forward pass. "
            "If not set, all prompts are processed at once. "
            "Use with -b to select a specific batch (qsub-friendly)."
        ),
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=None,
        help=(
            "Which batch to process (0-indexed). Requires --batch_size. "
            "If --batch_size is set but -b is not, all batches run sequentially."
        ),
    )
    args = parser.parse_args()

    # Validate threshold
    if args.attn_weight_thresh != "dynamic":
        thresh = float(args.attn_weight_thresh)
        if not 0.0 <= thresh <= 1.0:
            raise ValueError("attn_weight_thresh must be 'dynamic' or between 0 and 1.")

    # Validate label_mode + task combination
    if args.label_mode == "roles" and args.task == "facts":
        raise ValueError("Gram role labels not supported for facts. Use --label_mode tokens.")

    # Validate batch arguments
    if args.batch is not None and args.batch_size is None:
        raise ValueError("--batch requires --batch_size to be set.")

    model_cfg = MODEL_CONFIGS[args.model_name]
    model_short = args.model_name.split("/")[-1]

    # Task-specific constraints
    if args.task == "gp" and args.num_prompts != 100:
        print("Note: GenderedPronoun always generates 100 examples. Ignoring --num_prompts.")
    if args.task == "facts":
        print("Note: FactsDataset has 6 fixed prompts. Ignoring --num_prompts.")

    torch.set_grad_enabled(False)

    # --- Load model ---
    print(f"Loading {args.model_name}...")
    model = HookedTransformer.from_pretrained(args.model_name, device=args.device)

    # --- Create dataset ---
    print(f"Creating dataset for task '{args.task}'...")
    dataset, num_prompts = create_dataset(
        args.task, model, model_cfg, args.num_prompts, args.seed, args.device,
    )
    print(f"  {num_prompts} prompts")

    # --- Determine batch ranges ---
    if args.batch_size is not None:
        batch_size = args.batch_size
        n_batches = (num_prompts + batch_size - 1) // batch_size  # ceiling division
        if args.batch is not None:
            if args.batch >= n_batches:
                raise ValueError(
                    f"--batch {args.batch} is out of range "
                    f"(n_batches={n_batches} for {num_prompts} prompts "
                    f"with batch_size={batch_size})"
                )
            batch_indices = [args.batch]
        else:
            batch_indices = list(range(n_batches))
        print(
            f"  batch_size={batch_size}, n_batches={n_batches}, "
            f"processing batch(es): {batch_indices}"
        )
    else:
        # No batching: single batch covering all prompts
        batch_size = num_prompts
        n_batches = 1
        batch_indices = [0]

    # --- Initialize tracer ---
    print("Initializing Tracer (precomputing Omega SVD + pseudoinverses)...")
    tracer = Tracer(
        model,
        device=args.device,
        use_numpy_svd=model_cfg["use_numpy_svd"],
    )

    # --- Output directory ---
    output_dir = args.output_dir or f"traced_graphs/{model_short}/{args.task}"
    os.makedirs(output_dir, exist_ok=True)

    # --- Root node label ---
    if args.label_mode == "roles":
        root_labels = ROOT_NODE_LABELS_OLD_COMPAT
    else:
        root_labels = ROOT_NODE_LABELS
    root_label = root_labels[args.task]

    # --- Process each batch ---
    for batch_idx in batch_indices:
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_prompts)
        batch_range = range(batch_start, batch_end)

        print(f"\n--- Batch {batch_idx}/{n_batches}: prompts {batch_start}-{batch_end - 1} ---")

        # Forward pass on this batch only
        print(f"Running forward pass on {len(batch_range)} prompts...")
        logits, cache = model.run_with_cache(dataset.toks[batch_range])

        # Filter incorrect predictions (using batch-local logits)
        skip = set()
        if args.task in ("ioi", "ioi-balanced"):
            for local_idx, pid in enumerate(batch_range):
                end_pos = dataset.word_idx["end"][pid].item()
                io_token = dataset.toks[pid, dataset.word_idx["IO"][pid].item()]
                if logits[local_idx, end_pos, :].argmax() != io_token:
                    skip.add(pid)
        elif args.task == "gt":
            for local_idx, pid in enumerate(batch_range):
                end_pos = dataset.word_idx["end"][pid].item()
                predicted_tok = logits[local_idx, end_pos, :].argmax().item()
                if predicted_tok not in dataset.possible_targets_toks:
                    skip.add(pid)
                    continue
                prediction_idx = dataset.possible_targets_toks.index(predicted_tok)
                YY_idx = dataset.possible_targets_toks.index(dataset.YY_toks[pid])
                if prediction_idx <= YY_idx:
                    skip.add(pid)
        elif args.task == "gp":
            for local_idx, pid in enumerate(batch_range):
                end_pos = dataset.word_idx["end"][pid].item()
                if logits[local_idx, end_pos, :].argmax().item() != dataset.answers[pid]:
                    skip.add(pid)

        if skip:
            print(f"Skipping {len(skip)}/{len(batch_range)} prompts (incorrect predictions)")

        # Free logits — only the cache is needed for tracing
        del logits

        # Trace each prompt in this batch
        for local_idx, prompt_id in enumerate(
            tqdm(batch_range, desc=f"Tracing batch {batch_idx}")
        ):
            if prompt_id in skip:
                continue

            end_token_pos = dataset.word_idx["end"][prompt_id].item()
            logit_direction = compute_logit_direction(
                args.task, prompt_id, model, dataset
            )
            if args.label_mode == "roles":
                idx_to_token = build_idx_to_gram_roles(
                    args.task, prompt_id, model, dataset
                )
            else:
                idx_to_token = build_idx_to_token(prompt_id, model, dataset)
            root_node = (root_label, idx_to_token[end_token_pos])

            graph = tracer.trace_from_cache(
                cache=cache,
                logit_direction=logit_direction,
                end_token_pos=end_token_pos,
                idx_to_token=idx_to_token,
                root_node=root_node,
                prompt_idx=local_idx,
                attn_weight_thresh=args.attn_weight_thresh,
            )

            # Skip empty graphs (no seeds or no AH seeds) — matches old behavior
            if graph.number_of_nodes() == 0:
                continue

            # Save graph — filename uses global num_prompts and prompt_id
            filename = (
                f"{model_short}_{args.task}_n{num_prompts}_{prompt_id}"
                f"_{args.seed}_{args.attn_weight_thresh}_ig.graphml"
            )
            nx.write_graphml(graph, os.path.join(output_dir, filename))
            print(
                f"  Prompt {prompt_id}: "
                f"{graph.number_of_nodes()} nodes, "
                f"{graph.number_of_edges()} edges"
            )

        # Free cache before next batch
        del cache

    print(f"\nDone. Graphs saved to {output_dir}/")


if __name__ == "__main__":
    main()
