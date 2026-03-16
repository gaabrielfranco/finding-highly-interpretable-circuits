"""Stage 3: Score signal interpretations using a judge LLM (vLLM).

For each signal that has an interpretation (from Stage 2), builds mini-prompts
with positive (top-K) and negative (random) examples, queries the judge LLM
(Gemma-3-27b-it by default) to classify whether each example matches the
interpretation, and writes binary labels for precision/recall/F1 computation.

The judge prompt uses ``N_EXAMPLES`` total examples (positive + negative),
split into ``BATCH_SIZE // GROUP_SIZE`` mini-prompts of ``GROUP_SIZE * 2``
examples each. This smaller-context format makes the classification task
easier for the judge LLM.

Resumability: signals whose ``scoring_labels`` row is all zeros are treated
as unscored and will be processed. Already-scored signals are skipped.

Usage:
    python score_interpretations.py \\
        --model gpt2-small \\
        --layer 5 \\
        --activations_file data/autointerp/activations_5_ioi-balanced_gpt2-small.h5 \\
        --judge_model google/gemma-3-27b-it \\
        --tensor_parallel_size 2 \\
        --batch_size 256 \\
        --max_model_len 8192 \\
        --n_examples 20 \\
        --group_size 5 \\
        --device cuda \\
        --hf_cache_dir ~/.cache/huggingface
"""

import argparse
import gc
import logging
import os
import random
from typing import Any

import h5py
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, utils
from vllm import LLM, SamplingParams

from prompts import build_judge_prompt, parse_judge_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_SEED = 42


# ---------------------------------------------------------------------------
# Core scoring loop
# ---------------------------------------------------------------------------

def score_signals(
    layer: int,
    activations_file: str,
    llm_judge: LLM,
    sampling_params_judge: SamplingParams,
    tokenizer_judge: Any,
    tokenizer_gpt2: Any,
    dataset_the_pile: Any,
    batch_size: int,
    n_examples: int,
    group_size: int,
    debug_limit: int | None,
) -> None:
    """Score interpretations for one layer using the judge LLM.

    Reads ``topk_indices``, ``random_indices``, and ``topk_interpretations``
    from ``activations_file``, builds mini-prompts for the judge LLM, and
    writes ``scoring_labels`` (ground truth) and ``scoring_labels_pred``
    (judge predictions) back to the same file.

    For each signal, ``n_examples`` positive (top-K) and ``n_examples``
    negative (random) examples are split into ``n_examples // group_size``
    mini-prompts of ``group_size * 2`` examples each. Labels are
    concatenated across mini-prompts → shape ``(n_examples * 2,)`` per
    signal.

    Signals with an empty interpretation are assigned all-``-1`` labels
    (skipped). The resumability scan finds the first row in
    ``scoring_labels`` that is all zeros and starts processing from there.

    Args:
        layer: Layer index (selects group ``layer_{layer}`` in H5 file).
        activations_file: Path to the activation H5 file (Stage 1 + 2
            output, containing ``topk_interpretations``).
        llm_judge: Initialized vLLM ``LLM`` instance for the judge model.
        sampling_params_judge: vLLM ``SamplingParams`` for judge inference.
        tokenizer_judge: Tokenizer for the judge LLM; used to apply the
            chat template in ``build_judge_prompt``.
        tokenizer_gpt2: GPT-2 tokenizer used to decode Pile token IDs into
            text for prompt construction.
        dataset_the_pile: Tokenized ``NeelNanda/pile-10k`` dataset
            (``max_length=32``, ``add_bos_token=False``).
        batch_size: Number of signals to process per vLLM call.
        n_examples: Total positive (and negative) examples per signal.
            The first ``n_examples`` entries from ``topk_indices`` and
            ``random_indices`` are used.
        group_size: Examples per mini-prompt (positive + negative combined
            = ``group_size * 2``). Must divide ``n_examples`` evenly.
        debug_limit: If set, stop after processing this many signals total.
            ``None`` means no limit.
    """
    layer_name = f"layer_{layer}"
    n_labels = n_examples * 2          # total labels per signal
    n_elements_prompt = group_size * 2  # labels per mini-prompt

    _debug_limit = debug_limit if debug_limit is not None else int(1e9)

    with h5py.File(activations_file, "r+") as f:
        grp = f[layer_name]
        total_features = grp["topk_indices"].shape[0]

        # Create scoring datasets once if they don't exist
        if "scoring_labels" not in grp:
            grp.create_dataset(
                "scoring_labels", (total_features, n_labels), dtype=np.float32
            )
        if "scoring_labels_pred" not in grp:
            grp.create_dataset(
                "scoring_labels_pred", (total_features, n_labels), dtype=np.float32
            )

    # Resumability: find first all-zero row (= unscored signal)
    with h5py.File(activations_file, "r") as f:
        scoring_labels = f[layer_name]["scoring_labels"][:]
    start = 0
    while start < len(scoring_labels):
        if (scoring_labels[start] == 0.0).all():
            break
        start += 1

    log.info(f"Starting from index {start} (out of {total_features})")

    processed_count = 0

    for start_index in range(start, total_features, batch_size):
        if processed_count >= _debug_limit:
            log.info("Debug limit reached. Stopping.")
            break

        remaining_quota = _debug_limit - processed_count
        current_batch_size = min(
            batch_size, total_features - start_index, remaining_quota
        )
        end_index = start_index + current_batch_size

        if current_batch_size <= 0:
            break

        log.info(f"Processing {layer_name}: indices {start_index} to {end_index}...")

        with h5py.File(activations_file, "r+") as f:
            grp = f[layer_name]
            indices = grp["topk_indices"][start_index:end_index]
            indices_random = grp["random_indices"][start_index:end_index]
            interpretations = grp["topk_interpretations"][start_index:end_index]

        assert indices.shape[0] == indices_random.shape[0] == current_batch_size
        assert indices.shape[2] == 3  # (sentence_id, dest_token, src_token)

        prompts: list[str] = []
        labels: list[np.ndarray] = []

        for i in range(len(indices)):
            interp = interpretations[i]
            if isinstance(interp, bytes):
                interp = interp.decode("utf-8")
            else:
                interp = str(interp)

            curr_indices = indices[i][:n_examples]         # (n_examples, 3)
            curr_indices_random = indices_random[i][:n_examples]  # (n_examples, 3)

            # Pre-shuffle at the signal level (before splitting into groups)
            idx_shuffle = list(range(len(curr_indices)))
            random.shuffle(idx_shuffle)
            curr_indices = curr_indices[idx_shuffle]
            curr_indices_random = curr_indices_random[idx_shuffle]

            # Split into mini-prompts of group_size positive + group_size negative
            for j in range(0, len(curr_indices), group_size):
                p, l = build_judge_prompt(
                    interp,
                    curr_indices[j:j + group_size],
                    curr_indices_random[j:j + group_size],
                    tokenizer_gpt2,
                    dataset_the_pile,
                    tokenizer_judge,
                    n_examples=group_size,
                )
                if interp == "":  # No interpretation → skip this signal
                    l = np.full(n_elements_prompt, -1)
                prompts.append(p)
                labels.append(l)

        # Run judge inference
        outputs = llm_judge.generate(prompts, sampling_params_judge)

        predicted_labels: list[np.ndarray] = []
        for out_obj in outputs:
            raw_text = out_obj.outputs[0].text
            predicted_labels.append(parse_judge_labels(raw_text, n_elements_prompt))

        predicted_labels_arr = np.array(predicted_labels).reshape(
            (current_batch_size, -1)
        )
        labels_arr = np.array(labels).reshape((current_batch_size, -1))

        assert predicted_labels_arr.shape == labels_arr.shape == (
            current_batch_size, n_labels
        )

        with h5py.File(activations_file, "r+") as f:
            grp = f[layer_name]
            grp["scoring_labels"][start_index:end_index] = labels_arr
            grp["scoring_labels_pred"][start_index:end_index] = predicted_labels_arr

        processed_count += current_batch_size

        del outputs, prompts, indices
        gc.collect()

    log.info(f"Layer {layer}: done. Processed {processed_count} signals.")


# ---------------------------------------------------------------------------
# Top-level function
# ---------------------------------------------------------------------------

def score_interpretations(
    model_name: str,
    layer: int,
    activations_file: str,
    judge_model: str,
    tensor_parallel_size: int,
    batch_size: int,
    max_model_len: int,
    n_examples: int,
    group_size: int,
    debug_limit: int | None = None,
    device: str = "cuda",
    hf_cache_dir: str | None = None,
) -> None:
    """Run Stage 3: score signal interpretations using a vLLM judge LLM.

    Loads the model tokenizer (for Pile token decoding), the Pile dataset,
    and the judge LLM via vLLM, then calls ``score_signals`` to produce
    ``scoring_labels`` and ``scoring_labels_pred`` datasets in the
    activation H5.

    Args:
        model_name: TransformerLens model name (e.g. ``"gpt2-small"``).
            Used only to load the tokenizer for Pile sentence decoding.
        layer: Layer index to process.
        activations_file: Path to the activation H5 file (Stage 1 + 2
            output, must contain ``topk_interpretations``).
        judge_model: HuggingFace model ID for the judge LLM
            (e.g. ``"google/gemma-3-27b-it"``).
        tensor_parallel_size: Number of GPUs for vLLM tensor parallelism.
        batch_size: Number of signals per vLLM generation call.
        max_model_len: Maximum sequence length for vLLM context window.
        n_examples: Total positive (and negative) examples per signal.
        group_size: Examples per mini-prompt half (positive or negative);
            each mini-prompt has ``group_size * 2`` examples total.
        debug_limit: Optional hard cap on total signals processed.
            ``None`` means no limit.
        device: Torch device string. Accepted for API consistency; the TL
            model is always loaded on CPU (tokenizer only, no inference).
        hf_cache_dir: Optional HuggingFace cache directory. Sets
            ``HF_HOME`` and is passed as ``download_dir`` to vLLM.
    """
    if hf_cache_dir is not None:
        os.environ["HF_HOME"] = hf_cache_dir

    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)
    torch.cuda.manual_seed_all(_SEED)

    log.info(f"Loading tokenizer for: {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device="cpu")
    tokenizer_gpt2 = model.tokenizer
    del model

    log.info("Loading NeelNanda/pile-10k")
    the_pile_data = load_dataset("NeelNanda/pile-10k", split="train")
    dataset_the_pile = utils.tokenize_and_concatenate(
        the_pile_data, tokenizer_gpt2, max_length=32, add_bos_token=False
    )

    log.info(f"Initializing vLLM judge: {judge_model}")
    llm_judge = LLM(
        model=judge_model,
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        download_dir=hf_cache_dir,
        seed=_SEED,
    )

    log.info(f"Loading judge tokenizer: {judge_model}")
    tokenizer_judge = AutoTokenizer.from_pretrained(
        judge_model,
        cache_dir=hf_cache_dir,
    )

    sampling_params_judge = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        top_p=1.0,
        stop=["\n\n\n"],
        seed=_SEED,
    )

    score_signals(
        layer=layer,
        activations_file=activations_file,
        llm_judge=llm_judge,
        sampling_params_judge=sampling_params_judge,
        tokenizer_judge=tokenizer_judge,
        tokenizer_gpt2=tokenizer_gpt2,
        dataset_the_pile=dataset_the_pile,
        batch_size=batch_size,
        n_examples=n_examples,
        group_size=group_size,
        debug_limit=debug_limit,
    )

    log.info("Script finished!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3: Score signal interpretations using a judge LLM (vLLM)."
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="TransformerLens model name (e.g. gpt2-small). Used for tokenizer only.",
    )
    parser.add_argument(
        "--layer", "-l",
        type=int,
        required=True,
        help="Layer index to process.",
    )
    parser.add_argument(
        "--activations_file",
        required=True,
        help="Path to the activation H5 file (Stage 1 + 2 output).",
    )
    parser.add_argument(
        "--judge_model",
        default="google/gemma-3-27b-it",
        help="HuggingFace model ID for the judge LLM (default: google/gemma-3-27b-it).",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=2,
        help="Number of GPUs for vLLM tensor parallelism (default: 2).",
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=256,
        help="Number of signals per vLLM generation call (default: 256).",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Maximum sequence length for vLLM context window (default: 8192).",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=20,
        help="Total positive (and negative) examples per signal (default: 20).",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=5,
        help=(
            "Examples per mini-prompt half (default: 5). Each mini-prompt has "
            "group_size positive + group_size negative = group_size*2 examples. "
            "n_examples must be divisible by group_size."
        ),
    )
    parser.add_argument(
        "--debug_limit",
        type=int,
        default=None,
        help="Stop after processing this many signals (for testing). Default: no limit.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device string (default: cuda). Accepted for API consistency.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        default=None,
        help="Optional HuggingFace cache directory (sets HF_HOME).",
    )
    args = parser.parse_args()

    score_interpretations(
        model_name=args.model,
        layer=args.layer,
        activations_file=args.activations_file,
        judge_model=args.judge_model,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        n_examples=args.n_examples,
        group_size=args.group_size,
        debug_limit=args.debug_limit,
        device=args.device,
        hf_cache_dir=args.hf_cache_dir,
    )


if __name__ == "__main__":
    main()
