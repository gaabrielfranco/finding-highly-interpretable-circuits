"""Stage 2: Generate signal interpretations using an explainer LLM (vLLM).

For each signal in the top-K activation H5 file (from Stage 1), builds a
prompt showing the highlighted source/destination tokens in context, then
queries the explainer LLM (DeepSeek-R1-Distill-Llama-70B by default) via
vLLM to generate a natural-language interpretation.

Writes results to a shard H5 file for parallel execution. Use
``merge_shards.py`` (Stage 2b) to merge all shards back into the main
activation H5 after all shards are complete.

Usage:
    python interpret_signals.py \\
        --model gpt2-small \\
        --layer 5 \\
        --start 0 \\
        --end 5000 \\
        --activations_file data/autointerp/activations_5_ioi-balanced_gpt2-small.h5 \\
        --output_dir data/autointerp/shards/ \\
        --explainer_model neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic \\
        --tensor_parallel_size 2 \\
        --batch_size 128 \\
        --max_model_len 8192 \\
        --temperature 0.6 \\
        --max_tokens 4096 \\
        --device cuda \\
        --hf_cache_dir ~/.cache/huggingface
"""

import argparse
import gc
import logging
import os
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import ray
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, utils
from vllm import LLM, SamplingParams

from prompts import build_explainer_prompt, extract_interpretation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_SEED = 42


def generate_interpretations(
    layer: int,
    start: int,
    end: int,
    activations_file: str,
    output_dir: str,
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer_explainer: Any,
    tokenizer_gpt2: Any,
    dataset_the_pile: Any,
    batch_size: int,
    debug_limit: int | None,
) -> str:
    """Generate LLM interpretations for one shard of signals.

    Reads top-K activation indices from ``activations_file`` in the range
    ``[start, end)``, builds one prompt per signal, calls the explainer LLM,
    parses the ``[interpretation]:`` line from each response, and writes
    results to a shard H5 file. The shard covers only the requested slice;
    use ``merge_shards.py`` to merge all shards into the main H5.

    Args:
        layer: Layer index (selects group ``layer_{layer}`` in H5 files).
        start: First signal index (global, inclusive).
        end: Last signal index (global, exclusive). Clamped to the total
            number of signals in the layer.
        activations_file: Path to the top-K activation H5 file produced by
            Stage 1 (``extract_top_activations.py``).
        output_dir: Directory where the shard H5 file is written.
        llm: Initialized vLLM ``LLM`` instance.
        sampling_params: vLLM ``SamplingParams`` for generation.
        tokenizer_explainer: Tokenizer for the explainer LLM; used to apply
            the chat template in ``build_explainer_prompt``.
        tokenizer_gpt2: GPT-2 tokenizer used to decode Pile token IDs into
            text for prompt construction.
        dataset_the_pile: Tokenized ``NeelNanda/pile-10k`` dataset
            (``max_length=32``, ``add_bos_token=False``).
        batch_size: Number of signals to process per vLLM call.
        debug_limit: If set, stop after processing this many signals total.
            ``None`` means no limit.

    Returns:
        Path to the output shard H5 file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    layer_name = f"layer_{layer}"
    out_file = str(Path(output_dir) / f"activations_{layer}_balanced_{start}_{end}.h5")

    log.info(f"Shard output: {out_file}")

    # Read total_features once (READ-ONLY)
    with h5py.File(activations_file, "r") as f:
        if layer_name not in f:
            raise KeyError(f"{layer_name} not found in {activations_file}")
        total_features = f[layer_name]["topk_indices"].shape[0]

    if start >= total_features:
        log.info(f"No work: start={start} >= total_features={total_features}")
        return out_file

    # Clamp end
    end_global = min(end, total_features)
    if end_global <= start:
        log.info(f"No work: end={end_global} <= start={start}")
        return out_file

    shard_len = end_global - start

    # Create shard file structure once
    with h5py.File(out_file, "a") as f_out:
        grp = f_out.require_group(layer_name)
        dt = h5py.string_dtype(encoding="utf-8")

        if "topk_interpretations" not in grp:
            grp.create_dataset("topk_interpretations", (shard_len,), dtype=dt)
        if "topk_raw_answers" not in grp:
            grp.create_dataset("topk_raw_answers", (shard_len,), dtype=dt)

        grp.attrs["global_start"] = int(start)
        grp.attrs["global_end"] = int(end_global)

    processed_count = 0
    _debug_limit = debug_limit if debug_limit is not None else int(1e9)

    for start_index in range(start, end_global, batch_size):
        remaining_quota = _debug_limit - processed_count
        current_batch_size = min(batch_size, end_global - start_index, remaining_quota)
        if current_batch_size <= 0:
            break

        end_index = start_index + current_batch_size
        log.info(f"Processing {layer_name}: global {start_index} to {end_index}...")

        # PHASE A: READ (READ-ONLY)
        with h5py.File(activations_file, "r") as f:
            indices = f[layer_name]["topk_indices"][start_index:end_index]

        # PHASE B: COMPUTE
        prompts = [
            build_explainer_prompt(
                indices[i], tokenizer_gpt2, dataset_the_pile, tokenizer_explainer
            )
            for i in range(len(indices))
        ]
        outputs = llm.generate(prompts, sampling_params)

        batch_interpretations: list[str] = []
        batch_raw_answers: list[str] = []
        for out_obj in outputs:
            raw_text = out_obj.outputs[0].text
            clean = extract_interpretation(raw_text)
            batch_interpretations.append(clean if clean is not None else "")
            batch_raw_answers.append(raw_text)

        # PHASE C: WRITE (to shard; local indexing)
        local_start = start_index - start
        local_end = end_index - start

        with h5py.File(out_file, "r+") as f_out:
            grp = f_out[layer_name]
            grp["topk_interpretations"][local_start:local_end] = batch_interpretations
            grp["topk_raw_answers"][local_start:local_end] = batch_raw_answers

        processed_count += len(batch_interpretations)

        del outputs, prompts, indices
        gc.collect()

    log.info(f"Shard complete. Processed {processed_count} signals.")
    return out_file


def interpret_signals(
    model_name: str,
    layer: int,
    start: int,
    end: int,
    activations_file: str,
    output_dir: str,
    explainer_model: str,
    tensor_parallel_size: int,
    batch_size: int,
    max_model_len: int,
    temperature: float,
    max_tokens: int,
    debug_limit: int | None = None,
    device: str = "cuda",
    hf_cache_dir: str | None = None,
) -> str:
    """Run Stage 2: interpret signal activations using a vLLM explainer LLM.

    Loads the model tokenizer (for Pile token decoding), the Pile dataset,
    and the explainer LLM via vLLM, then calls ``generate_interpretations``
    to produce a shard H5 file with interpretations.

    Args:
        model_name: TransformerLens model name (e.g. ``"gpt2-small"``).
            Used only to load the tokenizer for Pile sentence decoding.
        layer: Layer index to process.
        start: First signal index (global, inclusive).
        end: Last signal index (global, exclusive).
        activations_file: Path to the top-K activation H5 (Stage 1 output).
        output_dir: Directory for shard output files.
        explainer_model: HuggingFace model ID for the explainer LLM
            (e.g. ``"neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic"``).
        tensor_parallel_size: Number of GPUs for vLLM tensor parallelism.
        batch_size: Number of signals per vLLM generation call.
        max_model_len: Maximum sequence length for vLLM context window.
        temperature: Sampling temperature for vLLM generation.
        max_tokens: Maximum new tokens per generation.
        debug_limit: Optional hard cap on total signals processed (for
            testing). ``None`` means no limit.
        device: Torch device string. Accepted for API consistency; the TL
            model is always loaded on CPU (tokenizer only, no inference).
        hf_cache_dir: Optional HuggingFace cache directory. Sets
            ``HF_HOME`` and is passed as ``download_dir`` to vLLM.

    Returns:
        Path to the output shard H5 file.
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

    log.info(f"Loading explainer tokenizer: {explainer_model}")
    tokenizer_explainer = AutoTokenizer.from_pretrained(
        explainer_model,
        cache_dir=hf_cache_dir,
    )

    log.info(f"Initializing vLLM: {explainer_model}")
    llm = LLM(
        model=explainer_model,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=max_model_len,
        download_dir=hf_cache_dir,
        seed=_SEED,
        distributed_executor_backend="ray",
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["\n\nText examples"],
        seed=_SEED,
    )

    out_file = generate_interpretations(
        layer=layer,
        start=start,
        end=end,
        activations_file=activations_file,
        output_dir=output_dir,
        llm=llm,
        sampling_params=sampling_params,
        tokenizer_explainer=tokenizer_explainer,
        tokenizer_gpt2=tokenizer_gpt2,
        dataset_the_pile=dataset_the_pile,
        batch_size=batch_size,
        debug_limit=debug_limit,
    )

    log.info("Script finished!")
    ray.shutdown()
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: Generate signal interpretations using an explainer LLM (vLLM)."
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="TransformerLens model name (e.g. gpt2-small). Used for tokenizer only.",
    )
    parser.add_argument("--layer", "-l", type=int, required=True,
                        help="Layer index to process.")
    parser.add_argument("--start", "-s", type=int, required=True,
                        help="First signal index (inclusive).")
    parser.add_argument("--end", "-e", type=int, required=True,
                        help="Last signal index (exclusive). Clamped to total signals.")
    parser.add_argument(
        "--activations_file",
        required=True,
        help="Path to the top-K activation H5 file (Stage 1 output).",
    )
    parser.add_argument(
        "--output_dir",
        default="data/autointerp/shards/",
        help="Directory for shard output files (default: data/autointerp/shards/).",
    )
    parser.add_argument(
        "--explainer_model",
        default="neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic",
        help="HuggingFace model ID for the explainer LLM.",
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
        default=128,
        help="Number of signals per vLLM generation call (default: 128).",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Maximum sequence length for vLLM context window (default: 8192).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for vLLM generation (default: 0.6).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum new tokens per generation (default: 4096).",
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

    interpret_signals(
        model_name=args.model,
        layer=args.layer,
        start=args.start,
        end=args.end,
        activations_file=args.activations_file,
        output_dir=args.output_dir,
        explainer_model=args.explainer_model,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        debug_limit=args.debug_limit,
        device=args.device,
        hf_cache_dir=args.hf_cache_dir,
    )


if __name__ == "__main__":
    main()
