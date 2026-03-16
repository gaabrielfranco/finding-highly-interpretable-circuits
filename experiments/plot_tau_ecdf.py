"""Plot ECDF of attention_weight * context_size for finding tau (Appendix D).

For each model-task combination, collects all attention weights A_ds multiplied
by their row context size d, and plots the empirical CDF. The dynamic threshold
tau = 2.5 / d corresponds to the vertical line at d * A_ds = 2.5.

Usage:
    python experiments/plot_tau_ecdf.py -m gpt2-small -d mps
    python experiments/plot_tau_ecdf.py -m EleutherAI/pythia-160m -d mps
    python experiments/plot_tau_ecdf.py -m gemma-2-2b -d mps
"""

import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

# Add src/ to path so accpp_tracer is importable without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from accpp_tracer.datasets import (
    GenderedPronoun,
    IOIDataset,
    YearDataset,
    get_valid_years,
)

torch.set_grad_enabled(False)

SUPPORTED_MODELS = [
    "gpt2-small",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-160m-deduped",
    "gemma-2-2b",
]

# (task, num_prompts) — matches the paper experiments
TASKS = [
    ("ioi", 256),
    ("gt", 256),
    ("gp", 100),
]

MODEL_CONFIGS = {
    "gpt2-small": {"family": "gpt2", "prepend_bos": False},
    "EleutherAI/pythia-160m": {"family": "pythia", "prepend_bos": False},
    "EleutherAI/pythia-160m-deduped": {"family": "pythia", "prepend_bos": False},
    "gemma-2-2b": {"family": "gemma", "prepend_bos": True},
}


def create_dataset(task, num_prompts, model, model_cfg, seed, device):
    """Create dataset for the given task."""
    family = model_cfg["family"]
    prepend_bos = model_cfg["prepend_bos"]

    if task == "ioi":
        return IOIDataset(
            model_family=family,
            prompt_type="mixed",
            N=num_prompts,
            tokenizer=model.tokenizer,
            prepend_bos=prepend_bos,
            seed=seed,
            device=device,
        )
    elif task == "gt":
        if family == "gemma":
            return None  # GT not run for Gemma in the paper
        years_to_sample_from = get_valid_years(model.tokenizer, 1000, 1900)
        return YearDataset(
            years_to_sample_from,
            num_prompts,
            model.tokenizer,
            balanced=True,
            device=device,
            eos=True,
            random_seed=seed,
        )
    elif task == "gp":
        return GenderedPronoun(
            model,
            model_family=family,
            device=device,
            prepend_bos=prepend_bos,
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot ECDF of attention_weight * context_size (Appendix D)"
    )
    parser.add_argument(
        "-m", "--model_name", choices=SUPPORTED_MODELS, required=True,
    )
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument(
        "-o", "--output_dir", type=str,
        default="figures/attention-scores-distribution",
    )
    args = parser.parse_args()

    model_short = args.model_name.split("/")[-1]

    print(f"Loading {args.model_name}...")
    model = HookedTransformer.from_pretrained(args.model_name, device=args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    for task, num_prompts in TASKS:
        print(f"\n--- {task} (n={num_prompts}) ---")

        dataset = create_dataset(
            task, num_prompts, model, MODEL_CONFIGS[args.model_name],
            args.seed, args.device,
        )
        if dataset is None:
            print(f"  Skipping {task} for {model_short}")
            continue

        _, cache = model.run_with_cache(dataset.toks)

        # Collect attention_weight * context_size for all layers/heads/tokens
        all_weighted_attn = []
        for layer in tqdm(range(model.cfg.n_layers), desc=f"  Collecting {task}"):
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
            for prompt_id in range(len(dataset)):
                end_pos = dataset.word_idx["end"][prompt_id]
                for tok_id in range(end_pos + 1):
                    # A_ds for all heads at this (prompt, dest_token), up to causal mask
                    row = pattern[prompt_id, :, tok_id, :tok_id + 1]
                    context_size = tok_id + 1
                    all_weighted_attn.extend(
                        (row.reshape(-1) * context_size).tolist()
                    )

        del cache

        # Plot ECDF using sorted values
        data = np.sort(all_weighted_attn)
        ecdf_y = np.arange(1, len(data) + 1) / len(data)

        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        plt.rc("font", size=6)

        fig, ax = plt.subplots(1, 1, figsize=(2, 1.5))
        ax.plot(data, ecdf_y)
        ax.axvline(2.5, linestyle="--", color="black")
        ax.set_xlabel("Attention weight * context size")
        ax.set_ylabel("Cumulative probability")

        path = os.path.join(
            args.output_dir, f"{model_short}_{task}_{num_prompts}.pdf"
        )
        plt.savefig(path, bbox_inches="tight", dpi=800)
        plt.close()
        print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
