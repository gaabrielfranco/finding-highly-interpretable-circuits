"""Plot condition number heatmaps for W_Q and W_K (Appendix B).

Computes the condition number (ratio of largest to smallest singular value)
of W_Q and W_K across all layers and attention heads, and saves heatmap PDFs.

Usage:
    python experiments/plot_condition_numbers.py -m gpt2-small -d mps
    python experiments/plot_condition_numbers.py -m EleutherAI/pythia-160m -d mps
    python experiments/plot_condition_numbers.py -m gemma-2-2b -d mps
"""

import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

SUPPORTED_MODELS = [
    "gpt2-small",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-160m-deduped",
    "gemma-2-2b",
]


def main():
    parser = argparse.ArgumentParser(
        description="Plot condition number heatmaps for W_Q and W_K (Appendix B)"
    )
    parser.add_argument(
        "-m", "--model_name", choices=SUPPORTED_MODELS, required=True,
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cpu",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="figures/condition-numbers",
        help="Directory to save PDF plots (default: figures/condition-numbers)",
    )
    args = parser.parse_args()

    model_short = args.model_name.split("/")[-1]

    print(f"Loading {args.model_name}...")
    model = HookedTransformer.from_pretrained(args.model_name, device=args.device)

    # Compute singular values of W_Q and W_K
    _, S_Q, _ = torch.linalg.svd(model.W_Q)
    _, S_K, _ = torch.linalg.svd(model.W_K.transpose(2, 3))

    # Condition number = largest / smallest singular value
    cond_Q = (S_Q[:, :, 0] / S_Q[:, :, -1]).cpu().numpy()
    cond_K = (S_K[:, :, 0] / S_K[:, :, -1]).cpu().numpy()

    os.makedirs(args.output_dir, exist_ok=True)

    # Publication-quality settings
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rc("font", size=8)

    # W_Q condition numbers
    fig, ax = plt.subplots(1, 1, figsize=(2, 1.5))
    sns.heatmap(cond_Q, ax=ax)
    ax.set_ylabel("Layer")
    ax.set_xlabel("AH idx")
    path_q = os.path.join(args.output_dir, f"{model_short}_W_Q_condition_numbers.pdf")
    plt.savefig(path_q, bbox_inches="tight", dpi=800)
    plt.close()
    print(f"Saved: {path_q}")

    # W_K condition numbers
    fig, ax = plt.subplots(1, 1, figsize=(2, 1.5))
    sns.heatmap(cond_K, ax=ax)
    ax.set_ylabel("Layer")
    ax.set_xlabel("AH idx")
    path_k = os.path.join(args.output_dir, f"{model_short}_W_K_condition_numbers.pdf")
    plt.savefig(path_k, bbox_inches="tight", dpi=800)
    plt.close()
    print(f"Saved: {path_k}")


if __name__ == "__main__":
    main()
