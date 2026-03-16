"""Clustering plots from processed balanced IOI circuit components.

Generates per-model and combined cross-model plots:
- Template distribution bar charts (high-level ABBA/BABA, low-level 0-14)
- Prompts × Components heatmaps (binary membership)
- Clustermaps using Jaccard distance with hierarchical clustering
- Combined 3-model clustermaps (Figures 1, 22-23 in the paper)

This is Step 3 of the Section 3 / Appendix F pipeline.

Usage:
    # Per-model plots
    python experiments/plot_clustering.py -m gpt2-small
    python experiments/plot_clustering.py -m pythia-160m
    python experiments/plot_clustering.py -m gemma-2-2b

    # Combined 3-model clustermaps (requires all 3 models processed)
    python experiments/plot_clustering.py --combined
"""

import argparse
import os

import colorcet as cc
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hierarchy
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.spatial import distance

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
plt.rc("font", size=8)

MODELS = ["gpt2-small", "pythia-160m", "gemma-2-2b"]
MODEL_DISPLAY_NAMES = { 
    "gpt2-small": "GPT-2 Small",
    "pythia-160m": "Pythia-160M",
    "gemma-2-2b": "Gemma-2 2B",
}
COMPONENT_TYPES = ["head_as_component", "edge_as_component", "sv_as_component"]
VMIN_VALUES = {
    "head_as_component": 0.3,
    "edge_as_component": 0.55,
    "sv_as_component": 0.75,
}


# -----------------------------------------------------------------------
# Per-model plots (from clustering_code/plot_figures.py)
# -----------------------------------------------------------------------

def plot_high_level_template_distribution(df, model, folder):
    """Bar chart of ABBA vs BABA prompt counts."""
    dist = df["high_level_template"].value_counts()
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(6.4, 3.2)
    ax.bar(dist.index, dist.values, color="tab:blue")
    for i, count in enumerate(dist.values):
        ax.text(i, count + 0.5, str(count), color="black", ha="center")
    fig.tight_layout()
    fig.savefig(f"{folder}/{model}_high_level_template_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_low_level_template_distribution(df, model, folder):
    """Bar chart of low-level template counts, split by ABBA/BABA."""
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(6.4, 3.2)
    abba_data = df[df["high_level_template"] == "ABBA"]["low_level_template"].value_counts().sort_index()
    ax[0].bar(abba_data.index, abba_data.values, color="tab:blue")
    ax[0].set_xticks(np.arange(0, 15, 1))
    for i, count in enumerate(abba_data.values):
        ax[0].text(i, count + 0.5, str(count), color="black", ha="center")
    ax[0].set_title("ABBA")
    baba_data = df[df["high_level_template"] == "BABA"]["low_level_template"].value_counts().sort_index()
    ax[1].bar(baba_data.index, baba_data.values, color="tab:blue")
    ax[1].set_xticks(np.arange(0, 15, 1))
    for i, count in enumerate(baba_data.values):
        ax[1].text(i, count + 0.5, str(count), color="black", ha="center")
    ax[1].set_title("BABA")
    fig.tight_layout()
    fig.savefig(f"{folder}/{model}_low_level_template_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_prompts_by_components(df, model, components_type, folder):
    """Binary heatmap of prompts x components."""
    plot_df = df.sort_values(by=["high_level_template", "low_level_template"])
    cmatrix = np.stack(plot_df[components_type].to_numpy())
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(6.4, 3.2)
    sns.heatmap(cmatrix, cmap=ListedColormap(["#000000", "#1f77b4"]), cbar=False, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Components")
    ax.set_ylabel("Prompts")
    fig.tight_layout()
    fig.savefig(f"{folder}/{model}_prompts_by_components_{components_type}.png", dpi=1000, bbox_inches="tight")
    plt.close(fig)


def plot_clustermap(df, model, components_type, folder):
    """Clustermap using Jaccard distance with row/column color annotations."""
    cmatrix = np.stack(df[components_type].to_numpy())
    clustering = hierarchy.linkage(cmatrix, method="average", metric="jaccard")
    dmatrix = distance.pdist(cmatrix, metric="jaccard")
    colors = sns.color_palette(cc.glasbey, n_colors=15)
    vmin_val = VMIN_VALUES[components_type]

    g = sns.clustermap(
        distance.squareform(dmatrix),
        row_linkage=clustering,
        col_linkage=clustering,
        cmap="rocket_r",
        col_colors=["tab:blue" if a == "ABBA" else "tab:green" for a in df["high_level_template"]],
        row_colors=[colors[i] for i in df["low_level_template"]],
        yticklabels=False, xticklabels=False,
        cbar_pos=(0.4, -0.05, 0.4, 0.03),
        cbar_kws={"orientation": "horizontal"},
        colors_ratio=0.2, vmin=vmin_val,
        vmax=1,
    )
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    g.ax_cbar.set_xlabel("Jaccard Distance", fontsize=6, labelpad=2)
    g.fig.set_size_inches(2.1, 2.1)
    g.cax.tick_params(labelsize=6)
    g.fig.tight_layout()
    g.fig.savefig(f"{folder}/{model}_clustermap_{components_type}.png", dpi=800, bbox_inches="tight")
    plt.close(g.fig)


# -----------------------------------------------------------------------
# Combined 3-model clustermaps (from clustering_code/plot_clustermap_paper.py)
# -----------------------------------------------------------------------

def plot_combined_clustermaps(dfs, display_names, components_type, output_path):
    """Side-by-side clustermaps for all models using GridSpec layout."""
    vmin_val = VMIN_VALUES[components_type]
    vmax_val = 1

    n_models = len(display_names)
    fig = plt.figure(figsize=(6.4, 2.1))

    width_ratios = [0.5, 1, 0.05,
                    0.5, 1, 0.05,
                    0.5, 1]
    height_ratios = [0.5, 1]

    gs = gridspec.GridSpec(2, 8, width_ratios=width_ratios, height_ratios=height_ratios, wspace=0., hspace=0.)

    colors_palette = sns.color_palette(cc.glasbey, n_colors=15)

    for idx, (df, display_name) in enumerate(zip(dfs, display_names)):

        # Clustering
        cmatrix = np.stack(df[components_type].to_numpy())
        dist_array = distance.pdist(cmatrix, metric="jaccard")
        linkage = hierarchy.linkage(dist_array, method="average")
        dendro = hierarchy.dendrogram(linkage, no_plot=True)
        reorder_idx = dendro["leaves"]
        dmatrix_square = distance.squareform(dist_array)
        dmatrix_sorted = dmatrix_square[reorder_idx, :][:, reorder_idx]

        # Color bars
        hl_colors = ["tab:blue" if df["high_level_template"].iloc[i] == "ABBA" else "tab:green"
                     for i in reorder_idx]
        hl_rgb = [mcolors.to_rgb(c) for c in hl_colors]
        hl_img = np.array([hl_rgb])

        ll_indices = [df["low_level_template"].iloc[i] for i in reorder_idx]
        ll_colors = [colors_palette[i] for i in ll_indices]
        ll_img = np.array([ll_colors]).transpose(1, 0, 2)

        col_idx_base = idx * 3

        # Top bar
        ax_col_colors = fig.add_subplot(gs[0, col_idx_base + 1])
        ax_col_colors.imshow(hl_img, aspect="auto")
        ax_col_colors.set_axis_off()
        ax_col_colors.set_title(display_name, fontsize=8, pad=3)

        # Left bar
        ax_row_colors = fig.add_subplot(gs[1, col_idx_base])
        ax_row_colors.imshow(ll_img, aspect="auto")
        ax_row_colors.set_axis_off()

        # Main heatmap
        ax_heatmap = fig.add_subplot(gs[1, col_idx_base + 1])

        show_cbar = (idx == n_models - 1)
        cbar_ax = None

        if show_cbar:
            cbar_ax = ax_heatmap.inset_axes([1.05, 0.2, 0.05, 1.])

        sns.heatmap(
            dmatrix_sorted,
            ax=ax_heatmap,
            cmap="rocket_r",
            vmin=vmin_val, vmax=vmax_val,
            cbar=show_cbar,
            cbar_ax=cbar_ax,
            xticklabels=False, yticklabels=False,
        )

        ax_heatmap.set_xlabel("")
        ax_heatmap.set_ylabel("")

        if show_cbar:
            cbar_ax.set_ylabel("Jaccard Distance", fontsize=6)
            cbar_ax.tick_params(labelsize=6, width=0.2, length=1, pad=2)

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=800)
    plt.close()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clustering plots from processed components.")
    parser.add_argument(
        "-m", "--model", choices=MODELS,
        help="Model name (for per-model plots).",
    )
    parser.add_argument(
        "--combined", action="store_true",
        help="Generate combined 3-model clustermaps (requires all models processed).",
    )
    parser.add_argument(
        "--data_dir", default="data/clustering",
        help="Base directory for processed data. Default: data/clustering/",
    )
    parser.add_argument(
        "--output_dir", default="figures/clustering",
        help="Output directory for plots. Default: figures/clustering/",
    )
    args = parser.parse_args()

    if not args.model and not args.combined:
        parser.error("Specify -m MODEL for per-model plots, --combined for 3-model plots, or both.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Per-model plots
    if args.model:
        parquet_path = f"{args.data_dir}/{args.model}/processed_components.parquet"
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df)} prompts from {parquet_path}")

        print(f"  Generating template distributions...")
        plot_high_level_template_distribution(df, args.model, args.output_dir)
        plot_low_level_template_distribution(df, args.model, args.output_dir)

        for ct in COMPONENT_TYPES:
            print(f"  Generating {ct} plots...")
            plot_prompts_by_components(df, args.model, ct, args.output_dir)
            plot_clustermap(df, args.model, ct, args.output_dir)

        print(f"  Per-model plots saved to {args.output_dir}/")

    # Combined 3-model clustermaps
    if args.combined:
        dfs = []
        display_names = []
        for model in MODELS:
            parquet_path = f"{args.data_dir}/{model}/processed_components.parquet"
            if not os.path.exists(parquet_path):
                print(f"ERROR: {parquet_path} not found. Run process_graphs.py for {model} first.")
                return
            dfs.append(pd.read_parquet(parquet_path))
            display_names.append(MODEL_DISPLAY_NAMES[model])
        print(f"Loaded all 3 models for combined plots.")

        for ct in COMPONENT_TYPES:
            output_path = f"{args.output_dir}/combined_cmap_{ct}.png"
            print(f"  Generating combined {ct} clustermap...")
            plot_combined_clustermaps(dfs, display_names, ct, output_path)

        print(f"  Combined plots saved to {args.output_dir}/")

    print("Done.")


if __name__ == "__main__":
    main()
