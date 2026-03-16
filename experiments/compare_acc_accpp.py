"""Compare ACC vs ACC++ circuit graphs (Appendix E).

Loads pre-traced circuit graphs from two directories (ACC and ACC++), computes
graph metrics (n_nodes, n_edges, in-degree), and generates comparison plots:
  - Point plots of node/edge counts per task (with error bars)
  - ECDF plots of in-degree distribution per model/task

Usage:
    python experiments/compare_acc_accpp.py \\
        --acc_dir data/traced_graphs_acc \\
        --accpp_dir data/traced_graphs_accpp

    # Custom output directory:
    python experiments/compare_acc_accpp.py \\
        --acc_dir data/traced_graphs_acc \\
        --accpp_dir data/traced_graphs_accpp \\
        -o figures/acc_accpp_comparison
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

MODELS = ["gemma-2-2b", "gpt2-small", "pythia-160m"]
TASKS = ["ioi", "gp", "gt"]
METHODS = ["acc", "accpp"]
LABEL_MAP = {"acc": "ACC", "accpp": "ACC++"}
METRIC_LABEL_MAP = {"n_nodes": "Number of Nodes", "n_edges": "Number of Edges"}


def load_graphs(acc_dir: str, accpp_dir: str) -> pd.DataFrame:
    """Load all graphml files and compute graph metrics.

    Args:
        acc_dir: Directory containing ACC traced graphs (model/task subfolders).
        accpp_dir: Directory containing ACC++ traced graphs.

    Returns:
        DataFrame with columns: method, model, task, file, n_nodes, n_edges,
        in_degrees.
    """
    method_dirs = {"acc": acc_dir, "accpp": accpp_dir}

    rows = []
    for method, base_dir in method_dirs.items():
        for model in MODELS:
            for task in TASKS:
                pattern = os.path.join(base_dir, model, task, "*.graphml")
                files = glob.glob(pattern)
                for fp in files:
                    G = nx.read_graphml(fp, force_multigraph=True)
                    rows.append({
                        "method": method,
                        "model": model,
                        "task": task,
                        "file": Path(fp).name,
                        "n_nodes": G.number_of_nodes(),
                        "n_edges": G.number_of_edges(),
                        "in_degrees": [d for _, d in G.in_degree() if d > 0],
                    })

    df = pd.DataFrame(rows)
    return df


def setup_matplotlib():
    """Set publication-quality matplotlib defaults."""
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rc("font", size=8)


def plot_metric(df: pd.DataFrame, task: str, metric: str, output_dir: str):
    """Plot a point plot comparing ACC vs ACC++ for one metric and task.

    Args:
        df: Full DataFrame.
        task: Task name (ioi, gp, gt).
        metric: Column name (n_nodes or n_edges).
        output_dir: Directory to save the PDF.
    """
    df_task = df[df.task == task].reset_index(drop=True)
    if df_task.empty:
        return

    fig, ax = plt.subplots(1, 1, figsize=(3.25, 2))
    sns.pointplot(
        data=df_task, x="model", y=metric, hue="method",
        errorbar="sd", capsize=0.4, linestyle="none", ax=ax, log_scale=True,
    )
    ax.set_xlabel(None)
    ax.set_ylabel(METRIC_LABEL_MAP[metric])

    # Rename legend labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [LABEL_MAP.get(l, l) for l in labels], title=None)

    fig.tight_layout()
    path = os.path.join(output_dir, f"{metric}_{task}.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=800)
    plt.close()
    print(f"  Saved: {path}")


def plot_ecdf(df: pd.DataFrame, task: str, output_dir: str):
    """Plot ECDF of in-degree per model for one task.

    Args:
        df: Full DataFrame.
        task: Task name.
        output_dir: Directory to save the PDFs.
    """
    df_task = df[df.task == task].reset_index(drop=True)
    if df_task.empty:
        return

    # Explode in_degrees list into individual rows
    deg_long = (
        df_task[["model", "method", "in_degrees"]]
        .explode("in_degrees")
        .rename(columns={"in_degrees": "in_degree"})
    )
    deg_long["in_degree"] = deg_long["in_degree"].astype(int)

    for model, d in deg_long.groupby("model", sort=True):
        if d.empty:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(3.25, 2))
        sns.ecdfplot(data=d, x="in_degree", hue="method", ax=ax, legend=True)

        ax.set_xscale("log")
        ax.set_xlabel("In-degree")
        ax.set_ylabel("ECDF")

        # Rename legend labels
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.texts:
                t.set_text(LABEL_MAP.get(t.get_text(), t.get_text()))
            leg.set_title(None)

        plt.tight_layout()
        path = os.path.join(output_dir, f"{task}_ecdf_in-degree_{model}.pdf")
        plt.savefig(path, bbox_inches="tight", dpi=800)
        plt.close()
        print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare ACC vs ACC++ circuit graphs (Appendix E)"
    )
    parser.add_argument(
        "--acc_dir", type=str, required=True,
        help="Directory with ACC traced graphs (model/task subfolders)",
    )
    parser.add_argument(
        "--accpp_dir", type=str, required=True,
        help="Directory with ACC++ traced graphs (model/task subfolders)",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="figures/acc_accpp_comparison",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_matplotlib()

    print("Loading graphs...")
    df = load_graphs(args.acc_dir, args.accpp_dir)
    print(f"  Loaded {len(df)} graphs total")

    # Summary
    for method in METHODS:
        for task in TASKS:
            count = len(df[(df.method == method) & (df.task == task)])
            if count > 0:
                print(f"  {method}/{task}: {count} graphs")

    # Generate plots per task
    for task in TASKS:
        print(f"\n--- {task.upper()} ---")

        # Node count and edge count point plots
        for metric in ["n_nodes", "n_edges"]:
            plot_metric(df, task, metric, args.output_dir)

        # ECDF of in-degree per model
        plot_ecdf(df, task, args.output_dir)

    print(f"\nDone. All plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
