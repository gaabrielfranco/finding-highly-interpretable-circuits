"""Summary barplot from circuit comparison CSVs.

Reads the per-model/task CSV files produced by circuit_comparison.py (Step 5)
and generates a combined barplot of precision/recall/F1-score.

This is Step 6 of the Appendix E pipeline.

Usage:
    python experiments/plot_circuit_comparison.py
    python experiments/plot_circuit_comparison.py --threshold 0.2
"""

import argparse
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=6)


def main():
    parser = argparse.ArgumentParser(
        description="Summary barplot from circuit comparison CSVs (Step 6)"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Only load CSVs for this threshold (e.g. 0.1). Default: load all CSVs.",
    )
    parser.add_argument(
        "--ours_label", type=str, default=None,
        help="Label to replace 'Ours w/ th=X' in the plot (default: 'Ours')",
    )
    args = parser.parse_args()

    # Load CSVs from Step 5
    if args.threshold is not None:
        csv_files = glob.glob(f"data/circuit_comparison/*_{args.threshold}.csv")
    else:
        csv_files = glob.glob("data/circuit_comparison/*.csv")

    # Exclude the RA baselines file from the glob (it's merged separately)
    ra_baselines_path = "data/circuit_comparison/ra_baselines.csv"
    csv_files = [f for f in csv_files if os.path.basename(f) != "ra_baselines.csv"]

    if not csv_files:
        print("No CSV files found in data/circuit_comparison/")
        return

    dfs = []
    for file in csv_files:
        dfs.append(pd.read_csv(file))

    # Merge RA (old ACC) baselines if available
    if os.path.exists(ra_baselines_path):
        dfs.append(pd.read_csv(ra_baselines_path))

    df = pd.concat(dfs, ignore_index=True)

    # Replace "Ours w/ th=X" labels and shorten "Edge Pruning"
    ours_label = args.ours_label or "Ours"
    ours_pattern = {col: ours_label for col in df["Method B"].unique() if col.startswith("Ours w/ th=")}
    ours_pattern["Edge Pruning"] = "EP"
    df["Method B"] = df["Method B"].replace(ours_pattern)

    fig, ax = plt.subplots(1, 4, sharey=True, sharex=False, figsize=(6.8, 2))

    i = 0
    for task in ["ioi", "gp", "gt"]:
        if task == "ioi":
            for model in df.Model.unique():
                sns.barplot(df[(df.Task == task) & (df.Model == model)], x="Method B", y="Value", hue="Metric", ax=ax[i], legend=False)
                ax[i].set_title(f"{model}, {task.upper()}")
                ax[i].set_xlabel(None)
                ax[i].set_ylabel("Metric Value")
                i += 1
        else:
            sns.barplot(df[(df.Task == task) & (df.Model == model)], x="Method B", y="Value", hue="Metric", ax=ax[i], legend=False if i < 3 else True)
            ax[i].set_title(f"{model}, {task.upper()}")
            ax[i].set_xlabel(None)
            i += 1
    plt.legend(loc='lower center', bbox_to_anchor=(-1.2, -0.3), ncol=3)

    folder = "figures/circuit_comparison"
    os.makedirs(folder, exist_ok=True)

    plt.savefig(f'{folder}/circuit_comparison.pdf', bbox_inches='tight', dpi=800)
    plt.close()
    print(f"Saved: {folder}/circuit_comparison.pdf")


if __name__ == "__main__":
    main()
