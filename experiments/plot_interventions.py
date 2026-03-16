"""Violin/bar plots from curated edge intervention data.

Reads parquet files produced by interventions.py (Step 3a) and generates:
- Violin plot of logit_diff_metric across all models (1x3 row, IOI only)
- Per-model violin plots for IOI
- Bar plot of attention weight effect (1x3 row, IOI only)
- Histograms of cosine similarity and norm ratio distributions

This is Step 7 of the Appendix E pipeline.

Usage:
    python experiments/plot_interventions.py
"""

import os
from copy import deepcopy

import glob
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=8)

MODELS = ["gpt2-small", "pythia-160m", "gemma-2-2b"]


def main():
    folder = "figures/interventions"
    os.makedirs(folder, exist_ok=True)

    # Load IOI parquet files
    dfs = []
    files = glob.glob("data/intervention_data/*_ioi.parquet")
    for file in files:
        df_file = pd.read_parquet(file)
        df_file["task"] = "ioi"
        dfs.append(df_file)

    if not dfs:
        print("No parquet files found in data/intervention_data/")
        return

    logit_diff_ablations = pd.concat(dfs, ignore_index=True)

    logit_diff_ablations["edge_labeled_group"] = logit_diff_ablations["edge_labeled_group"].apply(lambda x: "->\n".join(x.split("-> ")))

    # -----------------------------------------------------------------------
    # Plot 1: Violin plot of logit_diff_metric (1x3 row, IOI only)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(1, len(MODELS), sharex=False, sharey=False, figsize=(5.5, 1.8))
    task = "ioi"
    for j, model in enumerate(MODELS):
        data = logit_diff_ablations[
            (logit_diff_ablations.is_ablated) &
            (logit_diff_ablations.intervention_type == "local") &
            (logit_diff_ablations.model_name == model) &
            (logit_diff_ablations.task == task)
        ]
        sns.violinplot(
            data,
            x="logit_diff_metric",
            y="edge_labeled_group",
            hue="operation_performed",
            hue_order=['Removing (Random)', 'Boosting (Random)', 'Removing (SVs)', 'Boosting (SVs)'],
            linewidth=0.5,
            legend=True if j == len(MODELS) - 1 else False,
            density_norm='width',
            inner=None,
            cut=0,
            ax=ax[j],
        )

        ax[j].set_xlabel("(F(E, h) - F) / F")
        if j == 0:
            ax[j].set_ylabel("IOI")
        else:
            ax[j].set_ylabel("")

        if j > 0:
            ax[j].set_yticklabels([])

        ax[j].axvline(0, color='black', linestyle='--', linewidth=0.5)
        ax[j].set_title(model)

    ax[-1].legend(loc="center", bbox_to_anchor=(1.5, 0.5), fontsize=6)
    plt.savefig(f'{folder}/interventions.pdf', bbox_inches='tight', dpi=800);
    plt.close();
    print(f"Saved: {folder}/interventions.pdf")

    # -----------------------------------------------------------------------
    # Plot 2: Single model violin plots for IOI
    # -----------------------------------------------------------------------
    for model in MODELS:
        fig, ax = plt.subplots(1, 1, sharex=False, sharey=True, figsize=(2, 1.5))
        task = "ioi"

        data = deepcopy(logit_diff_ablations[
            (logit_diff_ablations.is_ablated) &
            (logit_diff_ablations.intervention_type == "local") &
            (logit_diff_ablations.model_name == model) &
            (logit_diff_ablations.task == task)
        ])

        data["edge_labeled_group"] = data["edge_labeled_group"].replace({
            'S-Inhibition Head ->\nName Mover Head': 'S-Inhibition ->\nName Mover',
            'Induction Head ->\nS-Inhibition Head': 'Induction ->\nS-Inhibition',
            'Previous Token Head ->\nInduction Head': 'Prev. Token ->\nInduction'
        })

        sns.violinplot(
            data,
            x="edge_labeled_group",
            y="logit_diff_metric",
            hue="operation_performed",
            hue_order=['Removing (Random)', 'Boosting (Random)', 'Removing (SVs)', 'Boosting (SVs)'],
            linewidth=0.5,
            legend=False,
            density_norm='width',
            inner=None,
            cut=0,
            ax=ax
        )

        ax.set_ylabel("(F(E, h) - F) / F");
        ax.set_xlabel("")
        if model == "pythia-160m":
            ax.set_yticks([-0.5, 0, 0.5])
            ax.set_ylim(-0.5, 0.5)
        elif model == "gpt2-small":
            ax.set_yticks([-0.25, 0, 0.25])
            ax.set_ylim(-0.25, 0.25)
        ax.tick_params(axis='x', rotation=60)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.savefig(f'{folder}/interventions_ioi_{model}.pdf', bbox_inches='tight', dpi=800);
        plt.close();
        print(f"Saved: {folder}/interventions_ioi_{model}.pdf")

    # -----------------------------------------------------------------------
    # Plot 3: Bar plot of attention weight effect (1x3 row, IOI only)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(1, len(MODELS), sharex=False, sharey=True, figsize=(5.5, 1.8))
    task = "ioi"
    for j, model in enumerate(MODELS):
        data = logit_diff_ablations[
            (logit_diff_ablations.is_ablated) &
            (logit_diff_ablations.intervention_type == "local") &
            (logit_diff_ablations.model_name == model) &
            (logit_diff_ablations.task == task)
        ]

        sns.barplot(
            data,
            x="edge_labeled_group",
            y="scores_dest_src_diff_metric",
            hue="operation_performed",
            hue_order=['Removing (Random)', 'Boosting (Random)', 'Removing (SVs)', 'Boosting (SVs)'],
            legend=True if j == len(MODELS) - 1 else False,
            errorbar="sd",
            ax=ax[j],
            err_kws={"linewidth": 1},
        )

        ax[j].set_xlabel("Edges")
        if j == 0:
            ax[j].set_ylabel("IOI\n" + r"$A_{ds}^{\text{interv}} - A_{ds}$")

        ax[j].set_xticklabels([])
        ax[j].set_title(model)

    ax[-1].legend(loc="center", bbox_to_anchor=(1.5, 0.5), fontsize=6)
    plt.savefig(f'{folder}/interventions_attn_weight_effect.pdf', bbox_inches='tight', dpi=800);
    plt.close();
    print(f"Saved: {folder}/interventions_attn_weight_effect.pdf")

    # -----------------------------------------------------------------------
    # Plot 4: Histograms of cosine similarity and norm ratio
    # -----------------------------------------------------------------------
    for metric in ["cosine_similarity", "norm_ratio"]:
        for is_random in [True, False]:
            fig, ax = plt.subplots(1, 1, figsize=(2.5, 1.9))
            sns.histplot(
                logit_diff_ablations[(logit_diff_ablations.is_ablated) & (logit_diff_ablations.is_random == is_random)],
                x=metric,
                ax=ax
            );
            if metric == "cosine_similarity":
                plt.xlabel("Cosine similarity");
            else:
                plt.xlabel("Norm ratio");
            random_name = "random" if is_random else "not-random"
            plt.tight_layout();
            plt.savefig(f'{folder}/interventions_{metric}_{random_name}.pdf', bbox_inches='tight', dpi=800);
            plt.close();
            print(f"Saved: {folder}/interventions_{metric}_{random_name}.pdf")

    print(f"\nDone. All plots saved to {folder}/")


if __name__ == "__main__":
    main()
