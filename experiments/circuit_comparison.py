"""Compare ACC++ circuits against published baselines.

Computes precision/recall/F1-score between ACC++ intervention-pruned circuits
and published baselines (Path Patching, ACDC, EAP, EAP-IG, Edge Pruning).
Produces line plots, heatmaps, and CSV files for the barplot step.

This is Step 5 of the Appendix E pipeline.

Usage:
    python experiments/circuit_comparison.py -m gpt2-small -t ioi -th 0.3 -i
    python experiments/circuit_comparison.py -m pythia-160m -t ioi -th 0.2 -i
"""

import argparse
import glob
import os
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

# -------------------------------------------------------------------------
# Components to exclude from comparison per task/method.
# These are components that the baseline methods do not trace.
# -------------------------------------------------------------------------
ELEMS_REMOVE = {
    "ioi": {
        "ACDC": ["AH bias", "Embedding"],
        "EAP": ["AH bias", "Embedding"],
        "Path Patching": ["AH bias", "Embedding", "MLP"],
        "EAP-IG": ["AH bias", "Embedding", "MLP"]
    },
    "gp": {
        "ACDC": ["AH bias", "Embedding"],
        "Edge Pruning": ["AH bias", "Embedding"],
    },
    "gt": {
        "Path Patching": ["AH bias", "Embedding"],
        "EAP": ["AH bias", "Embedding"],
        "ACDC": ["AH bias", "Embedding"],
        "Edge Pruning": ["AH bias", "Embedding"],
    }
}

# Which baseline is considered "canonical" for each task/model (used in barplot CSV).
CANONICAL_METHOD = {
    "ioi": {
        "gpt2-small": "Path Patching",
        "pythia-160m": "EAP-IG"
    },
    "gt": {
        "gpt2-small": "Path Patching"
    },
    "gp": {
        "gpt2-small": "ACDC"
    }
}


def get_baseline_circuits(model_name: str, task: str) -> dict:
    """Return published baseline circuits for a given model/task.

    Each circuit is a list of (layer, head_idx) tuples, where head_idx
    can be an int (attention head) or "MLP".
    """
    ALL_CIRCUITS = {}
    if model_name == "gpt2-small" and task == "ioi":
        # Interpretability in the wild — Path Patching circuit
        CIRCUIT = {
            "name mover": [(9, 9), (10, 0), (9, 6)],
            "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)],
            "negative name mover": [(10, 7), (11, 10)],
            "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
            "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
            "duplicate token": [(0, 1), (0, 10), (3, 0)],
            "previous token": [(2, 2), (4, 11)],
        }

        ALL_CIRCUITS["Path Patching"] = [x for elem in list(CIRCUIT.values()) for x in elem]

        ALL_CIRCUITS["ACDC"] = [
                (0, "MLP"),
                (1, "MLP"),
                (2, "MLP"),
                (3, "MLP"),
                (3, 0),
                (5, 5),
                (8, 6),
                (5, "MLP"),
                (6, "MLP"),
                (8, 10),
                (9, 9),
                (9, 6),
                (10, 0),
                (10, 6),
                (10, 10),
        ]

        ALL_CIRCUITS["EAP"] = [
            # a nodes
            (0, 5), (0, 10), (0, 1),
            (2, 9), (2, 2), (2, 11),
            (3, 3), (3, 4), (3, 7), (3, 0),
            (4, 4), (4, 7), (4, 3), (4, 11),
            (5, 6), (5, 5), (5, 9),
            (6, 9), (6, 0), (6, 6),
            (7, 3), (7, 1), (7, 9),
            (8, 6), (8, 10),
            (9, 6), (9, 9), (9, 8), (9, 7),
            (10, 6), (10, 3), (10, 1), (10, 0), (10, 10), (10, 7), (10, 2),
            (11, 10), (11, 2),

            # m nodes
            (0, "MLP"),
            (1, "MLP"),
            (2, "MLP"),
            (3, "MLP"),
            (4, "MLP"),
            (5, "MLP"),
            (6, "MLP"),
            (7, "MLP"),
            (8, "MLP"),
            (9, "MLP"),
            (10, "MLP")
        ]

    elif model_name == "pythia-160m" and task == "ioi":
        EAP_IG_CIRCUIT_PYTHIA = {
            "name mover": [(8, 10), (8, 2), (9, 4), (10, 7)],
            "negative name mover": [(9, 1)],
            "s2 inhibition": [(6, 6), (7, 2), (7, 9)],
            "induction": [(4, 11), (4, 6), (5, 0)],
            "previous token": [(2, 6)],
            "negative copy suppression": [(9, 5)],
            "positive copy suppression": [(8, 9)],
        }

        ALL_CIRCUITS["EAP-IG"] = [x for elem in list(EAP_IG_CIRCUIT_PYTHIA.values()) for x in elem]

    elif model_name == "gpt2-small" and task == "gt":
        ALL_CIRCUITS["Path Patching"] = [
                (5, 1),
                (5, 5),
                (6, 1),
                (6, 9),
                (7, 10),
                (8, 8),
                (8, 11),
                (9, 1),
                (8, "MLP"),
                (9, "MLP"),
                (10, "MLP"),
                (11, "MLP"),
        ]

        ALL_CIRCUITS["EAP"] = [
            (0, 5),    # <a0.5>
            (0, 10),   # <a0.10>
            (0, "MLP"),  # <m0>
            (0, 1),    # <a0.1>
            (1, "MLP"), # <m1>
            (2, "MLP"), # <m2>
            (3, "MLP"), # <m3>
            (4, "MLP"), # <m4>
            (5, "MLP"), # <m5>
            (5, 8),    # <a5.8>
            (6, 1),    # <a6.1>
            (6, 9),    # <a6.9>
            (7, 10),   # <a7.10>
            (7, "MLP"), # <m7>
            (8, "MLP"), # <m8>
            (8, 11),   # <a8.11>
            (8, 10),   # <a8.10>
            (8, 8),    # <a8.8>
            (9, 1),    # <a9.1>
            (9, "MLP"), # <m9>
            (10, "MLP"), # <m10>
            (10, 7),   # <a10.7>
            (11, "MLP"), # <m11>
        ]

        ALL_CIRCUITS["ACDC"] = [
            (0, 1),       # a0.1
            (0, 5),       # a0.5
            (0, "MLP"),    # m0
            (1, "MLP"),   # m1
            (2, "MLP"),   # m2
            (3, "MLP"),   # m3
            (7, 11),      # a7.11
            (6, 1),       # a6.1
            (8, 8),       # a8.8
            (6, 9),       # a6.9
            (5, 5),       # a5.5
            (8, 11),      # a8.11
            (7, 10),      # a7.10
            (9, 1),       # a9.1
            (10, 4),      # a10.4
            (8, "MLP"),   # m8
            (9, "MLP"),   # m9
            (10, "MLP"),  # m10
            (11, "MLP"),  # m11
        ]

        EDGE_PRUNNING_CIRCUIT_GT = [
            (0, 1, "q"),    # Head 0.1.Q
            (0, 1, "k"),    # Head 0.1.K
            (0, 1, "v"),    # Head 0.1.V
            (0, "MLP"),     # MLP 0
            (0, 1, "o"),    # Head 0.1.O
            (1, "MLP"),     # MLP 1
            (2, "MLP"),     # MLP 2
            (3, 0, "o"),    # Head 3.0.O
            (8, 11, "v"),   # Head 8.11.V
            (3, "MLP"),     # MLP 3
            (6, 1, "v"),    # Head 6.1.V
            (6, 9, "v"),    # Head 6.9.V
            (9, 1, "v"),    # Head 9.1.V
            (7, 10, "v"),   # Head 7.10.V
            (7, 10, "q"),   # Head 7.10.Q
            (6, "MLP"),     # MLP 6
            (5, "MLP"),     # MLP 5
            (7, 10, "o"),   # Head 7.10.O
            (8, 11, "o"),   # Head 8.11.O
            (7, "MLP"),     # MLP 7
            (6, 1, "o"),    # Head 6.1.O
            (9, 1, "o"),    # Head 9.1.O
            (8, "MLP"),     # MLP 8
            (6, 9, "o"),    # Head 6.9.O
            (9, "MLP"),     # MLP 9
            (8, 8, "o"),    # Head 8.8.O
            (10, "MLP"),    # MLP 10
            (11, "MLP"),    # MLP 11
            (8, 8, "v")     # Head 8.8.V
        ]

        # Removing q, k, v, o nodes
        ALL_CIRCUITS["Edge Pruning"] = sorted(list(set([(x[0], x[1]) for x in EDGE_PRUNNING_CIRCUIT_GT])), key=lambda x: x[0])

    elif model_name == "gpt2-small" and task == "gp":
        ALL_CIRCUITS["ACDC"] = [
            (0, 4),         # a0.h4
            (0, "MLP"),     # m0_
            (1, 10),        # a1.h10
            (1, 9),         # a1.h9
            (1, "MLP"),     # m1_
            (3, "MLP"),     # m3_
            (4, "MLP"),     # m3_
            (4, 3),         # a4.h3
            (5, "MLP"),     # m5_
            (6, 0),         # a6.h0
            (7, 1),         # a7.h1
            (7, "MLP"),     # m7_
            (9, 7),         # a9.h7
            (10, 9),        # a10.h9
            (10, "MLP"),    # m10_
            (11, "MLP")     # m11_
        ]

        EDGE_PRUNNING_CIRCUIT_GP = [
            (0, 1, "v"),    # Head 0.1.V
            (0, 1, "o"),    # Head 0.1.O
            (0, 4, "q"),    # Head 0.4.Q
            (0, 4, "o"),    # Head 0.4.O
            (0, "MLP"),     # MLP 0
            (1, "MLP"),     # MLP 1
            (2, "MLP"),     # MLP 2
            (2, 9, "v"),    # Head 2.9.V
            (2, 9, "o"),    # Head 2.9.O
            (3, "MLP"),     # MLP 3
            (3, 6, "v"),    # Head 3.6.V
            (3, 6, "o"),    # Head 3.6.O
            (4, 3, "k"),    # Head 4.3.K
            (4, 3, "v"),    # Head 4.3.V
            (4, 3, "o"),    # Head 4.3.O
            (5, "MLP"),     # MLP 5
            (5, 10, "v"),   # Head 5.10.V
            (5, 10, "o"),   # Head 5.10.O
            (6, 0, "v"),    # Head 6.0.V
            (6, 0, "o"),    # Head 6.0.O
            (7, "MLP"),     # MLP 7
            (7, 5, "v"),    # Head 7.5.V
            (7, 5, "o"),    # Head 7.5.O
            (8, 5, "o"),    # Head 8.5.O
            (9, 7, "v"),    # Head 9.7.V
            (9, 7, "o"),    # Head 9.7.O
            (10, 9, "v"),   # Head 10.9.V
            (10, 9, "o"),   # Head 10.9.O
            (11, 1, "v"),   # Head 11.1.V
            (11, 1, "o"),   # Head 11.1.O
            (11, 7, "o"),   # Head 11.7.O
            (11, 7, "v"),   # Head 11.7.V
            (11, 8, "v"),   # Head 11.8.V
            (11, 8, "o"),   # Head 11.8.O
            (10, "MLP"),    # MLP 10
            (11, "MLP")     # MLP 11
        ]

        ALL_CIRCUITS["Edge Pruning"] = sorted(list(set([(x[0], x[1]) for x in EDGE_PRUNNING_CIRCUIT_GP])), key=lambda x: x[0])

    return ALL_CIRCUITS


def load_graphs(model: str, task: str, interventions: bool = False) -> dict:
    """Load combined graph files for a model/task."""
    if not interventions:
        graph_files = sorted(glob.glob(f"data/combined_graphs/{model}/{task}/*"))
    else:
        graph_files = sorted(glob.glob(f"data/combined_graphs_intervention/{model}/{task}/*"))

    if not graph_files:
        raise FileNotFoundError(f"No combined graph files found for {model}/{task} (interventions={interventions})")

    GRAPHS = {}
    for file in graph_files:
        thresh = file.split("_")[-1].split(".graphml")[0]
        GRAPHS[thresh] = nx.read_graphml(file, force_multigraph=True)

    return GRAPHS


def compute_precision_recall(G: nx.DiGraph,
                             circuit_gt: list,
                             elems_remove: list = ["AH bias", "Embedding"]) -> Tuple[int, int]:
    """Compute precision and recall of a traced graph against a baseline circuit."""
    relevant_retrieved_instances = set()
    all_retrieved_instances = set()
    for node in G.nodes():
        node_tuple = eval(node)

        # If it's the root node: ('IO-S direction', 'end')
        if len(node_tuple) != 4:
            continue

        layer, ah_idx, _, _ = node_tuple

        if (layer, ah_idx) in circuit_gt:
            relevant_retrieved_instances.add((layer, ah_idx))

        if ah_idx in elems_remove:
            continue # They do not trace these components, so it's not comparable

        all_retrieved_instances.add((layer, ah_idx))

    if len(all_retrieved_instances) == 0:
        precision = 0.0
    else:
        precision = len(relevant_retrieved_instances) / len(all_retrieved_instances)
    recall = len(relevant_retrieved_instances) / len(circuit_gt)

    return precision, recall


def compute_precision_recall_baselines(circuit_A: list, circuit_B: list) -> Tuple[int, int]:
    """Compute precision and recall between two baseline circuits."""
    all_retrieved_instances = set(circuit_B)
    relevant_retrieved_instances = set()
    for component in circuit_B:
        if component in circuit_A:
            relevant_retrieved_instances.add(component)

    if len(all_retrieved_instances) == 0:
        precision = 0.0
    else:
        precision = len(relevant_retrieved_instances) / len(all_retrieved_instances)
    recall = len(relevant_retrieved_instances) / len(circuit_A)

    return precision, recall


def construct_df_results(GRAPHS: dict,
                         CIRCUIT: list,
                         elems_remove: list,
                         method: str) -> pd.DataFrame:
    """Build dataframe with precision/recall/F1 per threshold."""
    datapoints = []
    for thresh in GRAPHS:
        precision, recall = compute_precision_recall(GRAPHS[thresh], CIRCUIT, elems_remove)
        # Precision
        datapoints.append([method, precision, "Precision", thresh])
        # Recall
        datapoints.append([method, recall, "Recall", thresh])
        # f1-score
        if np.isclose(precision, 0) and np.isclose(recall, 0):
            f1_score = 0.
        else:
            f1_score = (2 * precision * recall) / (precision + recall)
        datapoints.append([method, f1_score, "F1-Score", thresh])

    return pd.DataFrame(datapoints, columns=["Baseline Method", "Metric Value", "Metric", "Threshold"]) 


def plot_lineplot_circuit(df: pd.DataFrame, results_baselines: pd.DataFrame, model_name: str, task: str, interventions: bool, folder: str) -> None:
    """Line plot comparing ACC++ precision/recall/F1 against baselines across thresholds."""
    fig = sns.catplot(
        data=df, x="Threshold", y="Metric Value", hue="Metric", col="Baseline Method",
        kind="point", dodge=True, legend=True
    );

    for ax in fig.axes[0]:
        method = ax.get_title().split("= ")[-1]
        df_method = results_baselines[(results_baselines["Method A"] == method) & (results_baselines["Method B"] != method)]

        ax.tick_params(axis='x', labelrotation=90)

        if len(df_method) == 0:
            ax.set_ylabel("F1-Score");
            continue
        elif len(df_method) == 1:
            ax.axhline(y=df_method.iloc[0]["F1-Score"], label="Other baseline F1-Score", color="red", ls="--");
        else:
            min_f1, max_f1 = df_method["F1-Score"].min(), df_method["F1-Score"].max()
            ax.axhline(y=min_f1, label="Min F1-Score\nbetween baselines", color="red", ls="--");
            ax.axhline(y=max_f1, label="Max F1-Score\nbetween baselines", color="green", ls="--");
            ax.set_ylabel("F1-Score");

    fig.legend.remove()

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    fig.fig.subplots_adjust(top=0.85)
    fig.fig.suptitle(f"Task: {task}; Model: {model_name}");
    interv = "_no-interv" if not interventions else ""
    plt.savefig(f"{folder}/lineplot_{model_name}_{task}{interv}.pdf", bbox_inches="tight", pad_inches=0.01, dpi=800)
    plt.close()


def comparison_dataframes_computation(CIRCUITS, GRAPHS, task):
    """Compute baseline-vs-baseline and ACC++-vs-baseline comparisons."""
    # Comparing baselines
    baseline_rows = [] 

    # Computing baseline comparisons
    for method_A in CIRCUITS.keys():
        for method_B in CIRCUITS.keys():
            circuit_A, circuit_B = CIRCUITS[method_A], CIRCUITS[method_B]
            if task == "ioi" and (method_A == "Path Patching" or method_A == "EAP-IG"):
                print(f"Comparing {method_A} with {method_B} and REMOVING MLPs...")
                circuit_A = [x for x in circuit_A if "MLP" not in x]
                circuit_B = [x for x in circuit_B if "MLP" not in x]
                print()

            prec, rec = compute_precision_recall_baselines(circuit_A, circuit_B)
            baseline_rows.append([method_A, method_B, prec, rec])
    results_baselines = pd.DataFrame(baseline_rows, columns=["Method A", "Method B", "Precision", "Recall"])
    results_baselines["F1-Score"] = (2 * results_baselines["Precision"] * results_baselines["Recall"]) / (results_baselines["Precision"] + results_baselines["Recall"])

    # Computing the comparison against our method
    dfs = []
    for baseline_method in CIRCUITS:
        dfs.append(construct_df_results(GRAPHS, CIRCUITS[baseline_method], ELEMS_REMOVE[task][baseline_method], baseline_method))
    df = pd.concat(dfs, ignore_index=True)

    return df, results_baselines


def get_circuit_threshold(GRAPHS: dict, best_threshold: float) -> list:
    """Extract the set of (layer, head_idx) components at a given threshold."""
    best_threshold = str(best_threshold)
    BEST_CIRCUIT = set()

    G = GRAPHS[best_threshold]
    for node in G.nodes():
        node_tuple = eval(node)

        # If it's the root node: ('IO-S direction', 'end')
        if len(node_tuple) != 4:
            continue

        layer, ah_idx, _, _ = node_tuple

        if ah_idx in ["AH bias", "Embedding"]:
            continue

        BEST_CIRCUIT.add((layer, ah_idx))

    BEST_CIRCUIT = list(BEST_CIRCUIT)

    return BEST_CIRCUIT


def plot_heatmap_best_circuit(ALL_CIRCUITS: dict, BEST_CIRCUIT: list, best_threshold: float, task: str, model_name: str, interventions: bool, folder: str) -> None:
    """Heatmap comparing ACC++ best circuit against all baselines."""
    # Comparing baselines
    rows = []
    for method_A in ALL_CIRCUITS.keys():
        for method_B in ALL_CIRCUITS.keys():
            circuit_A, circuit_B = ALL_CIRCUITS[method_A], ALL_CIRCUITS[method_B]
            if task == "ioi" and (method_A == "Path Patching" or method_A == "EAP-IG"):
                print(f"Comparing {method_A} with {method_B} and REMOVING MLPs...")
                circuit_A = [x for x in circuit_A if "MLP" not in x]
                circuit_B = [x for x in circuit_B if "MLP" not in x]
                print()

            prec, rec = compute_precision_recall_baselines(circuit_A, circuit_B)
            rows.append([method_A, method_B, prec, rec])

        # Comparing against our best circuit
        prec, rec = compute_precision_recall_baselines(circuit_A, BEST_CIRCUIT[method_A])
        rows.append([method_A, f"Ours w/\nth={best_threshold}", prec, rec])
    for method_B in ALL_CIRCUITS.keys():
        # Comparing against our best circuit
        prec, rec = compute_precision_recall_baselines(BEST_CIRCUIT[method_B], ALL_CIRCUITS[method_B])
        rows.append([f"Ours w/\nth={best_threshold}", method_B, prec, rec])

    # Comparing our best circuit against itself
    prec, rec = compute_precision_recall_baselines(BEST_CIRCUIT[method_A], BEST_CIRCUIT[method_A]) # Whatever key will work here
    rows.append([f"Ours w/\nth={best_threshold}", f"Ours w/\nth={best_threshold}", prec, rec])

    results_baselines_and_best = pd.DataFrame(rows, columns=["Method A", "Method B", "Precision", "Recall"])
    results_baselines_and_best["F1-Score"] = (2 * results_baselines_and_best["Precision"] * results_baselines_and_best["Recall"]) / (results_baselines_and_best["Precision"] + results_baselines_and_best["Recall"])

    results_baselines_and_best["Method A"] = results_baselines_and_best["Method A"].replace({"Path Patching": "Path\nPatching", "Edge Pruning": "Edge\nPruning"})
    results_baselines_and_best["Method B"] = results_baselines_and_best["Method B"].replace({"Path Patching": "Path\nPatching", "Edge Pruning": "Edge\nPruning"})

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc('font', size=8)

    for metric in ["Precision", "Recall", "F1-Score"]:
        fig, ax = plt.subplots(1, 1, figsize=(3, 1.9))
        sns.heatmap(results_baselines_and_best.pivot(index="Method A", columns="Method B", values=metric), annot=True);
        plt.title(f"Task: {task}; Model: {model_name}");

        interv = "_no-interv" if not interventions else ""
        plt.savefig(f"{folder}/heatmap_{model_name}_{task}{interv}_{metric.lower()}.pdf", bbox_inches="tight", pad_inches=0.01, dpi=800)
        plt.close()


def main(args, folder):
    model_name, task, best_threshold = args.model_name, args.task, args.best_threshold

    GRAPHS = load_graphs(model_name, task, interventions=args.interventions)

    ALL_CIRCUITS = get_baseline_circuits(model_name, task)

    # Comparing baselines
    df, results_baselines = comparison_dataframes_computation(ALL_CIRCUITS, GRAPHS, task)

    plot_lineplot_circuit(df, results_baselines, model_name, task, args.interventions, folder)

    BEST_CIRCUIT = get_circuit_threshold(GRAPHS, best_threshold)

    BEST_CIRCUIT_PER_BASELINE = {}
    for baseline_method in ALL_CIRCUITS:
        BEST_CIRCUIT_PER_BASELINE[baseline_method] = [x for x in BEST_CIRCUIT if x[1] not in ELEMS_REMOVE[task][baseline_method]]

    plot_heatmap_best_circuit(ALL_CIRCUITS, BEST_CIRCUIT_PER_BASELINE, best_threshold, task, model_name, args.interventions, folder)

    # Barplot data
    canonical_method = CANONICAL_METHOD[task][model_name]
    ALL_CIRCUITS[f"Ours w/ th={best_threshold}"] = BEST_CIRCUIT_PER_BASELINE[canonical_method]

    # Removing the elements of comparison that are not in the canonical circuit.
    for baseline_method in ALL_CIRCUITS:
        ALL_CIRCUITS[baseline_method] = [x for x in ALL_CIRCUITS[baseline_method] if x[1] not in ELEMS_REMOVE[task][canonical_method]]

    rows = []

    for baseline_method in ALL_CIRCUITS:
        if baseline_method == canonical_method:
            continue

        prec, rec = compute_precision_recall_baselines(ALL_CIRCUITS[canonical_method], ALL_CIRCUITS[baseline_method])
        rows.append([canonical_method, baseline_method, prec, rec])

    results_baselines_and_best = pd.DataFrame(rows, columns=["Method A", "Method B", "Precision", "Recall"])
    results_baselines_and_best["F1-Score"] = (2 * results_baselines_and_best["Precision"] * results_baselines_and_best["Recall"]) / (results_baselines_and_best["Precision"] + results_baselines_and_best["Recall"])

    df_melted = results_baselines_and_best.melt(id_vars='Method B', value_vars=['Precision', 'Recall', 'F1-Score'], var_name='Metric', value_name='Value')

    df_melted["Model"] = model_name
    df_melted["Task"] = task

    df_melted.to_csv(f"{folder}/{model_name}_{task}_{best_threshold}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare ACC++ circuits against published baselines (Step 5)"
    )
    parser.add_argument("-m", "--model_name", choices=["gpt2-small", "pythia-160m"], required=True)
    parser.add_argument("-t", "--task", choices=["ioi", "gt", "gp"], required=True)
    parser.add_argument("-th", "--best_threshold", type=float, required=True,
                        help="Threshold at which to extract the 'best' ACC++ circuit for heatmaps/barplot CSV")
    parser.add_argument("-i", "--interventions", action="store_true",
                        help="Use intervention-pruned graphs (from Step 4) instead of frequency-pruned graphs")
    args = parser.parse_args()

    folder = "data/circuit_comparison"
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    main(args, folder)
