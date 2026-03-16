"""Find representative prompts for each template cluster.

For each component granularity (heads, edges, edge-SV pairs), finds the most
central prompt (minimum sum of Jaccard distances to all others) for:
- ALL prompts
- ABBA template prompts
- BABA template prompts
- Each of the 15 low-level templates

This is Step 4 of the Section 3 / Appendix F pipeline.

Usage:
    python experiments/find_representatives.py -m gpt2-small
    python experiments/find_representatives.py -m pythia-160m
    python experiments/find_representatives.py -m gemma-2-2b
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import distance

MODELS = ["gpt2-small", "pythia-160m", "gemma-2-2b"]
COMPONENT_TYPES = ["head_as_component", "edge_as_component", "sv_as_component"]


def main():
    parser = argparse.ArgumentParser(description="Find representative prompts per template cluster.")
    parser.add_argument(
        "-m", "--model", required=True, choices=MODELS,
        help="Model name.",
    )
    parser.add_argument(
        "--data_dir", default="data/clustering",
        help="Base directory for processed data. Default: data/clustering/",
    )
    parser.add_argument(
        "--output_dir", default="figures/clustering",
        help="Output directory for results. Default: figures/clustering/",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    parquet_path = f"{args.data_dir}/{args.model}/processed_components.parquet"
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} prompts from {parquet_path}")

    abba_mask = df["high_level_template"].str.contains("ABBA").to_numpy()
    baba_mask = df["high_level_template"].str.contains("BABA").to_numpy()
    abba_df = df[abba_mask]
    baba_df = df[baba_mask]

    json_dict: dict[str, dict[str, dict]] = {}

    output_path = f"{args.output_dir}/{args.model}_representatives.txt"
    with open(output_path, "w") as f:
        for k in COMPONENT_TYPES:
            cmatrix = np.stack(df[k].to_numpy())
            dmatrix = distance.squareform(distance.pdist(cmatrix, metric="jaccard"))

            # Find representatives (argmin of sum of distances)
            all_repr_idx = np.argmin(np.sum(dmatrix, axis=1))
            abba_repr_idx = np.argmin(np.sum(dmatrix[np.ix_(abba_mask, abba_mask)], axis=1))
            baba_repr_idx = np.argmin(np.sum(dmatrix[np.ix_(baba_mask, baba_mask)], axis=1))

            all_repr = df.iloc[all_repr_idx]
            abba_repr = abba_df.iloc[abba_repr_idx]
            baba_repr = baba_df.iloc[baba_repr_idx]

            f.write(f"--- COMPONENT TYPE: {k} ---\n")
            f.write(f"ALL (id={all_repr['id']}, high_level_template={all_repr['high_level_template']}, low_level_template={all_repr['low_level_template']}): {all_repr['text']}\n")
            f.write(f"ABBA (id={abba_repr['id']}, high_level_template={abba_repr['high_level_template']}, low_level_template={abba_repr['low_level_template']}): {abba_repr['text']}\n")
            f.write(f"BABA (id={baba_repr['id']}, high_level_template={baba_repr['high_level_template']}, low_level_template={baba_repr['low_level_template']}): {baba_repr['text']}\n")

            def _repr_entry(row: pd.Series) -> dict:
                return {
                    "id": int(row["id"]),
                    "high_level_template": str(row["high_level_template"]),
                    "low_level_template": int(row["low_level_template"]),
                    "text": str(row["text"]),
                }

            comp_dict: dict[str, dict] = {}
            comp_dict["ALL"] = _repr_entry(all_repr)
            comp_dict["ABBA"] = _repr_entry(abba_repr)
            comp_dict["BABA"] = _repr_entry(baba_repr)

            for i in range(15):
                templ_mask = (df["low_level_template"] == i)
                templ_repr_idx = np.argmin(np.sum(dmatrix[np.ix_(templ_mask, templ_mask)], axis=1))
                templ_repr = df[templ_mask].iloc[templ_repr_idx]
                f.write(f"Template {i} (id={templ_repr['id']}, high_level_template={templ_repr['high_level_template']}, low_level_template={templ_repr['low_level_template']}): {templ_repr['text']}\n")
                comp_dict[f"Template {i}"] = _repr_entry(templ_repr)
            f.write("\n")
            json_dict[k] = comp_dict
            print(f"  {k}: ALL={all_repr['id']}, ABBA={abba_repr['id']}, BABA={baba_repr['id']}")

    json_path = f"{args.output_dir}/{args.model}_representatives.json"
    with open(json_path, "w") as f:
        json.dump(json_dict, f, indent=2)

    print(f"\nSaved to {output_path}")
    print(f"Saved to {json_path}")


if __name__ == "__main__":
    main()
