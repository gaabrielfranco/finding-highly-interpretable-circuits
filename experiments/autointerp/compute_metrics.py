"""Stage 4: Compute metrics, generate Table 1 and paper figures.

Reads scored activation H5 files (from Stage 3, containing
``scoring_labels`` and ``scoring_labels_pred``) and produces:

- ``table_interpretability.tex``: LaTeX table with per-model median
  accuracy/precision/recall and IQR (25%--75%), including a hardcoded SAE
  baseline row from the literature.
- ``fraction_significant_fdr_per_model.pdf``: aggregate barplot with one
  bar per model showing overall FDR-significant fraction with Wilson CIs.
- ``fdr_per_model.csv``: per-model FDR stats (n, k, frac, ci_low, ci_high).
- ``fraction_significant_fdr_5pct_{model}.pdf``: per-layer fraction of
  FDR-corrected significant signals with Wilson 95% CIs (one per model).
- ``median_metrics_by_layer_{model}.pdf``: per-layer median
  accuracy/precision/recall with IQR bands and overall median h-lines
  (one per model).
- ``metrics.csv``: raw per-signal metrics (model, layer, signal_idx,
  fisher_p_value, fdr_significant, accuracy, precision, recall).

New dependencies (not required by the core library):
  scipy, statsmodels, scikit-learn  (see pyproject.toml [autointerp])

Usage:
    python compute_metrics.py \\
        --models gpt2-small pythia-160m gemma-2-2b \\
        --task ioi-balanced \\
        --activations_dir data/autointerp/ \\
        --output_dir data/autointerp/results/
"""

import argparse
import logging
from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import confusion_matrix
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Model short-name → filename suffix used by FDR plot filenames in the notebook.
_MODEL_FILENAME_SUFFIX = {
    "gpt2-small": "chat",
    "pythia-160m": "pythia160m",
    "gemma-2-2b": "gemma2_2b",
}


def _setup_paper_rc() -> None:
    """Set matplotlib RC params for paper-quality PDF output."""
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rc("font", size=8)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_model_data(
    model_name: str,
    task: str,
    activations_dir: str,
) -> pd.DataFrame:
    """Load per-signal scoring data for all layers of one model.

    Scans ``activations_dir`` for files matching
    ``activations_*_{task}_{model_short}.h5`` and reads
    ``scoring_labels`` and ``scoring_labels_pred`` from each layer group.

    Args:
        model_name: TransformerLens model name (e.g. ``"gpt2-small"``).
        task: Task identifier (e.g. ``"ioi-balanced"``).
        activations_dir: Directory containing activation H5 files.

    Returns:
        DataFrame with columns: model, layer, signal_idx,
        labels (object — numpy row), preds (object — numpy row).
        Only layers that have ``scoring_labels`` are included.
    """
    model_short = model_name.split("/")[-1]
    pattern = str(
        Path(activations_dir) / f"activations_*_{task}_{model_short}.h5"
    )
    import glob
    files = sorted(glob.glob(pattern))
    if not files:
        log.warning(f"No activation files found for {model_name} / {task}")
        return pd.DataFrame()

    rows = []
    for fpath in files:
        with h5py.File(fpath, "r") as f:
            for key in sorted(f.keys()):
                if not key.startswith("layer_"):
                    continue
                grp = f[key]
                if "scoring_labels" not in grp or "scoring_labels_pred" not in grp:
                    continue
                layer = int(key.split("_")[1])
                labels_all = grp["scoring_labels"][:]    # (N, n_labels)
                preds_all  = grp["scoring_labels_pred"][:]
                for i in range(len(labels_all)):
                    rows.append({
                        "model":      model_short,
                        "layer":      layer,
                        "signal_idx": i,
                        "labels":     labels_all[i],
                        "preds":      preds_all[i],
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical computations
# ---------------------------------------------------------------------------

def _is_uninterpreted(labels: np.ndarray) -> bool:
    """Return True if all label entries are -1 (signal was not interpreted)."""
    return bool(np.all(labels == -1))


def _is_failed(preds: np.ndarray) -> bool:
    """Return True if all prediction entries are -1 (judge scoring failed)."""
    return bool(np.all(preds == -1))


def _fisher_pvalue(labels: np.ndarray, preds: np.ndarray) -> float | None:
    """Compute a one-sided Fisher exact p-value for one signal.

    Builds a 2x2 contingency table from valid (non -1) entries:

    .. code-block:: text

                   Pred=1   Pred=0
        True=1:      a        b      (top-K examples)
        True=0:      c        d      (random examples)

    Tests ``alternative="greater"`` (top-K more likely classified as 1
    than random).

    Args:
        labels: Ground-truth vector, values in {-1, 0, 1}.
        preds: Judge prediction vector, values in {-1, 0, 1}.

    Returns:
        One-sided p-value, or ``None`` if the signal is invalid (all -1,
        or only one class present).
    """
    mask = (labels != -1) & (preds != -1)
    y_true = labels[mask]
    y_pred = preds[mask]

    if len(y_true) == 0:
        return None
    if len(np.unique(y_true)) < 2:
        return None  # only one class — test is undefined

    a = int(((y_true == 1) & (y_pred == 1)).sum())
    b = int(((y_true == 1) & (y_pred == 0)).sum())
    c = int(((y_true == 0) & (y_pred == 1)).sum())
    d = int(((y_true == 0) & (y_pred == 0)).sum())

    _, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
    return float(pvalue)


def _classification_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
) -> dict[str, float] | None:
    """Compute accuracy, precision, recall for one signal.

    Filters out -1 entries from both vectors before computing metrics.

    Args:
        labels: Ground-truth vector, values in {-1, 0, 1}.
        preds: Judge prediction vector, values in {-1, 0, 1}.

    Returns:
        Dict with keys ``accuracy``, ``precision``, ``recall``, or
        ``None`` if no valid entries remain.
    """
    mask = (labels != -1) & (preds != -1)
    y_true = labels[mask].astype(int)
    y_pred = preds[mask].astype(int)

    if len(y_true) == 0:
        return None

    C = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = C.ravel()

    total   = tn + fp + fn + tp
    accuracy  = (tp + tn) / total if total > 0 else np.nan
    precision = tp / (tp + fp)     if (tp + fp) > 0 else np.nan
    recall    = tp / (tp + fn)     if (tp + fn) > 0 else np.nan

    return {"accuracy": accuracy, "precision": precision, "recall": recall}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def compute_all_metrics(
    models: list[str],
    task: str,
    activations_dir: str,
    output_dir: str,
) -> None:
    """Run the full metrics pipeline: load data, compute stats, save outputs.

    Args:
        models: List of TransformerLens model names.
        task: Task identifier (e.g. ``"ioi-balanced"``).
        activations_dir: Directory containing activation H5 files.
        output_dir: Directory for all output files (created if needed).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    # ── Per-model summary for Table 1 ──────────────────────────────────────
    table_rows: list[dict] = []

    for model_name in models:
        model_short = model_name.split("/")[-1]
        log.info(f"Loading data for {model_short}...")
        df = load_model_data(model_name, task, activations_dir)
        if df.empty:
            log.warning(f"No data for {model_short}, skipping.")
            continue

        # ── Per-signal flags ────────────────────────────────────────────────
        df["uninterpreted"] = df.apply(
            lambda r: _is_uninterpreted(r["labels"]), axis=1
        )
        df["failed"] = df.apply(
            lambda r: _is_failed(r["preds"]), axis=1
        )

        # ── Fisher p-values ─────────────────────────────────────────────────
        df["fisher_p_value"] = df.apply(
            lambda r: _fisher_pvalue(r["labels"], r["preds"]), axis=1
        )

        # ── Classification metrics ──────────────────────────────────────────
        def _unpack_metrics(row: pd.Series) -> pd.Series:
            m = _classification_metrics(row["labels"], row["preds"])
            if m is None:
                return pd.Series({"accuracy": np.nan, "precision": np.nan, "recall": np.nan})
            return pd.Series(m)

        df[["accuracy", "precision", "recall"]] = df.apply(_unpack_metrics, axis=1)

        # ── BH FDR correction (per model) ───────────────────────────────────
        valid_mask = df["fisher_p_value"].notna()
        df["fdr_significant"] = False
        df["p_value_fdr"] = np.nan

        if valid_mask.sum() > 0:
            p_vals = df.loc[valid_mask, "fisher_p_value"].values
            reject, pvals_fdr, _, _ = multipletests(
                p_vals, alpha=0.05, method="fdr_bh"
            )
            df.loc[valid_mask, "fdr_significant"] = reject
            df.loc[valid_mask, "p_value_fdr"]    = pvals_fdr

        # ── Accumulate for CSV ──────────────────────────────────────────────
        keep_cols = [
            "model", "layer", "signal_idx",
            "uninterpreted", "failed",
            "fisher_p_value", "fdr_significant", "p_value_fdr",
            "accuracy", "precision", "recall",
        ]
        all_rows.append(df[keep_cols])

        # ── Per-model stats for tables ─────────────────────────────────────
        n_total = len(df)
        frac_uninterp = df["uninterpreted"].mean()
        frac_failed   = df["failed"].mean()

        # Metrics: median and IQR
        median_acc  = df["accuracy"].median()
        median_prec = df["precision"].median()
        median_rec  = df["recall"].median()
        q25_acc, q75_acc   = df["accuracy"].quantile(0.25), df["accuracy"].quantile(0.75)
        q25_prec, q75_prec = df["precision"].quantile(0.25), df["precision"].quantile(0.75)
        q25_rec, q75_rec   = df["recall"].quantile(0.25), df["recall"].quantile(0.75)

        table_rows.append({
            "model_short":  model_short,
            "n_total":      n_total,
            "frac_uninterp": frac_uninterp,
            "frac_failed":  frac_failed,
            "median_acc":   median_acc,  "q25_acc":  q25_acc,  "q75_acc":  q75_acc,
            "median_prec":  median_prec, "q25_prec": q25_prec, "q75_prec": q75_prec,
            "median_rec":   median_rec,  "q25_rec":  q25_rec,  "q75_rec":  q75_rec,
        })

        log.info(
            f"{model_short}: n={n_total}, uninterp={frac_uninterp:.3f}, "
            f"failed={frac_failed:.3f}, "
            f"acc={median_acc:.3f}, prec={median_prec:.3f}, rec={median_rec:.3f}"
        )

    if not all_rows:
        log.error("No data loaded for any model. Exiting.")
        return

    full_df = pd.concat(all_rows, ignore_index=True)

    # ── Save metrics CSV ────────────────────────────────────────────────────
    csv_path = str(Path(output_dir) / "metrics.csv")
    full_df.drop(columns=["labels", "preds"], errors="ignore").to_csv(
        csv_path, index=False
    )
    log.info(f"Saved metrics CSV: {csv_path}")

    # ── Tables (LaTeX) ─────────────────────────────────────────────────────
    _save_table_interpretability(table_rows, output_dir)

    # ── Paper figures (one PDF per model) ───────────────────────────────────
    _figure_fdr_significance(full_df, models, output_dir)
    _figure_layer_metrics(full_df, models, output_dir)
    _figure_fdr_per_model(full_df, models, output_dir)

    log.info("All outputs saved.")


# ---------------------------------------------------------------------------
# Table: interpretability comparison (median + IQR)
# ---------------------------------------------------------------------------

# Hardcoded SAE baseline from \cite{paulo2025automatically}.
_SAE_ROW = (
    r"SAE features (AutoInterp) &"  "\n"
    r"$0.76\,(0.67\text{--}0.86)$ &"  "\n"
    r"-- &"  "\n"
    r"-- \\"
)

_MODEL_DISPLAY_NAMES = {
    "gpt2-small": "GPT-2 Small",
    "pythia-160m": "Pythia-160M",
    "gemma-2-2b": "Gemma-2-2B",
}


def _fmt_median_iqr(median: float, q25: float, q75: float) -> str:
    """Format a metric as ``$median\\,(q25\\text{--}q75)$``."""
    return f"${median:.2f}\\,({q25:.2f}\\text{{--}}{q75:.2f})$"


def _save_table_interpretability(
    table_rows: list[dict],
    output_dir: str,
) -> None:
    """Generate Table 1: interpretability comparison (median + IQR).

    Includes a hardcoded SAE baseline row from the literature.

    Args:
        table_rows: List of per-model stat dicts from the pipeline.
        output_dir: Directory for the output file.
    """
    model_lines = []
    for row in table_rows:
        name = _MODEL_DISPLAY_NAMES.get(row["model_short"], row["model_short"])
        acc  = _fmt_median_iqr(row["median_acc"],  row["q25_acc"],  row["q75_acc"])
        prec = _fmt_median_iqr(row["median_prec"], row["q25_prec"], row["q75_prec"])
        rec  = _fmt_median_iqr(row["median_rec"],  row["q25_rec"],  row["q75_rec"])
        model_lines.append(f"{name} (ACC++ signals) &\n{acc} &\n{prec} &\n{rec} \\\\")

    body = "\n".join(model_lines)

    latex = (
        r"\begin{table}[t]" "\n"
        r"\caption{ACC++ signals achieve nontrivial autointerpretation performance." "\n"
        r"We report median fuzzing metrics with interquartile range (25\%--75\%).}" "\n"
        r"\label{tab:interpretability-comparison}" "\n"
        r"\centering" "\n"
        r"\resizebox{\columnwidth}{!}{%" "\n"
        r"\begin{tabular}{lccc}" "\n"
        r"\toprule" "\n"
        r"Model & Accuracy & Precision & Recall \\" "\n"
        r"\midrule" "\n"
        f"{_SAE_ROW}\n"
        r"\midrule" "\n"
        f"{body}\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"}" "\n"
        r"\end{table}" "\n"
    )

    out_path = str(Path(output_dir) / "table_interpretability.tex")
    Path(out_path).write_text(latex)
    log.info(f"Saved interpretability table: {out_path}")


# ---------------------------------------------------------------------------
# Aggregate FDR barplot + CSV (one bar per model, Wilson CIs)
# ---------------------------------------------------------------------------

def _figure_fdr_per_model(
    full_df: pd.DataFrame,
    models: list[str],
    output_dir: str,
) -> None:
    """Save aggregate FDR barplot and CSV with Wilson 95% CIs.

    Matches ``plots_paper.ipynb`` cell 31: one bar per model showing the
    overall fraction of FDR-significant signals with Wilson CIs.
    Also saves the ``agg`` DataFrame as ``fdr_per_model.csv``.

    Args:
        full_df: Combined per-signal DataFrame from the pipeline.
        models: Ordered list of model names.
        output_dir: Directory for the output PDF and CSV.
    """
    _setup_paper_rc()

    model_shorts = [m.split("/")[-1] for m in models]
    present = [m for m in model_shorts if m in full_df["model"].unique()]
    if not present:
        return

    # Filter to valid (non-NaN fisher) signals with FDR results
    valid = full_df[full_df["fisher_p_value"].notna()].copy()
    valid["fdr_significant"] = valid["fdr_significant"].astype(bool)

    # Aggregate per model: total count, significant count, fraction
    g = valid.groupby("model")["fdr_significant"]
    agg = pd.DataFrame({
        "model": g.size().index,
        "n": g.size().values,
        "k": g.sum().values,
    })
    agg["frac"] = agg["k"] / agg["n"]

    # Wilson 95% CIs
    ci_low, ci_high = proportion_confint(
        count=agg["k"].to_numpy(),
        nobs=agg["n"].to_numpy(),
        alpha=0.05,
        method="wilson",
    )
    agg["ci_low"] = ci_low
    agg["ci_high"] = ci_high

    # Display names for plotting
    agg["model_display"] = agg["model"].map(_MODEL_DISPLAY_NAMES).fillna(agg["model"])

    # Save CSV (before renaming for plot)
    csv_path = str(Path(output_dir) / "fdr_per_model.csv")
    agg.to_csv(csv_path, index=False)
    log.info(f"Saved FDR CSV: {csv_path}")

    # Plot order: match models arg order
    order = [_MODEL_DISPLAY_NAMES.get(m, m) for m in present
             if _MODEL_DISPLAY_NAMES.get(m, m) in agg["model_display"].values]

    # Build plot data in order
    agg_plot = agg.set_index("model_display").loc[order].reset_index()

    plt.figure(figsize=(3.1, 1.5))
    x_pos = np.arange(len(agg_plot))
    y = agg_plot["frac"].to_numpy()
    yerr = np.vstack([
        y - agg_plot["ci_low"].to_numpy(),
        agg_plot["ci_high"].to_numpy() - y,
    ])

    plt.bar(x_pos, y, color="#1f77b4")
    plt.errorbar(x_pos, y, yerr=yerr, fmt="none", capsize=4, linewidth=1, color="black")

    plt.xticks(x_pos, agg_plot["model_display"])
    plt.ylabel("Frac. of signals\n(FDR \u2264 5\%)")
    plt.xlabel(None)

    plt.tight_layout()
    out_path = str(Path(output_dir) / "fraction_significant_fdr_per_model.pdf")
    plt.savefig(out_path, bbox_inches="tight", dpi=800)
    plt.close()
    log.info(f"Saved FDR per-model figure: {out_path}")


# ---------------------------------------------------------------------------
# FDR significance per layer (one PDF per model)
# ---------------------------------------------------------------------------

def _figure_fdr_significance(
    full_df: pd.DataFrame,
    models: list[str],
    output_dir: str,
) -> None:
    """Save per-layer FDR-corrected significance fraction (one PDF per model).

    Matches ``plots_paper.ipynb`` cells 38/45/52: paper-sized figures with
    thick Wilson CI errorbars and 1-indexed layer x-ticks.

    Args:
        full_df: Combined per-signal DataFrame from the pipeline.
        models: Ordered list of model names.
        output_dir: Directory for the output PDFs.
    """
    _setup_paper_rc()
    bar_color = "#1f77b4"
    ci_color = "black"

    for model_name in models:
        model_short = model_name.split("/")[-1]
        if model_short not in full_df["model"].unique():
            continue

        mdf = full_df[full_df["model"] == model_short]
        valid = mdf[mdf["fisher_p_value"].notna()]

        per_layer = (
            valid.groupby("layer")
            .agg(
                n_signals=("fdr_significant", "size"),
                n_significant=("fdr_significant", "sum"),
            )
            .assign(fraction_significant=lambda d: d["n_significant"] / d["n_signals"])
            .reset_index()
            .sort_values("layer")
        )

        ci_low, ci_high = proportion_confint(
            count=per_layer["n_significant"].to_numpy(),
            nobs=per_layer["n_signals"].to_numpy(),
            alpha=0.05,
            method="wilson",
        )

        y = per_layer["fraction_significant"].to_numpy()
        yerr = np.vstack([y - ci_low, ci_high - y])

        n_layers = len(per_layer)
        plt.figure(figsize=(3.25, 2))
        plt.bar(range(n_layers), y, color=bar_color)
        plt.errorbar(
            range(n_layers), y, yerr=yerr,
            fmt="none", ecolor=ci_color,
            elinewidth=1.5, capsize=3.0, capthick=1.0,
        )

        # Layers already start at 1 (no layer-0 signals)
        plt.xticks(range(n_layers), per_layer["layer"].to_numpy())
        plt.ylabel("Frac. of signals (FDR \u2264 5%)")
        plt.xlabel("Layer")

        plt.tight_layout()
        suffix = _MODEL_FILENAME_SUFFIX.get(model_short, model_short)
        out_path = str(Path(output_dir) / f"fraction_significant_fdr_5pct_{suffix}.pdf")
        plt.savefig(out_path, bbox_inches="tight", dpi=800)
        plt.close()
        log.info(f"Saved FDR figure: {out_path}")


# ---------------------------------------------------------------------------
# Per-layer median metrics with IQR bands (one PDF per model)
# ---------------------------------------------------------------------------

def _figure_layer_metrics(
    full_df: pd.DataFrame,
    models: list[str],
    output_dir: str,
) -> None:
    """Save per-layer median accuracy/precision/recall with IQR (one PDF per model).

    Matches ``plots_paper.ipynb`` cells 40/47/54: paper-sized figures with
    overall-median horizontal dashed lines and 1-indexed layer x-axis.
    Gemma uses sparse x-tick labels (1, 5, 10, 15, 20, 25).

    Args:
        full_df: Combined per-signal DataFrame from the pipeline.
        models: Ordered list of model names.
        output_dir: Directory for the output PDFs.
    """
    _setup_paper_rc()

    for model_name in models:
        model_short = model_name.split("/")[-1]
        if model_short not in full_df["model"].unique():
            continue

        mdf = full_df[full_df["model"] == model_short].copy()

        per_layer = (
            mdf
            .groupby("layer")
            .agg(
                accuracy_median=("accuracy", "median"),
                accuracy_q25=("accuracy", lambda x: x.quantile(0.25)),
                accuracy_q75=("accuracy", lambda x: x.quantile(0.75)),
                precision_median=("precision", "median"),
                precision_q25=("precision", lambda x: x.quantile(0.25)),
                precision_q75=("precision", lambda x: x.quantile(0.75)),
                recall_median=("recall", "median"),
                recall_q25=("recall", lambda x: x.quantile(0.25)),
                recall_q75=("recall", lambda x: x.quantile(0.75)),
            )
            .reset_index()
            .sort_values("layer")
        )

        # Layers already start at 1 (no layer-0 signals)
        x = per_layer["layer"].to_numpy()

        # Ensure numeric
        for c in [
            "accuracy_median", "accuracy_q25", "accuracy_q75",
            "precision_median", "precision_q25", "precision_q75",
            "recall_median", "recall_q25", "recall_q75",
        ]:
            per_layer[c] = pd.to_numeric(per_layer[c], errors="coerce")

        plt.figure(figsize=(2.5, 1.5))

        acc_line, = plt.plot(x, per_layer["accuracy_median"], label="Accuracy")
        plt.fill_between(x, per_layer["accuracy_q25"], per_layer["accuracy_q75"], alpha=0.2)

        prec_line, = plt.plot(x, per_layer["precision_median"], label="Precision")
        plt.fill_between(x, per_layer["precision_q25"], per_layer["precision_q75"], alpha=0.2)

        rec_line, = plt.plot(x, per_layer["recall_median"], label="Recall")
        plt.fill_between(x, per_layer["recall_q25"], per_layer["recall_q75"], alpha=0.2)

        # Overall-median horizontal dashed lines (with labels)
        plt.axhline(per_layer["accuracy_median"].median(), color=acc_line.get_color(),
                     linestyle="--", linewidth=1, label="Accuracy median (all layers)")
        plt.axhline(per_layer["precision_median"].median(), color=prec_line.get_color(),
                     linestyle="--", linewidth=1, label="Precision median (all layers)")
        plt.axhline(per_layer["recall_median"].median(), color=rec_line.get_color(),
                     linestyle="--", linewidth=1, label="Recall median (all layers)")

        # X-tick formatting: Gemma has many layers → sparse labels
        if model_short == "gemma-2-2b":
            tick_labels = [str(t) if (t == 1 or t % 5 == 0) else "" for t in x]
            plt.xticks(x, tick_labels)
        else:
            plt.xticks(x, x)

        plt.ylabel("Metric value")
        plt.xlabel("Layer")

        plt.tight_layout()
        out_path = str(Path(output_dir) / f"median_metrics_by_layer_{model_short}.pdf")
        plt.savefig(out_path, bbox_inches="tight", dpi=800)
        plt.close()
        log.info(f"Saved layer metrics figure: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: Compute metrics, generate Table 1 and paper figures."
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        required=True,
        help="TransformerLens model names (space-separated, e.g. gpt2-small pythia-160m).",
    )
    parser.add_argument(
        "--task",
        default="ioi-balanced",
        help="Task identifier (default: ioi-balanced).",
    )
    parser.add_argument(
        "--activations_dir",
        default="data/autointerp/",
        help="Directory containing activation H5 files (default: data/autointerp/).",
    )
    parser.add_argument(
        "--output_dir",
        default="data/autointerp/results/",
        help="Directory for all output files (default: data/autointerp/results/).",
    )
    args = parser.parse_args()

    compute_all_metrics(
        models=args.models,
        task=args.task,
        activations_dir=args.activations_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
