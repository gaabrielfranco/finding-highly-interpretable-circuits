"""Stage 2b: Merge interpretation shard H5 files into the main activation H5.

After ``interpret_signals.py`` runs in parallel (potentially across many
SCC array jobs), each job produces a shard H5 file covering a slice of
signals for one layer. This script merges all shard files for every
requested layer back into the main activation H5.

Only non-empty shard entries overwrite the main file — already-filled
slots are preserved (shard-safe merge). This allows partial re-runs.

Usage:
    python merge_shards.py \\
        --model gpt2-small \\
        --task ioi-balanced \\
        --layers 1 2 3 4 5 6 7 8 9 10 11 \\
        --activations_dir data/autointerp/ \\
        --shards_dir data/autointerp/shards/
"""

import argparse
import glob
import logging
import re
from pathlib import Path

import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Only topk signals have interpretations generated (random indices are not
# sent to the explainer LLM).
_INDICE_TYPE = "topk"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_start_end_from_filename(path: str) -> tuple[int, int]:
    """Parse the global start and end indices from a shard filename.

    Shard files follow the naming convention
    ``activations_{layer}_balanced_{start}_{end}.h5``.

    Args:
        path: Full path or filename of the shard H5 file.

    Returns:
        Tuple of (start, end) as parsed from the filename.

    Raises:
        ValueError: If the filename does not match the expected pattern.
    """
    m = re.search(r"_(\d+)_(\d+)\.h5$", path)
    if not m:
        raise ValueError(f"Could not parse start/end from shard filename: {path}")
    return int(m.group(1)), int(m.group(2))


def _is_empty(x: object) -> bool:
    """Return True if an H5 string value is empty or None.

    h5py may return ``bytes`` or ``str`` depending on dtype and settings.

    Args:
        x: Value read from an H5 string dataset.

    Returns:
        True if the value is empty (``None``, zero-length bytes, or empty
        string), False otherwise.
    """
    if x is None:
        return True
    if isinstance(x, (bytes, np.bytes_)):
        return len(x) == 0
    if isinstance(x, str):
        return x == ""
    return False


# ---------------------------------------------------------------------------
# Per-layer merge
# ---------------------------------------------------------------------------

def merge_layer_shards(
    layer: int,
    main_file: str,
    shards_dir: str,
) -> None:
    """Merge all interpretation shard files for one layer into the main H5.

    Scans ``shards_dir`` for files matching
    ``activations_{layer}_balanced*.h5``, reads each shard's
    ``topk_interpretations`` and ``topk_raw_answers`` datasets, and writes
    non-empty entries into the corresponding slice of the main file.
    Already-filled slots in the main file are not overwritten.

    The main file must already contain the ``layer_{layer}`` group with
    ``topk_indices`` (to determine total feature count). The
    ``topk_interpretations`` and ``topk_raw_answers`` datasets are created
    in the main file if they do not exist yet.

    Args:
        layer: Layer index to process.
        main_file: Path to the main activation H5 file (Stage 1 output,
            progressively filled by Stages 2 and 3).
        shards_dir: Directory containing shard H5 files produced by
            ``interpret_signals.py``.
    """
    layer_name = f"layer_{layer}"
    indice_type = _INDICE_TYPE

    shard_pattern = str(Path(shards_dir) / f"activations_{layer}_balanced*.h5")
    shards = sorted(glob.glob(shard_pattern))

    if not shards:
        log.warning(f"No shards found for layer {layer} in {shards_dir}")
        return

    log.info(f"Layer {layer}: found {len(shards)} shard(s)")

    with h5py.File(main_file, "r+") as f_main:
        if layer_name not in f_main:
            raise KeyError(f"{layer_name} not found in {main_file}")

        grp_main = f_main[layer_name]

        idx_name = f"{indice_type}_indices"
        if idx_name not in grp_main:
            raise KeyError(f"Missing {idx_name} in {main_file}:{layer_name}")

        total_features = grp_main[idx_name].shape[0]

        # Ensure output datasets exist in main file
        dt = h5py.string_dtype(encoding="utf-8")
        interp_name = f"{indice_type}_interpretations"
        raw_name = f"{indice_type}_raw_answers"

        if interp_name not in grp_main:
            grp_main.create_dataset(interp_name, (total_features,), dtype=dt)
        if raw_name not in grp_main:
            grp_main.create_dataset(raw_name, (total_features,), dtype=dt)

        dset_interp_main = grp_main[interp_name]
        dset_raw_main = grp_main[raw_name]

        for shard_path in shards:
            with h5py.File(shard_path, "r") as f_shard:
                if layer_name not in f_shard:
                    raise KeyError(f"{layer_name} not found in shard {shard_path}")

                grp_shard = f_shard[layer_name]
                if interp_name not in grp_shard or raw_name not in grp_shard:
                    raise KeyError(
                        f"Shard {shard_path} missing datasets "
                        f"{interp_name}/{raw_name}"
                    )

                dset_interp_shard = grp_shard[interp_name]
                dset_raw_shard = grp_shard[raw_name]
                shard_len = dset_interp_shard.shape[0]

                # Determine global start/end from attrs or filename fallback
                if "global_start" in grp_shard.attrs and "global_end" in grp_shard.attrs:
                    gstart = int(grp_shard.attrs["global_start"])
                    gend = int(grp_shard.attrs["global_end"])
                else:
                    gstart_fn, gend_fn = _parse_start_end_from_filename(shard_path)
                    gstart = gstart_fn
                    # Prefer actual shard length (filename end may overshoot)
                    gend = gstart + shard_len
                    if gend > gend_fn:
                        gend = gend_fn

                # Clamp to main bounds
                if gstart < 0:
                    raise ValueError(
                        f"Negative global_start in shard {shard_path}: {gstart}"
                    )
                if gstart >= total_features:
                    log.info(
                        f"Skipping shard {shard_path}: "
                        f"start {gstart} >= total_features {total_features}"
                    )
                    continue

                gend = min(gend, total_features)
                expected_len = gend - gstart
                if expected_len <= 0:
                    log.info(
                        f"Skipping shard {shard_path}: empty range after clamping"
                    )
                    continue

                if expected_len != shard_len:
                    # If mismatch, only use the overlap
                    use_len = min(expected_len, shard_len)
                    log.warning(
                        f"{shard_path}: shard_len={shard_len}, "
                        f"range_len={expected_len}. Using {use_len}."
                    )
                else:
                    use_len = shard_len

                # Read shard data (local indexing always starts at 0)
                interp_vals = dset_interp_shard[0:use_len]
                raw_vals = dset_raw_shard[0:use_len]

                # Read current main slice, then only overwrite non-empty entries
                main_interp_slice = dset_interp_main[gstart:gstart + use_len]
                main_raw_slice = dset_raw_main[gstart:gstart + use_len]

                # Convert to mutable Python lists (safe for mixed bytes/str)
                main_interp_list = list(main_interp_slice)
                main_raw_list = list(main_raw_slice)

                wrote = 0
                for i in range(use_len):
                    if not _is_empty(interp_vals[i]):
                        main_interp_list[i] = interp_vals[i]
                        main_raw_list[i] = raw_vals[i]
                        wrote += 1

                # Write back
                dset_interp_main[gstart:gstart + use_len] = main_interp_list
                dset_raw_main[gstart:gstart + use_len] = main_raw_list

                log.info(
                    f"Merged {shard_path}: wrote {wrote}/{use_len} entries "
                    f"into [{gstart}, {gstart + use_len})"
                )

        f_main.flush()
    log.info(f"Layer {layer}: done.")


# ---------------------------------------------------------------------------
# Top-level function
# ---------------------------------------------------------------------------

def merge_shards(
    model_name: str,
    task: str,
    layers: list[int],
    activations_dir: str,
    shards_dir: str,
) -> None:
    """Merge all interpretation shards for the given layers into their main H5s.

    Args:
        model_name: TransformerLens model name (e.g. ``"gpt2-small"``).
            Used to locate the activation H5 files via the pattern
            ``activations_{layer}_{task}_{model_short}.h5``.
        task: Task identifier (e.g. ``"ioi-balanced"``).
        layers: List of layer indices to merge.
        activations_dir: Directory containing the main activation H5 files
            (Stage 1 output).
        shards_dir: Directory containing shard H5 files (Stage 2 output).
    """
    model_short = model_name.split("/")[-1]

    for layer in layers:
        log.info(f"\nProcessing layer {layer}")
        main_file = str(
            Path(activations_dir) / f"activations_{layer}_{task}_{model_short}.h5"
        )
        merge_layer_shards(
            layer=layer,
            main_file=main_file,
            shards_dir=shards_dir,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2b: Merge interpretation shard H5 files into the main activation H5."
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="TransformerLens model name (e.g. gpt2-small).",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task identifier (e.g. ioi-balanced).",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
        help="Layer indices to merge (space-separated, e.g. 1 2 3 ... 11).",
    )
    parser.add_argument(
        "--activations_dir",
        default="data/autointerp/",
        help="Directory containing main activation H5 files (default: data/autointerp/).",
    )
    parser.add_argument(
        "--shards_dir",
        default="data/autointerp/shards/",
        help="Directory containing shard H5 files (default: data/autointerp/shards/).",
    )
    args = parser.parse_args()

    merge_shards(
        model_name=args.model,
        task=args.task,
        layers=args.layers,
        activations_dir=args.activations_dir,
        shards_dir=args.shards_dir,
    )


if __name__ == "__main__":
    main()
