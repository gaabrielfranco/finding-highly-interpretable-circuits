"""Shared H5 I/O utilities for the autointerp pipeline."""

from typing import Tuple

import h5py
import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor


# ---------------------------------------------------------------------------
# H5 field name constants
# ---------------------------------------------------------------------------

TOPK_VALUES_KEY = "topk_values"
TOPK_INDICES_KEY = "topk_indices"
RANDOM_VALUES_KEY = "random_values"
RANDOM_INDICES_KEY = "random_indices"

COMPRESSION = "lzf"


# ---------------------------------------------------------------------------
# Signal H5 I/O
# ---------------------------------------------------------------------------

def load_layer_signals(
    filename: str,
    layer_idx: int,
    device: str = "cuda",
) -> Tuple[
    Float[Tensor, "n_signals d_model"],
    Float[Tensor, "n_signals d_model"],
    Int[Tensor, "n_signals 7"],
    np.ndarray,
    np.ndarray,
]:
    """Load signal vectors and metadata for one layer from an H5 file.

    The H5 file format (produced by ``extract_signals.py``) has groups
    ``layer_0``, ``layer_1``, etc., each containing:

    - ``S_U``: destination-side signal vectors, shape (n_signals, d_model)
    - ``S_V``: source-side signal vectors, shape (n_signals, d_model)
    - ``metadata``: int32 array, shape (n_signals, 7), columns:
      [prompt_id, downstream_ah_idx, downstream_dest_token,
       downstream_src_token, upstream_layer, upstream_component_id,
       upstream_src_token]
    - ``edge_type``: bytes array of shape (n_signals,), values ``b"d"``
      or ``b"s"``
    - ``edges``: structured array with fields (u, v, key)

    Args:
        filename: Path to the signals H5 file.
        layer_idx: Layer index to load (group ``layer_{layer_idx}``).
        device: Torch device string for the returned tensors.

    Returns:
        Tuple of (S_U, S_V, metadata, edge_type, edges) where:
        - S_U, S_V are float32 tensors on ``device``
        - metadata is an int32 tensor (CPU)
        - edge_type is a numpy str array (values "d" or "s")
        - edges is a structured numpy array with fields (u, v, key)
    """
    with h5py.File(filename, "r") as f:
        grp_name = f"layer_{layer_idx}"
        grp = f[grp_name]

        S_U = torch.from_numpy(grp["S_U"][:]).to(device)
        S_V = torch.from_numpy(grp["S_V"][:]).to(device)

        metadata = torch.from_numpy(grp["metadata"][:])

        edge_type = grp["edge_type"][:].astype(str)

        edges = grp["edges"][:]

    return S_U, S_V, metadata, edge_type, edges


# ---------------------------------------------------------------------------
# Activation H5 I/O
# ---------------------------------------------------------------------------

def save_layer_activations(
    filename: str,
    layer_idx: int,
    topk_vals: torch.Tensor,
    topk_idx: torch.Tensor,
    rand_vals: torch.Tensor,
    rand_idx: torch.Tensor,
) -> None:
    """Save top-K and random activation results for one layer to an H5 file.

    Creates (or overwrites) ``filename`` with a single group
    ``layer_{layer_idx}`` containing four datasets:

    - ``topk_values``: shape (n_signals, k), float
    - ``topk_indices``: shape (n_signals, k, 3), int — [sentence_id, dest_token, src_token]
    - ``random_values``: shape (n_signals, k), float
    - ``random_indices``: shape (n_signals, k, 3), int

    All datasets are compressed with ``lzf``.

    Args:
        filename: Output H5 file path.
        layer_idx: Layer index (used to name the H5 group).
        topk_vals: Top-K activation scores, shape (n_signals, k).
        topk_idx: Top-K activation indices, shape (n_signals, k, 3).
        rand_vals: Random activation scores, shape (n_signals, k).
        rand_idx: Random activation indices, shape (n_signals, k, 3).
    """
    with h5py.File(filename, "w") as f:
        grp_name = f"layer_{layer_idx}"

        # If we are re-running a layer, delete the old group to avoid conflicts
        if grp_name in f:
            del f[grp_name]

        grp = f.create_group(grp_name)

        grp.create_dataset(TOPK_VALUES_KEY,   data=topk_vals.cpu().numpy(), compression=COMPRESSION)
        grp.create_dataset(TOPK_INDICES_KEY,  data=topk_idx.cpu().numpy(),  compression=COMPRESSION)

        grp.create_dataset(RANDOM_VALUES_KEY, data=rand_vals.cpu().numpy(), compression=COMPRESSION)
        grp.create_dataset(RANDOM_INDICES_KEY, data=rand_idx.cpu().numpy(), compression=COMPRESSION)

        print(f"Saved results for {grp_name} to {filename}")


# ---------------------------------------------------------------------------
# Edge decoder
# ---------------------------------------------------------------------------

def parse_edge(edge: np.void) -> Tuple[str, str, int]:
    """Decode a structured edge tuple loaded from an H5 ``edges`` dataset.

    The ``edges`` dataset uses a structured numpy dtype with fields
    ``(u, v, key)`` where ``u`` and ``v`` are UTF-8 encoded byte strings
    and ``key`` is an int32 multigraph key.

    Args:
        edge: A single row from the ``edges`` structured numpy array,
            as returned by ``load_layer_signals``.

    Returns:
        Tuple of (source_node, target_node, edge_key) where source and
        target are decoded Python strings and key is an int.
    """
    source = edge[0].decode('utf-8')
    target = edge[1].decode('utf-8')
    key = int(edge[2])

    return source, target, key
