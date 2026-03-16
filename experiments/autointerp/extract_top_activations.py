"""Stage 1: Extract top-K and random activations from Pile residuals.

For each signal vector in the signals H5 file (one layer at a time), streams
through the pre-computed Pile residuals (pile_activations H5) and finds the
top-K sentence/token pairs where the signal fires most strongly, plus a
random-K sample for unbiased baseline comparison.

The activation score for a (destination_token, source_token) pair is the
product of the signal projections:  (x^d · S_U) * (x^s · S_V), with RoPE
rotation applied when the model uses rotary position embeddings.

Output file: {output_dir}/activations_{layer}_{task}_{model_short}.h5
  Group: layer_{layer_idx}
  Datasets:
    topk_values:   (n_signals, k)    float   — top-K activation scores
    topk_indices:  (n_signals, k, 3) int     — [sentence_id, dest_token, src_token]
    random_values: (n_signals, k)    float
    random_indices:(n_signals, k, 3) int

Usage:
    python extract_top_activations.py \\
        --model gpt2-small \\
        --task ioi-balanced \\
        --layer 3 \\
        --signals_file data/signals_balanced_gpt2-small_not-norm.h5 \\
        --pile_activations data/autointerp/pile_activations_gpt2-small.h5 \\
        --output_dir data/autointerp/ \\
        --device cuda

    # Pythia (use numpy SVD for numerical stability)
    python extract_top_activations.py \\
        --model EleutherAI/pythia-160m \\
        --task ioi-balanced \\
        --layer 5 \\
        --signals_file data/signals_balanced_pythia-160m_not-norm.h5 \\
        --pile_activations data/autointerp/pile_activations_pythia-160m.h5 \\
        --output_dir data/autointerp/ \\
        --device cuda \\
        --use_numpy_svd
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from einops import einsum
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer

from accpp_tracer.decomposition import compute_weight_pseudoinverses
from accpp_tracer.models import get_model_config
from accpp_tracer.rope import get_rotation_matrix 
from h5_utils import load_layer_signals, save_layer_activations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# Dataset — kept exactly from original
# ---------------------------------------------------------------------------

class ThePileH5Dataset(Dataset):
    """Lazy-loading H5 dataset for pre-computed Pile residuals.

    Opens the H5 file on demand in each DataLoader worker (h5py is not
    thread-safe; opening once per worker avoids contention).

    Args:
        h5_path: Path to the pile_activations H5 file.
        dataset_name: H5 dataset name to load (e.g. ``"L3"`` for layer 3).
    """

    def __init__(self, h5_path: str, dataset_name: str) -> None:
        self.h5_path = h5_path
        self.dataset_name = dataset_name
        self.f = None  # File handle, will be opened by worker

        # Open once to get dimensions
        with h5py.File(self.h5_path, "r") as f:
            self.shape = f[self.dataset_name].shape
            self.length = self.shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> np.ndarray:
        # Open the file on demand in each worker process.
        # This is crucial as h5py objects are not thread-safe.
        if self.f is None:
            self.f = h5py.File(self.h5_path, "r")

        # Shape: (seq_len, d_model)
        return self.f[self.dataset_name][idx]


# ---------------------------------------------------------------------------
# Reservoir sampler — kept exactly from original
# ---------------------------------------------------------------------------

def update_reservoir(
    global_vals: Tensor,
    global_meta: Tensor,
    batch_vals: Tensor,
    batch_meta: Tensor,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Merge batch results into global top-K reservoir.

    Keeps the top-K largest values from the combined pool
    (current global best + new batch candidates).

    Args:
        global_vals: Current global top-K scores, shape (n_signals, k).
        global_meta: Metadata for current top-K, shape (n_signals, k, 3).
        batch_vals: Batch top-K scores, shape (n_signals, k).
        batch_meta: Batch top-K metadata, shape (n_signals, k, 3).
        k: Number of entries to keep.

    Returns:
        Tuple of (new_global_vals, new_global_meta) after merging.
    """
    # 1. Concatenate current global best with new batch candidates
    # Shape: (n_signals, 2*k)
    combined_vals = torch.cat([global_vals, batch_vals], dim=1)
    combined_meta = torch.cat([global_meta, batch_meta], dim=1)

    # 2. Find the new winners (Top K)
    best_vals, best_indices = combined_vals.topk(k, dim=1)

    # 3. Gather the metadata for the winners
    # Expand indices to match metadata dims: (n_signals, k, 3)
    gather_idx = best_indices.unsqueeze(-1).expand(-1, -1, 3)
    best_meta = torch.gather(combined_meta, 1, gather_idx)

    return best_vals, best_meta


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_top_activations(
    model_name: str,
    task: str,
    layer_idx: int,
    signals_file: str,
    pile_activations_file: str,
    output_dir: str = ".",
    device: str = "cuda",
    batch_size: int = 8,
    num_workers: int = 16,
    k: int = 40,
    seed: int = 42,
    use_numpy_svd: bool = False,
) -> str:
    """Extract top-K and random activations for one layer.

    Streams through all Pile residuals for the given layer, scoring each
    (sentence, dest_token, src_token) triple against every signal vector,
    and retains the top-K highest-scoring triples plus a random-K sample.

    Args:
        model_name: TransformerLens model name (e.g. ``"gpt2-small"``).
        task: Task label used only in the output filename (e.g.
            ``"ioi-balanced"``).
        layer_idx: Layer index to process (must match the signals H5).
        signals_file: Path to the signals H5 file (from
            ``extract_signals.py``).
        pile_activations_file: Path to the pile activations H5 file (from
            ``collect_pile_activations.py``).
        output_dir: Directory for the output H5 file.
        device: Torch device string.
        batch_size: DataLoader batch size.
        num_workers: Number of DataLoader worker processes.
        k: Number of top-K and random-K entries to retain per signal.
        seed: Random seed for the random reservoir.
        use_numpy_svd: Use ``numpy.linalg.pinv`` instead of
            ``torch.linalg.pinv`` for computing weight pseudoinverses.
            Recommended for Pythia-160m (numerical stability).

    Returns:
        Path to the output H5 file.

    Raises:
        SystemExit: If the output file already exists (refuses to overwrite).
    """
    model_short = model_name.split("/")[-1]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_h5 = str(Path(output_dir) / f"activations_{layer_idx}_{task}_{model_short}.h5")

    if os.path.exists(out_h5):
        raise SystemExit(f"File already exists, refusing to overwrite: {out_h5}")

    log.info(f"Model: {model_name}")
    log.info(f"Task: {task}")
    log.info(f"Layer: {layer_idx}")
    log.info(f"signals_file={signals_file}")
    log.info(f"pile_activations_file={pile_activations_file}")

    log.info(f"Loading model: {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    model_config = get_model_config(model, use_numpy_svd=use_numpy_svd)

    # Read chunk size from pile activations attrs
    log.info(f"Reading chunk_size from {pile_activations_file}")
    with h5py.File(pile_activations_file, "r") as pf:
        chunk_size: int = int(pf.attrs["seq_len"])

    log.info(f"chunk_size={chunk_size}, k={k}")

    # ------------------------------------------------------------------
    # Compute weight pseudoinverses
    # ------------------------------------------------------------------
    log.info("Computing weight pseudoinverses...")
    W_Q_pinv_all, W_K_pinv_all = compute_weight_pseudoinverses(model, model_config, device)
    W_Q_pinv: Tensor = W_Q_pinv_all[layer_idx]  # (n_heads, d_head, d_model)
    W_K_pinv: Tensor = W_K_pinv_all[layer_idx]  # (n_kv_heads, d_head, d_model)
    del W_Q_pinv_all, W_K_pinv_all
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Bias offset constants (c_d, c_s)
    # ------------------------------------------------------------------
    use_bias_offset: bool = model.b_Q[layer_idx].any().item()
    log.info(f"use_bias_offset={use_bias_offset}")

    if use_bias_offset:
        c_d = einsum(
            model.b_Q[layer_idx], W_Q_pinv,
            "n_heads d_head, n_heads d_head d_model -> n_heads d_model",
        )
        c_s = einsum(
            model.b_K[layer_idx], W_K_pinv,
            "n_heads d_head, n_heads d_head d_model -> n_heads d_model",
        )
        assert c_d.shape == c_s.shape == (model.cfg.n_heads, model.cfg.d_model)
    else:
        c_d, c_s = None, None

    # ------------------------------------------------------------------
    # RoPE rotation matrices and projection maps (M_d_all, M_s_all)
    # ------------------------------------------------------------------
    if model_config.has_rope:
        log.info("Computing rotation matrices...")
        R = torch.stack([get_rotation_matrix(model, i, device) for i in range(chunk_size + 1)])
        # (chunk_size+1, d_head, d_head)

        M_d_all = einsum(
            model.W_Q[layer_idx],
            R.transpose(1, 2),
            W_Q_pinv,
            "n_heads d_model d_head_a, n_tokens d_head_a d_head_b, n_heads d_head_b d_model_out"
            " -> n_heads n_tokens d_model d_model_out",
        )
        # (n_heads, chunk_size+1, d_model, d_model)

        Wpinv_T = W_K_pinv.transpose(-1, -2)           # (n_kv_heads, d_model, d_head)
        WK_T = model.W_K.transpose(-1, -2)              # (n_layers, n_kv_heads, d_head, d_model)

        M_s_all = einsum(
            Wpinv_T,
            R,
            WK_T[layer_idx],
            "head d_model d_head_a, n_tokens d_head_a d_head_b, head d_head_b d_model_out"
            " -> head n_tokens d_model d_model_out",
        )
        # (n_heads, chunk_size+1, d_model, d_model)

        del Wpinv_T, WK_T, R

    del W_Q_pinv, W_K_pinv
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Load signals
    # ------------------------------------------------------------------
    S_U, S_V, metadata, edge_type, edges = load_layer_signals(signals_file, layer_idx, device)
    # Normalize to unit norm — signals H5 stores unnormalized vectors
    S_U = S_U / S_U.norm(dim=-1, keepdim=True)
    S_V = S_V / S_V.norm(dim=-1, keepdim=True)
    n_signals = S_U.shape[0]
    log.info(f"n_signals={n_signals}")

    head_indices = metadata[:, 1].to(torch.long)  # column 1 = downstream_ah_idx
    assert ((head_indices < model.cfg.n_heads) & (head_indices >= 0)).all()

    # ------------------------------------------------------------------
    # Bias corrections for non-RoPE path (precomputed, token-independent)
    # ------------------------------------------------------------------
    if not model_config.has_rope:
        # For models without RoPE, bias corrections are scalar per signal
        relevant_c_d = c_d[head_indices]  # (n_signals, d_model)
        relevant_c_s = c_s[head_indices]  # (n_signals, d_model)
        bias_correction_u = einsum(
            relevant_c_d, S_U,
            "n_signals d_model, n_signals d_model -> n_signals",
        ).reshape(1, 1, -1)
        bias_correction_v = einsum(
            relevant_c_s, S_V,
            "n_signals d_model, n_signals d_model -> n_signals",
        ).reshape(1, 1, -1)
    else:
        bias_correction_u = None
        bias_correction_v = None

    # ------------------------------------------------------------------
    # Initialize reservoirs
    # ------------------------------------------------------------------
    topk_vals = torch.full((n_signals, k), -float("inf"), device=device)
    topk_idx  = torch.zeros((n_signals, k, 3), dtype=torch.long, device=device)

    rand_vals = torch.full((n_signals, k), -float("inf"), device=device)
    rand_idx  = torch.zeros((n_signals, k, 3), dtype=torch.long, device=device)

    # Causal mask: (1, chunk_size, chunk_size, 1) — broadcast over batch and signals
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=device)).view(
        1, chunk_size, chunk_size, 1
    )

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------
    dataset = ThePileH5Dataset(pile_activations_file, f"L{layer_idx}")
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # ------------------------------------------------------------------
    # Main streaming loop
    # ------------------------------------------------------------------
    current_global_idx = 0
    for residuals_batch_cpu in tqdm(data_loader, desc=f"Layer {layer_idx}"):
        x = residuals_batch_cpu.to(device, non_blocking=True)
        # x: (B, chunk_size, d_model)

        B, C, _ = x.shape

        # --- A. Batched Projection ---
        # Non-RoPE:  u_proj[d] = (x^d + c^d) · S_U,  v_proj[s] = (x^s + c^s) · S_V
        # RoPE:      u_proj[d] = ((x^d + c^d) M_d) · S_U,
        #            v_proj[s] = ((x^s + c^s) M_s^T) · S_V
        # (TransformerLens row-vector convention)

        if not model_config.has_rope:
            # Pure projections: (B, C, d_model) @ (d_model, n_signals) -> (B, C, n_signals)
            u_proj_pure = x @ S_U.T
            v_proj_pure = x @ S_V.T

            # Add token-independent bias corrections (broadcast over batch/tokens)
            u_proj = u_proj_pure + bias_correction_u
            v_proj = v_proj_pure + bias_correction_v

        else:
            # RoPE-aware projections are head- and position-dependent.
            u_proj = torch.empty((B, C, n_signals), device=device, dtype=x.dtype)
            v_proj = torch.empty((B, C, n_signals), device=device, dtype=x.dtype)

            for h in range(model.cfg.n_heads):
                sig_ids = (head_indices == h).nonzero(as_tuple=False).squeeze(-1)
                if sig_ids.numel() == 0:
                    continue

                S_U_h = S_U[sig_ids]  # (n_sig_h, d_model)
                S_V_h = S_V[sig_ids]  # (n_sig_h, d_model)

                # Rotation maps for this head, all token positions in the chunk.
                # M_d_all: dest-side map; M_s_all: src-side map (pre-transposed for row-vec matmul).
                M_d_h = M_d_all[h, :C]                          # (C, d_model, d_model)
                M_s_h_T = M_s_all[h, :C].transpose(-1, -2)      # (C, d_model, d_model)

                # Rotate the residuals: (B, C, d_model) @ (C, d_model, d_model) -> (B, C, d_model)
                x_d_rot = einsum(x, M_d_h,   "b t m, t m n -> b t n")
                x_s_rot = einsum(x, M_s_h_T, "b t m, t m n -> b t n")

                # Project onto signal directions
                u_h = x_d_rot @ S_U_h.T  # (B, C, n_sig_h)
                v_h = x_s_rot @ S_V_h.T  # (B, C, n_sig_h)

                if use_bias_offset:
                    # Add RoPE-rotated bias terms (position-dependent)
                    c_d_h = c_d[h]   # (d_model,)
                    c_s_h = c_s[h]   # (d_model,)

                    c_d_rot = einsum(c_d_h, M_d_h,   "m, t m n -> t n")   # (C, d_model)
                    c_s_rot = einsum(c_s_h, M_s_h_T, "m, t m n -> t n")   # (C, d_model)

                    bias_u_h = c_d_rot @ S_U_h.T  # (C, n_sig_h)
                    bias_v_h = c_s_rot @ S_V_h.T  # (C, n_sig_h)

                    u_h = u_h + bias_u_h.unsqueeze(0)  # broadcast over batch
                    v_h = v_h + bias_v_h.unsqueeze(0)

                # Scatter back into the full (B, C, n_signals) tensors
                u_proj[:, :, sig_ids] = u_h
                v_proj[:, :, sig_ids] = v_h

        # --- B. Pairwise Scores ---
        scores_real = einsum(
            u_proj, v_proj,
            "n_sentences n_tokens_u n_signals, n_sentences n_tokens_v n_signals"
            " -> n_sentences n_tokens_u n_tokens_v n_signals",
        )
        scores_real = scores_real * causal_mask

        # --- C. Random Priorities ---
        # Generate a random priority for every valid pair to select unbiased samples.
        scores_rand = torch.rand(scores_real.shape, device=device, generator=rng) * causal_mask

        # --- D. Process Streams ---
        def process_stream(
            scores_tensor: Tensor,
            global_v: Tensor,
            global_i: Tensor,
        ) -> tuple[Tensor, Tensor]:
            # Permute to (n_signals, B, C, C) and flatten to (n_signals, -1)
            flat = scores_tensor.permute(3, 0, 1, 2).reshape(n_signals, -1)

            # Get Batch Top-K
            b_vals, b_flat_idx = flat.topk(k, dim=1)

            # Unravel flat indices back to (Batch, Dest_Token, Src_Token)
            b_idx, u_idx, v_idx = torch.unravel_index(b_flat_idx, (B, C, C))

            # Shift batch index to global sentence index
            sent_idx = b_idx + current_global_idx

            # Stack: (n_signals, k, 3) — [global_sentence_id, dest_token, src_token]
            b_meta = torch.stack([sent_idx, u_idx, v_idx], dim=2)

            return update_reservoir(global_v, global_i, b_vals, b_meta, k)

        topk_vals, topk_idx = process_stream(scores_real, topk_vals, topk_idx)
        rand_vals, rand_idx = process_stream(scores_rand, rand_vals, rand_idx)

        current_global_idx += B

    save_layer_activations(out_h5, layer_idx, topk_vals, topk_idx, rand_vals, rand_idx)
    log.info(f"Done. Saved: {out_h5}")
    return out_h5


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract top-K and random activations from Pile residuals."
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="TransformerLens model name (e.g. gpt2-small, EleutherAI/pythia-160m).",
    )
    parser.add_argument(
        "--task", "-t",
        required=True,
        help="Task label used in the output filename (e.g. ioi-balanced).",
    )
    parser.add_argument(
        "--layer", "-l",
        type=int,
        required=True,
        help="Layer index to process.",
    )
    parser.add_argument(
        "--signals_file",
        required=True,
        help="Path to the signals H5 file (from extract_signals.py).",
    )
    parser.add_argument(
        "--pile_activations",
        required=True,
        help="Path to the pile activations H5 file (from collect_pile_activations.py).",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory for the output H5 file (default: current directory).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda).",
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=8,
        help="DataLoader batch size (default: 8).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of DataLoader worker processes (default: 16).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=40,
        help="Number of top-K and random-K entries to keep per signal (default: 40).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the random reservoir (default: 42).",
    )
    parser.add_argument(
        "--use_numpy_svd",
        action="store_true",
        help="Use numpy.linalg.pinv for weight pseudoinverses (recommended for Pythia-160m).",
    )
    args = parser.parse_args()

    extract_top_activations(
        model_name=args.model,
        task=args.task,
        layer_idx=args.layer,
        signals_file=args.signals_file,
        pile_activations_file=args.pile_activations,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        k=args.k,
        seed=args.seed,
        use_numpy_svd=args.use_numpy_svd,
    )


if __name__ == "__main__":
    main()
