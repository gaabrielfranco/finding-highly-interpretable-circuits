"""Stage 0: Collect ln1.hook_normalized residuals from The Pile.

Streams through NeelNanda/pile-10k, runs each batch through the target
model, and saves the ln1.hook_normalized activations for all layers to a
single H5 file. These activations are the "corpus" that signals are
matched against in Stage 1 (extract_top_activations.py).

Output file: {output_dir}/pile_activations_{model_short}.h5
  Groups: L0, L1, ..., L{n_layers-1}
  Each group: shape (N_sentences, seq_len, d_model), float32, gzip-9

The output file can be very large:
  ~500 GB for GPT-2 small / Pythia-160m, ~4 TB for Gemma-2-2b.
These files are locally-generated intermediate data and are NOT hosted
on HuggingFace.

Usage (full run):
    python collect_pile_activations.py \\
        --model gpt2-small \\
        --device cuda \\
        --batch_size 32 \\
        --max_length 32 \\
        --output_dir data/autointerp/ \\
        --hf_cache_dir ~/.cache/huggingface

Usage (small sample for validation, first 5 batches = 160 sentences):
    python collect_pile_activations.py \\
        --model gpt2-small \\
        --device cpu \\
        --batch_size 32 \\
        --max_length 32 \\
        --max_batches 5 \\
        --output_dir /tmp/pile_sample/ \\
        --hf_cache_dir ~/.cache/huggingface
"""

import argparse
import os
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer, utils
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

torch.set_grad_enabled(False)


def collect_pile_activations(
    model_name: str,
    output_dir: str,
    device: str = "cuda",
    batch_size: int = 32,
    max_length: int = 32,
    max_batches: int | None = None,
    hf_cache_dir: str | None = None,
) -> str:
    """Collect ln1.hook_normalized activations from The Pile for all layers.

    Args:
        model_name: TransformerLens model name (e.g. ``"gpt2-small"``,
            ``"EleutherAI/pythia-160m"``, ``"gemma-2-2b"``).
        output_dir: Directory where the output H5 file is written.
        device: Torch device string.
        batch_size: Batch size for the DataLoader. Also used as the H5
            chunk size along the first axis.
        max_length: Sequence length for Pile tokenization (default 32).
        max_batches: If set, stop after processing this many batches.
            Useful for generating small validation samples. The output
            file will contain ``max_batches * batch_size`` rows.
        hf_cache_dir: Optional path to HuggingFace cache directory.
            If given, sets ``HF_HOME`` before loading the dataset.

    Returns:
        Path to the output H5 file.

    Raises:
        SystemExit: If the output file already exists (refuses to overwrite).
    """
    if hf_cache_dir is not None:
        os.environ["HF_HOME"] = hf_cache_dir

    model_short = model_name.split("/")[-1]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_h5 = os.path.join(output_dir, f"pile_activations_{model_short}.h5")

    if os.path.exists(out_h5):
        raise SystemExit(f"File already exists, refusing to overwrite: {out_h5}")

    log.info(f"Loading model: {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)

    log.info("Loading NeelNanda/pile-10k")
    the_pile_data = load_dataset("NeelNanda/pile-10k", split="train")
    dataset = utils.tokenize_and_concatenate(
        the_pile_data,
        model.tokenizer,
        max_length=max_length,
        add_bos_token=False,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    L = model.cfg.n_layers
    HOOKS = {f"blocks.{ell}.ln1.hook_normalized" for ell in range(L)}

    def names_filter(n: str) -> bool:
        return n in HOOKS

    model.eval()

    log.info(f"Output: {out_h5}")

    f = h5py.File(out_h5, "w")

    dsets: dict = {}         # layer -> h5 dataset
    write_ptr = [0] * L      # number of rows written per layer

    total_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
    for idx, data in enumerate(tqdm(dataloader, total=total_batches, desc="Collecting ln1.hook_normalized")):
        if max_batches is not None and idx >= max_batches:
            break
        toks = data["tokens"].to(model.cfg.device)

        _, cache = model.run_with_cache(
            toks,
            names_filter=names_filter,
            remove_batch_dim=False,
        )

        # Initialise datasets on the first batch
        if idx == 0:
            probe = cache["blocks.0.ln1.hook_normalized"]
            B, T, d_model = probe.shape

            f.attrs["model_name"] = model_name
            f.attrs["hook"]       = "ln1.hook_normalized"
            f.attrs["n_layers"]   = L
            f.attrs["seq_len"]    = T
            f.attrs["d_model"]    = d_model
            f.attrs["batch_size"] = B

            for ell in range(L):
                dsets[ell] = f.create_dataset(
                    f"L{ell}",
                    shape=(0, T, d_model),
                    maxshape=(None, T, d_model),
                    dtype="float32",
                    chunks=(B, T, d_model),
                    compression="gzip",
                    compression_opts=9,
                )

        # Append batch for every layer
        for ell in range(L):
            key   = f"blocks.{ell}.ln1.hook_normalized"
            X_btd = cache[key]
            X_np  = X_btd.detach().cpu().to(torch.float32).numpy()

            ds = dsets[ell]
            old = write_ptr[ell]
            ds.resize(old + X_np.shape[0], axis=0)
            ds[old:old + X_np.shape[0], :, :] = X_np
            write_ptr[ell] += X_np.shape[0]

    f.close()
    log.info(f"Done. Saved: {out_h5}")
    return out_h5


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect ln1.hook_normalized activations from The Pile."
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="TransformerLens model name (e.g. gpt2-small, gemma-2-2b).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda).",
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=32,
        help="DataLoader batch size (default: 32).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32,
        help="Sequence length for Pile tokenization (default: 32).",
    )
    parser.add_argument(
        "--output_dir",
        default="data/autointerp",
        help="Directory for output H5 file (default: data/autointerp).",
    )
    parser.add_argument(
        "--hf_cache_dir",
        default=None,
        help="Optional HuggingFace cache directory (sets HF_HOME).",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Stop after this many batches (for validation samples only).",
    )
    args = parser.parse_args()

    collect_pile_activations(
        model_name=args.model,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_batches=args.max_batches,
        hf_cache_dir=args.hf_cache_dir,
    )


if __name__ == "__main__":
    main()
