"""Stage 5a: Build Gemini batch JSONL for representative circuit prompts.

Five subcommands:
  build-request   Filter signal/activation H5s to representative prompt IDs,
                  build a Gemini batch JSONL file, and save a
                  ``{output_stem}_edge_examples.json`` side output with the
                  formatted text strings for each signal.
  submit-batch    Upload a JSONL request file to the Gemini batch API and
                  submit the job. Prints the job ID for later use.
                  Requires ``GEMINI_API_KEY`` environment variable.
  check-batch     Check the status of a submitted Gemini batch job.
                  Requires ``GEMINI_API_KEY`` environment variable.
  download-batch  Download the response JSONL from a completed Gemini batch
                  job. Requires ``GEMINI_API_KEY`` environment variable.
  parse-response  Parse the Gemini response JSONL and extract interpretations
                  to a JSON file.

Typical workflow:
    # Step 1: Build JSONL request + edge examples side output
    python interpret_representatives.py build-request \\
        --model gpt2-small \\
        --task ioi-balanced \\
        --representatives 129 1613 1372 \\
        --signals_file data/clustering/gpt2-small/signals_balanced_gpt2-small_not-norm.h5 \\
        --activations_dir data/autointerp/ \\
        --output prompts_representatives.jsonl \\
        --hf_cache_dir ~/.cache/huggingface
    # Outputs: prompts_representatives.jsonl
    #        + prompts_representatives_edge_examples.json

    # Step 2: Submit to Gemini batch API
    export GEMINI_API_KEY=your_key_here
    python interpret_representatives.py submit-batch \\
        --request_file prompts_representatives.jsonl
    # Prints: Job ID: batches/abc123...

    # Step 3: Check status (repeat until SUCCEEDED)
    python interpret_representatives.py check-batch \\
        --job_id batches/abc123...

    # Step 4: Download response
    python interpret_representatives.py download-batch \\
        --job_id batches/abc123... \\
        --output_file prompts_representatives_response.jsonl

    # Step 5: Parse interpretations
    python interpret_representatives.py parse-response \\
        --response_file prompts_representatives_response.jsonl \\
        --output_file interpretations_representatives.json
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from datasets import load_dataset
from transformer_lens import HookedTransformer, utils

from prompts import EXPLAINER_SYSTEM_PROMPT, build_explainer_prompt_gemini, extract_interpretation_gemini

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_pile_examples(
    examples_indices: np.ndarray,
    tokenizer_gpt2: Any,
    dataset_the_pile: Any,
) -> list[str]:
    """Format top-K activation indices as text strings for the side output.

    Produces the same ``"- {line!r}"`` strings used internally by
    ``build_explainer_prompt_gemini``. Stored in
    ``{output_stem}_edge_examples.json`` for the Bokeh viewer.

    Args:
        examples_indices: Array of shape (n_examples, 3) with columns
            [sentence_id, dest_token, src_token].
        tokenizer_gpt2: GPT-2 tokenizer used to decode Pile tokens.
        dataset_the_pile: HuggingFace ``NeelNanda/pile-10k`` dataset
            (tokenized, max_length=32).

    Returns:
        List of formatted text strings, one per example.
    """
    formatted_lines: list[str] = []
    for i in range(len(examples_indices)):
        sentence_id, dest_token, src_token = examples_indices[i]
        tokens = [tokenizer_gpt2.decode(t) for t in dataset_the_pile[sentence_id]["tokens"]]

        tokens[src_token] = f"[[{tokens[src_token]}]]"
        tokens[dest_token] = f"<<{tokens[dest_token]}>>"

        line = "".join(tokens)
        formatted_lines.append(f"- {line!r}")
    return formatted_lines


# ---------------------------------------------------------------------------
# build-request
# ---------------------------------------------------------------------------

def build_request(
    model_name: str,
    task: str,
    representatives: list[int],
    signals_file: str,
    activations_dir: str,
    output: str,
    hf_cache_dir: str | None = None,
) -> None:
    """Build a Gemini batch JSONL file for representative circuit prompts.

    Filters the signal H5 to signals whose ``prompt_id`` is in
    ``representatives``, loads the corresponding top-K activation indices
    from the activation H5s, builds one Gemini batch request per signal,
    and writes the results to ``output`` (JSONL) and a side output
    ``{output_stem}_edge_examples.json`` (formatted text strings keyed by
    the same signal key as the JSONL).

    JSONL key format: ``"layer_{L}_{global_signal_idx}"`` — uses the global
    index in the layer's signals H5 so the key can be used to look up the
    signal in the H5 directly.

    Args:
        model_name: TransformerLens model name (e.g. ``"gpt2-small"``).
            Used only to load the tokenizer for Pile token decoding.
        task: Task identifier (e.g. ``"ioi-balanced"``). Used to locate
            activation H5 files via the pattern
            ``activations_{layer}_{task}_{model_short}.h5``.
        representatives: List of representative prompt IDs (integers).
            Only signals whose ``prompt_id`` is in this list are included.
        signals_file: Path to the signal H5 file produced by
            ``extract_signals.py`` (Part 2, Step 5).
        activations_dir: Directory containing Stage 1 activation H5 files
            (``activations_{layer}_{task}_{model_short}.h5``).
        output: Output path for the Gemini batch JSONL file.
        hf_cache_dir: Optional HuggingFace cache directory. Sets
            ``HF_HOME`` and is passed to ``load_dataset``.
    """
    if hf_cache_dir is not None:
        os.environ["HF_HOME"] = hf_cache_dir

    log.info(f"Loading tokenizer for: {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device="cpu")
    tokenizer_gpt2 = model.tokenizer
    del model

    log.info("Loading NeelNanda/pile-10k")
    the_pile_data = load_dataset("NeelNanda/pile-10k", split="train")
    dataset_the_pile = utils.tokenize_and_concatenate(
        the_pile_data, tokenizer_gpt2, max_length=32, add_bos_token=False
    )

    model_short = model_name.split("/")[-1]
    representatives_set = set(representatives)

    # Enumerate all layer groups in the signals H5
    with h5py.File(signals_file, "r") as f:
        layer_names = sorted(
            [k for k in f.keys() if k.startswith("layer_")],
            key=lambda s: int(s.split("_")[1]),
        )
    log.info(f"Found {len(layer_names)} layers in {signals_file}")

    jsonl_records: list[dict] = []
    edge_examples: dict[str, list[str]] = {}

    for layer_name in layer_names:
        layer_idx = int(layer_name.split("_")[1])
        activations_file = str(
            Path(activations_dir) / f"activations_{layer_idx}_{task}_{model_short}.h5"
        )

        if not Path(activations_file).exists():
            log.warning(
                f"Activations file not found: {activations_file} — skipping layer {layer_idx}"
            )
            continue

        # Load prompt_ids from signal metadata (column 0)
        with h5py.File(signals_file, "r") as f:
            metadata = f[layer_name]["metadata"][:]  # (n_signals, 7), int32

        prompt_ids = metadata[:, 0].astype(int)
        signal_ids = np.where(np.isin(prompt_ids, list(representatives_set)))[0]

        if len(signal_ids) == 0:
            log.debug(f"Layer {layer_idx}: no representative signals found")
            continue

        log.info(f"Layer {layer_idx}: {len(signal_ids)} representative signal(s)")

        # Load topk_indices for those signals from the activation H5
        with h5py.File(activations_file, "r") as f:
            if layer_name not in f:
                log.warning(
                    f"{layer_name} not found in {activations_file} — skipping"
                )
                continue
            grp = f[layer_name]
            if "topk_indices" not in grp:
                log.warning(
                    f"topk_indices not found in {activations_file}:{layer_name} — skipping"
                )
                continue

            total_signals = grp["topk_indices"].shape[0]
            valid_mask = signal_ids < total_signals
            if not valid_mask.all():
                n_invalid = int((~valid_mask).sum())
                log.warning(
                    f"Layer {layer_idx}: {n_invalid} signal_id(s) out of range, skipping"
                )
                signal_ids = signal_ids[valid_mask]

            for signal_id in signal_ids:
                topk_idx = grp["topk_indices"][int(signal_id)]  # (k, 3)

                key = f"{layer_name}_{signal_id}"

                # Build Gemini user-content prompt
                user_content = build_explainer_prompt_gemini(
                    topk_idx, tokenizer_gpt2, dataset_the_pile
                )

                # Build formatted text strings for side output
                text_examples = _format_pile_examples(
                    topk_idx, tokenizer_gpt2, dataset_the_pile
                )
                edge_examples[key] = text_examples

                # Build JSONL record
                record: dict = {
                    "key": key,
                    "request": {
                        "contents": [{"parts": [{"text": user_content}]}],
                        "system_instruction": {"parts": [{"text": EXPLAINER_SYSTEM_PROMPT}]},
                        "generation_config": {"temperature": 0.3, "seed": 42},
                    },
                }
                jsonl_records.append(record)

    # Write JSONL
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in jsonl_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(jsonl_records)} request(s) to {output_path}")

    # Write side output: edge examples JSON
    examples_path = output_path.parent / f"{output_path.stem}_edge_examples.json"
    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(edge_examples, f, ensure_ascii=False, indent=2)
    log.info(f"Wrote edge examples to {examples_path}")


# ---------------------------------------------------------------------------
# parse-response
# ---------------------------------------------------------------------------

def parse_response(
    response_file: str,
    output_file: str,
) -> None:
    """Parse a Gemini batch API response JSONL and extract interpretations.

    Reads each line of the response JSONL, extracts the model's text output
    via ``response["candidates"][0]["content"]["parts"][0]["text"]``, and
    applies ``extract_interpretation_gemini`` to parse the
    ``[interpretation]:`` line. Failures are handled gracefully — if
    response parsing fails, the signal gets ``"interpretation failed"``.

    Args:
        response_file: Path to the Gemini batch API response JSONL file.
            Each line is a JSON object with a ``"key"`` field and a
            ``"response"`` field containing the Gemini API response.
        output_file: Output path for the parsed interpretations JSON file.
            Format: ``{key: {"interpretation": ..., "full_response": ...}}``.
    """
    results: dict[str, dict[str, str]] = {}

    with open(response_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"Line {line_num}: JSON parse error: {e}")
                continue

            key = data.get("key", f"unknown_{line_num}")
            text = ""
            interpretation = "interpretation failed"

            try:
                text = (
                    data["response"]["candidates"][0]["content"]["parts"][0]["text"]
                )
                interpretation_parsed, success = extract_interpretation_gemini(text)
                if success:
                    interpretation = interpretation_parsed
                else:
                    log.warning(
                        f"Key {key}: [interpretation]: line not found, "
                        f"storing empty interpretation"
                    )
                    interpretation = ""
            except (KeyError, IndexError, TypeError) as e:
                log.warning(
                    f"Key {key}: response parse error ({e}), "
                    f"storing 'interpretation failed'"
                )

            results[key] = {
                "interpretation": interpretation,
                "full_response": text,
            }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info(f"Parsed {len(results)} response(s) → {output_path}")


# ---------------------------------------------------------------------------
# submit-batch
# ---------------------------------------------------------------------------

_DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"


def _get_gemini_client():
    """Create a Gemini API client from the ``GEMINI_API_KEY`` env variable.

    Raises:
        RuntimeError: If ``GEMINI_API_KEY`` is not set.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it with: export GEMINI_API_KEY=your_key_here"
        )
    from google import genai
    return genai.Client(api_key=api_key)


def submit_batch(
    request_file: str,
    gemini_model: str = _DEFAULT_GEMINI_MODEL,
    display_name: str = "accpp-batch-job",
) -> None:
    """Upload a JSONL request file and submit a Gemini batch job.

    Prints the job ID and status so the user can use ``check-batch``
    and ``download-batch`` later.

    Args:
        request_file: Path to the Gemini batch JSONL request file
            (output of ``build-request``).
        gemini_model: Gemini model ID for the batch job
            (default: ``gemini-3-flash-preview``).
        display_name: Display name for the batch job in the Gemini console.
    """
    from google.genai import types

    client = _get_gemini_client()

    log.info(f"Uploading {request_file}...")
    batch_file = client.files.upload(
        file=request_file,
        config=types.UploadFileConfig(
            display_name=display_name,
            mime_type="jsonl",
        ),
    )
    log.info(f"Uploaded file: {batch_file.name}")

    log.info(f"Submitting batch job (model={gemini_model})...")
    job = client.batches.create(
        model=gemini_model,
        src=batch_file.name,
        config={"display_name": display_name},
    )

    print(f"Job ID: {job.name}")
    print(f"Status: {job.state}")


# ---------------------------------------------------------------------------
# check-batch
# ---------------------------------------------------------------------------

def check_batch(job_id: str) -> None:
    """Check the status of a Gemini batch job.

    Prints the job state.

    Args:
        job_id: Gemini batch job ID (e.g. ``"batches/abc123..."``).
    """
    client = _get_gemini_client()
    batch_job = client.batches.get(name=job_id)
    print(f"Job ID: {job_id}")
    print(f"Status: {batch_job.state}")


# ---------------------------------------------------------------------------
# download-batch
# ---------------------------------------------------------------------------

def download_batch(
    job_id: str,
    output_file: str,
) -> None:
    """Download the response JSONL from a completed Gemini batch job.

    Checks that the job has a destination file, downloads it, and writes
    it to ``output_file``.

    Args:
        job_id: Gemini batch job ID (e.g. ``"batches/abc123..."``).
        output_file: Path to write the downloaded response JSONL.
    """
    client = _get_gemini_client()
    batch_job = client.batches.get(name=job_id)

    print(f"Job ID: {job_id}")
    print(f"Status: {batch_job.state}")

    if not batch_job.dest or not batch_job.dest.file_name:
        log.error(
            "Job has no destination file. "
            "It may still be running — check status with check-batch."
        )
        return

    result_file_name = batch_job.dest.file_name
    log.info(f"Downloading result file: {result_file_name}")
    file_content = client.files.download(file=result_file_name)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(file_content)
    log.info(f"Saved response to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 5a: Build Gemini batch JSONL for representative circuit prompts, "
            "submit to Gemini batch API, and parse the response."
        )
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # -----------------------------------------------------------------
    # build-request subcommand
    # -----------------------------------------------------------------
    build_parser = subparsers.add_parser(
        "build-request",
        help=(
            "Build Gemini batch JSONL + edge_examples.json from signal/activation H5s."
        ),
    )
    build_parser.add_argument(
        "--model", "-m",
        required=True,
        help="TransformerLens model name (e.g. gpt2-small). Used for tokenizer only.",
    )
    build_parser.add_argument(
        "--task",
        required=True,
        help="Task identifier (e.g. ioi-balanced). Used to locate activation H5 files.",
    )
    build_parser.add_argument(
        "--representatives",
        type=int,
        nargs="+",
        required=True,
        help="Representative prompt IDs (space-separated integers).",
    )
    build_parser.add_argument(
        "--signals_file",
        required=True,
        help=(
            "Path to the signal H5 file "
            "(e.g. data/clustering/gpt2-small/signals_balanced_gpt2-small_not-norm.h5)."
        ),
    )
    build_parser.add_argument(
        "--activations_dir",
        default="data/autointerp/",
        help="Directory containing Stage 1 activation H5 files (default: data/autointerp/).",
    )
    build_parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output path for the Gemini batch JSONL file. "
            "Side output {stem}_edge_examples.json is written alongside it."
        ),
    )
    build_parser.add_argument(
        "--hf_cache_dir",
        default=None,
        help="Optional HuggingFace cache directory (sets HF_HOME).",
    )

    # -----------------------------------------------------------------
    # parse-response subcommand
    # -----------------------------------------------------------------
    parse_parser = subparsers.add_parser(
        "parse-response",
        help="Parse Gemini batch API response JSONL and extract interpretations.",
    )
    parse_parser.add_argument(
        "--response_file",
        required=True,
        help="Path to the Gemini batch API response JSONL file.",
    )
    parse_parser.add_argument(
        "--output_file",
        required=True,
        help=(
            "Output path for parsed interpretations JSON file. "
            "Format: {key: {interpretation: ..., full_response: ...}}."
        ),
    )

    # -----------------------------------------------------------------
    # submit-batch subcommand
    # -----------------------------------------------------------------
    submit_parser = subparsers.add_parser(
        "submit-batch",
        help=(
            "Upload JSONL to Gemini batch API and submit the job. "
            "Requires GEMINI_API_KEY env variable."
        ),
    )
    submit_parser.add_argument(
        "--request_file",
        required=True,
        help="Path to the Gemini batch JSONL request file (output of build-request).",
    )
    submit_parser.add_argument(
        "--gemini_model",
        default=_DEFAULT_GEMINI_MODEL,
        help=f"Gemini model ID (default: {_DEFAULT_GEMINI_MODEL}).",
    )
    submit_parser.add_argument(
        "--display_name",
        default="accpp-batch-job",
        help="Display name for the batch job in the Gemini console (default: accpp-batch-job).",
    )

    # -----------------------------------------------------------------
    # check-batch subcommand
    # -----------------------------------------------------------------
    check_parser = subparsers.add_parser(
        "check-batch",
        help=(
            "Check status of a Gemini batch job. "
            "Requires GEMINI_API_KEY env variable."
        ),
    )
    check_parser.add_argument(
        "--job_id",
        required=True,
        help='Gemini batch job ID (e.g. "batches/abc123...").',
    )

    # -----------------------------------------------------------------
    # download-batch subcommand
    # -----------------------------------------------------------------
    download_parser = subparsers.add_parser(
        "download-batch",
        help=(
            "Download response JSONL from a completed Gemini batch job. "
            "Requires GEMINI_API_KEY env variable."
        ),
    )
    download_parser.add_argument(
        "--job_id",
        required=True,
        help='Gemini batch job ID (e.g. "batches/abc123...").',
    )
    download_parser.add_argument(
        "--output_file",
        required=True,
        help="Path to write the downloaded response JSONL.",
    )

    args = parser.parse_args()

    if args.subcommand == "build-request":
        build_request(
            model_name=args.model,
            task=args.task,
            representatives=args.representatives,
            signals_file=args.signals_file,
            activations_dir=args.activations_dir,
            output=args.output,
            hf_cache_dir=args.hf_cache_dir,
        )
    elif args.subcommand == "parse-response":
        parse_response(
            response_file=args.response_file,
            output_file=args.output_file,
        )
    elif args.subcommand == "submit-batch":
        submit_batch(
            request_file=args.request_file,
            gemini_model=args.gemini_model,
            display_name=args.display_name,
        )
    elif args.subcommand == "check-batch":
        check_batch(job_id=args.job_id)
    elif args.subcommand == "download-batch":
        download_batch(
            job_id=args.job_id,
            output_file=args.output_file,
        )


if __name__ == "__main__":
    main()
