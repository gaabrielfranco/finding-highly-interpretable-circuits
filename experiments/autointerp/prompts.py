"""Shared prompt templates and parsers for the autointerp pipeline."""

import ast
import re
import random
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

EXPLAINER_SYSTEM_PROMPT = """You are a meticulous AI researcher conducting an important investigation into how hidden signals in language models correspond to patterns in text. Your task is to analyze the given text snippets and provide an interpretation that clearly summarizes the linguistic or semantic pattern they reveal.  

Guidelines:  
- You will be shown several short text examples.
- In each example, one or two important tokens will be surrounded by << >> and [[ ]] (e.g. "the [[ cat]] sat on the << mat>>").
- These brackets indicate where the model's internal signal was most active. Signals are found in the context of attention heads, meaning that the token highlighted with << >> acts as a destination token, while the token highlighted with [[ ]] acts as a source token. If only one token is highlighted with both << >> and [[ ]], it means that this token acts as both destination and source tokens.
- Each example may include additional context words around the highlighted token(s).
- Your goal is to produce a concise, high-level description of what these highlighted tokens have in common (their meaning, grammatical role, or thematic pattern possibility).
- Focus on the underlying pattern, not on quoting or restating the examples.
- Remember to look at the position of the tokens in the text chunk. This also matters. 
- Ignore the brackets and any tokenization artifacts when describing your interpretation.
- If the examples appear random or uninformative, say so briefly.
- Do not list multiple hypotheses; choose the single best interpretation.
- Keep your answer short and clear.
- If you are not more than 90% certain about an interpretation, say "no valid interpretation found".
- The final line of your response must provide your conclusion in this exact format:
[interpretation]: your concise description here
"""

JUDGE_SYSTEM_PROMPT = """You are an intelligent and meticulous linguistics researcher.

You will be given a specific linguistic feature of interest, such as  "male pronouns," "negative sentiment," or "surname tokens."

You will then be given several text examples that are claimed to contain this feature. Portions of the text that supposedly represent the feature have been marked using << >> and [[ ]].
These brackets indicate where the model's internal signal was most active.
Features are found in the context of attention heads, meaning that the token highlighted with << >> acts as a destination token, while the token highlighted with [[ ]] acts as a source token. If only one token is highlighted with both << >> and [[ ]], it means that this token acts as both destination and source tokens.

Your task is to determine whether EVERY token inside each << >> and [[ ]] span is correctly labeled as an instance of the feature.

Important:
- An example is correct ONLY if every marked tokens are representative of the feature.
- There are exactly 10 examples below.

For each example in turn:
- Return 1 if ALL marked spans correctly represent the feature.
- Return 0 if ANY marked token is mislabeled.

Important Output Format Rules:
1. Return the results as a Python Dictionary.
2. The keys must be the example numbers (1 to 10), and the values must be the binary label (0 or 1).
3. Do not assume the order; explicitly check the number I assigned to each example.
4. Ignore any numbers or formatting artifacts INSIDE the text strings (e.g., if a text contains "4).", ignore it).
5. Output format:
   {
    1: 0,
    2: 1,
    ...
    10: 0
   }

Here are the examples:

<user_prompt>
Feature interpretation: Words related to American football positions, specifically the tight end position.
Text examples:
1. Getty Images [[ Patriots]]<< tight>> end Rob Gronkowski had his boss
2. posted You should know this[[ about]] offensive line coaches: they are large, demanding<< men>>
3. Media Day 2015 LSU [[ defensive]] end Isaiah Washington (94) speaks to<< the>>
4. running [[ backs]],'' he said. .. Defensive << end>> Carroll Phillips is improving and his injury is
5. [[ line]], with the left side namely << tackle>> Byron Bell at tackle and guard Amini
<assistant_response>
{
1: 1,
2: 0,
3: 0,
4: 1,
5: 1,
}

Now evaluate the following examples:
"""


# ---------------------------------------------------------------------------
# Explainer prompt builders
# ---------------------------------------------------------------------------

def build_explainer_prompt(
    examples_indices: np.ndarray,
    tokenizer_gpt2: Any,
    dataset_the_pile: Any,
    tokenizer_explainer: Any,
) -> str:
    """Build a chat-template prompt for the explainer LLM (DeepSeek-R1).

    Wraps the text examples in << >> / [[ ]] brackets and applies the
    explainer tokenizer's chat template so vLLM can process it directly.

    Args:
        examples_indices: Array of shape (n_examples, 3) with columns
            [sentence_id, dest_token, src_token].
        tokenizer_gpt2: GPT-2 tokenizer used to decode Pile tokens.
        dataset_the_pile: HuggingFace ``NeelNanda/pile-10k`` dataset
            (tokenized, max_length=32).
        tokenizer_explainer: Tokenizer for the explainer LLM; used to
            apply the chat template.

    Returns:
        Chat-template formatted string ready for vLLM generation.
    """
    formatted_lines = []
    for i in range(len(examples_indices)):
        sentence_id, dest_token, src_token = examples_indices[i]
        tokens = [tokenizer_gpt2.decode(t) for t in dataset_the_pile[sentence_id]["tokens"]]

        tokens[src_token] = f"[[{tokens[src_token]}]]"
        tokens[dest_token] = f"<<{tokens[dest_token]}>>"

        line = "".join(tokens)

        formatted_lines.append(f"- {line!r}")

    user_content = "Text examples:\n\n" + "\n".join(formatted_lines)

    messages = [
        {"role": "system", "content": EXPLAINER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    return tokenizer_explainer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_explainer_prompt_gemini(
    examples_indices: np.ndarray,
    tokenizer_gpt2: Any,
    dataset_the_pile: Any,
) -> str:
    """Build a user-content prompt for the Gemini batch API.

    Returns only the user-content string (without system prompt), because
    the Gemini batch JSONL format passes system prompt separately via
    ``system_instruction``.

    Args:
        examples_indices: Array of shape (n_examples, 3) with columns
            [sentence_id, dest_token, src_token].
        tokenizer_gpt2: GPT-2 tokenizer used to decode Pile tokens.
        dataset_the_pile: HuggingFace ``NeelNanda/pile-10k`` dataset
            (tokenized, max_length=32).

    Returns:
        User-content string for the Gemini ``contents[0].parts[0].text``
        field.
    """
    formatted_lines = []
    for i in range(len(examples_indices)):
        sentence_id, dest_token, src_token = examples_indices[i]
        tokens = [tokenizer_gpt2.decode(t) for t in dataset_the_pile[sentence_id]["tokens"]]

        tokens[src_token] = f"[[{tokens[src_token]}]]"
        tokens[dest_token] = f"<<{tokens[dest_token]}>>"

        line = "".join(tokens)

        formatted_lines.append(f"- {line!r}")

    user_content = "Text examples:\n\n" + "\n".join(formatted_lines)

    return user_content


# ---------------------------------------------------------------------------
# Judge prompt builder
# ---------------------------------------------------------------------------

def build_judge_prompt(
    interpretation: str,
    examples_indices: np.ndarray,
    random_indices: np.ndarray,
    tokenizer_gpt2: Any,
    dataset_the_pile: Any,
    tokenizer_judge: Any,
    n_examples: int = 5,
) -> tuple[str, np.ndarray]:
    """Build a scoring prompt for the judge LLM (Gemma-3-27b).

    Constructs a mini-prompt with ``n_examples`` positive (top-K) examples
    and ``n_examples`` negative (random) examples, shuffled together.
    Returns both the prompt string and the ground-truth label array.

    Args:
        interpretation: The interpretation string generated by the explainer.
        examples_indices: Top-K activation indices, shape (>=n_examples, 3).
        random_indices: Random activation indices, shape (>=n_examples, 3).
        tokenizer_gpt2: GPT-2 tokenizer used to decode Pile tokens.
        dataset_the_pile: HuggingFace ``NeelNanda/pile-10k`` dataset
            (tokenized, max_length=32).
        tokenizer_judge: Tokenizer for the judge LLM; used to apply the
            chat template.
        n_examples: Number of positive and negative examples each (default
            5, giving 10 total per prompt).

    Returns:
        Tuple of (prompt_text, examples_labels) where ``examples_labels``
        is a numpy array of shape (2 * n_examples,) with 1 for positive
        examples and 0 for negative examples, in shuffled order.

    Raises:
        Exception: If n_examples exceeds the number of available indices.
    """
    formatted_lines = []

    if n_examples > len(examples_indices) or n_examples > len(random_indices):
        raise Exception(f"n_examples={examples_indices} is greater than the number of available examples")

    examples_labels = [1] * n_examples + [0] * n_examples
    for indices in [examples_indices, random_indices]:
        for i in range(n_examples):
            sentence_id, dest_token, src_token = indices[i]
            tokens = [tokenizer_gpt2.decode(t) for t in dataset_the_pile[sentence_id]["tokens"]]

            tokens[src_token] = f"[[{tokens[src_token]}]]"
            tokens[dest_token] = f"<<{tokens[dest_token]}>>"

            line = "".join(tokens)

            formatted_lines.append(f"- {line!r}")

    formatted_lines = np.array(formatted_lines)
    examples_labels = np.array(examples_labels)

    indices_shuffle = list(range(len(formatted_lines)))
    random.shuffle(indices_shuffle)
    formatted_lines = formatted_lines[indices_shuffle]
    examples_labels = examples_labels[indices_shuffle]

    for i in range(len(formatted_lines)):
        formatted_lines[i] = formatted_lines[i][1:]  # Removing the "-"
        formatted_lines[i] = f"{i+1}.{formatted_lines[i]}"

    user_content = f"Feature interpretation: {interpretation}\nText examples:\n\n" + "\n".join(formatted_lines)

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    return tokenizer_judge.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), examples_labels


# ---------------------------------------------------------------------------
# Interpretation parsers
# ---------------------------------------------------------------------------

def extract_interpretation(text: str) -> str | None:
    """Parse the ``[interpretation]:`` line from DeepSeek-R1 output.

    Applies strict validation: if the model did not finish thinking (no
    ``</think>`` tag), the response is treated as a failure and ``None``
    is returned. Otherwise extracts the interpretation with several
    fallback strategies.

    Args:
        text: Raw text output from the explainer LLM.

    Returns:
        The interpretation string, ``"no valid interpretation found"`` if
        the model explicitly said so, the full cleaned text as a last
        resort, or ``None`` if the model did not finish thinking.
    """
    # 1. STRICT CHECK: Did the model finish thinking?
    if '</think>' not in text:
        return None

    # 2. Get the answer part
    clean_text = text.split('</think>')[-1].strip()

    # 3. Check for specific "No valid interpretation" string
    if re.search(r'no\s+valid\s+interpretation\s+found', clean_text, re.IGNORECASE):
        return "no valid interpretation found"

    # 4. Strategy A: Exact Match [interpretation]:
    match_strict = re.search(r'\[interpretation\]:\s*(.*)', clean_text, re.IGNORECASE | re.DOTALL)
    if match_strict:
        return match_strict.group(1).strip()

    # 5. Strategy B: Hallucinated Label (Last line fallback)
    lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
    if lines:
        last_line = lines[-1]
        match_loose = re.match(r'^\[.*?\]:\s*(.*)', last_line)
        if match_loose:
            return match_loose.group(1).strip()

    # 6. Fallback: Return the cleaned text
    return clean_text


def extract_interpretation_gemini(text: str) -> tuple[str, bool]:
    """Parse the ``[interpretation]:`` line from Gemini output.

    Simpler than ``extract_interpretation`` — no ``</think>`` handling
    needed since Gemini does not use chain-of-thought tags.

    Args:
        text: Raw text output from the Gemini API.

    Returns:
        Tuple of (interpretation, extraction_success). If no
        ``[interpretation]:`` line is found and the text does not contain
        "no valid interpretation found", returns ``("", False)``.
    """
    match_strict = re.search(r'\[interpretation\]:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if match_strict:
        return match_strict.group(1).strip(), True

    if re.search(r'no\s+valid\s+interpretation\s+found', text, re.IGNORECASE):
        return "no valid interpretation found", True

    return "", False


# ---------------------------------------------------------------------------
# Judge label parser
# ---------------------------------------------------------------------------

def parse_judge_labels(text: str, n_examples: int) -> np.ndarray:
    """Parse the judge LLM's dict output into a binary label array.

    The judge returns a Python dict mapping example numbers (1-indexed) to
    binary labels (0 or 1). This function:

    1. Strips markdown code-block fences (``` python / ```) from ``text``.
    2. Parses the cleaned text with ``ast.literal_eval`` (safe alternative
       to ``eval``).
    3. Extracts labels for keys 1 through ``n_examples``.
    4. Returns ``np.full(n_examples, -1)`` on any parse failure or if the
       number of keys does not match ``n_examples``.

    Args:
        text: Raw text output from the judge LLM.
        n_examples: Expected total number of examples (positive +
            negative), i.e. keys 1 through ``n_examples`` in the dict.

    Returns:
        Integer numpy array of shape (n_examples,) with values in
        {0, 1, -1}. -1 indicates a parse failure or skipped entry.
    """
    text = text.replace("```python\n", "").replace("\n```", "")  # remove markdown notation
    try:
        data = ast.literal_eval(text)
        if len(data) != n_examples:
            return np.full(n_examples, -1)
        return np.array([data[i] for i in range(1, n_examples + 1)])
    except Exception:
        return np.full(n_examples, -1)
