"""Response parsers for the CoA attacker/judge, ported from upstream.

These mirror the extraction logic in MT-JailBench's
``engine/attacks/coa/coa_prompt_generator.py`` and
``coa_outcome_evaluator.py`` (parse-only; no prompt text is reproduced).
All parsers return ``None`` on failure so the optimizer can degrade the
same way upstream does (retry / discard) rather than raising.
"""

from __future__ import annotations

import ast
import json
import re

_JUDGE_SCORE_RE = re.compile(r"\[\[(\d+)\]\]")


def _loads_dict(json_str: str) -> dict | None:
    """Parse a dict from a string, trying ``ast.literal_eval`` then ``json``."""
    for loader in (ast.literal_eval, json.loads):
        try:
            parsed = loader(json_str)
        except (SyntaxError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def clean_prompt(prompt: object) -> str:
    """Normalise a generated prompt string (upstream ``_clean_prompt``)."""
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""
    text = str(prompt).strip()
    if text.startswith("['") and text.endswith("']"):
        text = text[2:-2]
    elif text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return text.strip()


def extract_chain(response: str) -> list[str] | None:
    """Extract the ordered round prompts from an init-chain response.

    Mirrors ``CoAPromptGenerator._extract_init_chain``: the response is a
    JSON/py-dict object whose non-``evaluation`` items each carry an
    ``improvement`` and a ``prompt``. Returns the list of prompts in order,
    or ``None`` if nothing parseable is found.
    """
    if not response:
        return None
    text = response.strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    parsed = _loads_dict(text[start:end])
    if parsed is None:
        return None

    prompts: list[str] = []
    for key, item in parsed.items():
        if key == "evaluation":
            continue
        if not isinstance(item, dict) or "prompt" not in item:
            return None
        prompt = clean_prompt(item["prompt"])
        if not prompt:
            return None
        prompts.append(prompt)
    return prompts or None


def extract_update_prompt(response: str) -> str | None:
    """Extract the refined prompt from an update response.

    Mirrors ``CoAPromptGenerator._extract_json`` + ``_clean_prompt``: the
    response must be an object with ``improvement`` and ``prompt`` keys.
    """
    if not response:
        return None
    start, end = response.find("{"), response.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    parsed = _loads_dict(response[start:end].replace("\n", ""))
    if parsed is None or not {"improvement", "prompt"} <= set(parsed):
        return None
    cleaned = clean_prompt(parsed["prompt"])
    return cleaned or None


def extract_judge_score(output: str) -> int | None:
    """Extract the binary CoA judge score (``[[0]]`` / ``[[1]]``).

    Mirrors ``CoAOutcomeEvaluator._extract_judge_score``: any integer in the
    ``[[n]]`` marker is clamped to at most 1.
    """
    if not output:
        return None
    match = _JUDGE_SCORE_RE.search(output)
    if match is None:
        return None
    return min(int(match.group(1)), 1)


def extract_similarity(output: str) -> float | None:
    """Extract a ``{"similarity": <float>}`` value, clamped to ``[0, 1]``."""
    if not output:
        return None
    start, end = output.find("{"), output.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    parsed = _loads_dict(output[start:end])
    if parsed is None or "similarity" not in parsed:
        return None
    try:
        value = float(parsed["similarity"])
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, value))
