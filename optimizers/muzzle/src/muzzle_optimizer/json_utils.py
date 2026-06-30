"""MUZZLE upstream utilities (behaviorally faithful to muzzle/prototype/agents/utils.py).

``load_system_prompt`` and ``extract_json_object`` are ported verbatim in behavior from
upstream gsiros/muzzle SHA ed611c0. The only change is mypy-strict-clean type annotations
(``extract_json_object`` may legitimately return ``None``, which the upstream annotation
``Dict[str, Any]`` understated). See ASSUMPTIONS.md.

``vendored_prompt_path`` resolves one of the five byte-identical vendored prompt YAMLs
shipped under ``muzzle_optimizer/data/prompts/``.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

import yaml


def vendored_prompt_path(name: str) -> str:
    """Filesystem path to a vendored prompt YAML, e.g. ``vendored_prompt_path("grafter")``."""
    resource = resources.files("muzzle_optimizer.data.prompts").joinpath(f"{name}.yaml")
    with resources.as_file(resource) as path:
        return str(path)


def load_system_prompt(prompt_path: str) -> str:
    """Load a system prompt string from a YAML file's ``system`` key.

    Always returns a non-empty string; falls back to a safe default when missing.
    Behavioral port of upstream ``agents/utils.py:load_system_prompt``.
    """
    try:
        with open(prompt_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        sp = data.get("system") if isinstance(data, dict) else None
        if isinstance(sp, str) and sp.strip():
            return sp
        return "Empty System Prompt"
    except Exception:
        return "Empty System Prompt"


def load_system_prompt_by_name(name: str) -> str:
    """Load a vendored prompt YAML by stem (e.g. ``"summarizer"``), de-escaping braces.

    The upstream ``summarizer``/``prompter`` YAML schema examples are written with
    ``.format()``-style escapes (``{{``/``}}``), but the upstream code never calls
    ``.format()`` (it cannot: the same prompts also contain illustrative ``{target_url}``
    placeholders that would raise ``KeyError``), so it sends the raw string and the model
    copies the ``{{`` back, producing invalid JSON that the parser drops, on a complex
    real-agent transcript the Summarizer then exhausts its retries and the playbook falls
    back to empty, silently disabling MUZZLE's trajectory grounding. We apply the brace
    collapse ``.format()`` was authored for (``{{`` -> ``{``, ``}}`` -> ``}``), which only
    touches doubled braces and leaves single ``{placeholder}`` tokens intact. The vendored
    YAML files stay byte-identical to upstream (the asset tests still pass); only the
    model-facing string is de-escaped. See ASSUMPTIONS.md deviation 9.
    """
    return load_system_prompt(vendored_prompt_path(name)).replace("{{", "{").replace("}}", "}")


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from raw model text.

    Behavioral port of upstream ``agents/utils.py:extract_json_object`` (strips a
    ``</think>`` prefix, then tries a direct parse, then the first ``{...}`` block).
    Return type widened to ``| None`` for mypy strict; behavior is unchanged.
    """
    # Ditch thinking <think></think> tags:
    raw_text = raw_text.split("</think>")[-1]

    # The model is instructed to return pure JSON, but be robust.
    raw_text = (raw_text or "").strip()

    # Fast path: direct JSON
    try:
        obj = json.loads(raw_text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Heuristic: find first {...} block
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw_text[start : end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


__all__ = [
    "vendored_prompt_path",
    "load_system_prompt",
    "load_system_prompt_by_name",
    "extract_json_object",
]
