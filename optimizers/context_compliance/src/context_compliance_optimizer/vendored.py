"""Access to the byte-identically-vendored PyRIT context-compliance templates.

The upstream prompt bodies are never retyped into this package. They live under
``_vendor/`` exactly as shipped by microsoft/PyRIT at the pinned commit (see
``scripts/sync_upstream.py``) and are read from there at runtime. Each vendored
file keeps its full upstream repository path under ``_vendor/`` (for example
``_vendor/pyrit/datasets/executors/...``), so the sync checker can byte-compare
it against the pinned commit and the content rule is honoured: this file holds
loader/renderer logic only, no payload text.

The two templates are PyRIT ``SeedPrompt`` YAML documents. PyRIT renders their
``value`` with Jinja2; both context-compliance templates use only plain
``{{ variable }}`` substitution -- no control blocks, filters, or expressions --
so a stdlib substitution is byte-equivalent to Jinja2 for these files and keeps
the package free of a Jinja2 runtime dependency (see ASSUMPTIONS.md).
"""

from __future__ import annotations

import re
from functools import cache
from importlib.resources import files
from typing import Any

import yaml

_PKG = "context_compliance_optimizer"
_VENDOR_ROOT = "_vendor"

#: A single ``{{ name }}`` placeholder with optional surrounding whitespace.
_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def _read_vendored(*parts: str) -> str:
    """Return the text of a vendored file addressed by its upstream repo path."""
    resource = files(_PKG).joinpath(_VENDOR_ROOT)
    for part in parts:
        resource = resource.joinpath(part)
    return resource.read_text(encoding="utf-8")


@cache
def load_seed_prompt(*parts: str) -> dict[str, Any]:
    """Load a vendored PyRIT ``SeedPrompt`` YAML (addressed by its upstream path)."""
    parsed = yaml.safe_load(_read_vendored(*parts))
    if not isinstance(parsed, dict):
        raise ValueError(f"vendored SeedPrompt {'/'.join(parts)} did not parse to a mapping")
    return parsed


def render_seed_prompt(seed: dict[str, Any], /, **params: object) -> str:
    """Render a ``SeedPrompt``'s ``value`` by substituting ``{{ name }}`` placeholders.

    Only the parameters supplied are substituted; any placeholder without a
    matching parameter is left untouched. All values are stringified, mirroring
    Jinja2's default text rendering for these ``data_type: text`` templates.
    """
    template = seed.get("value")
    if not isinstance(template, str):
        raise TypeError("SeedPrompt 'value' must be a string")
    str_params = {key: str(value) for key, value in params.items()}

    def _sub(match: re.Match[str]) -> str:
        return str_params.get(match.group(1), match.group(0))

    return _PLACEHOLDER.sub(_sub, template)
