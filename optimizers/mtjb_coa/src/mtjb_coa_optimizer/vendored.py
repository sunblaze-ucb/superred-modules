"""Access to byte-identically-vendored MT-JailBench payloads.

The upstream prompt/rubric/payload bodies are never retyped into this
package. They live under ``_vendor/`` exactly as shipped by MT-JailBench at
the pinned commit (see ``scripts/sync_upstream.py``) and are read from there
at runtime:

* YAML prompt templates are loaded with :func:`load_yaml`.
* String constants embedded in upstream ``.py`` files (e.g. the CoA LLM
  similarity-scoring prompt) are extracted with :func:`load_source_constant`,
  which parses the vendored source with :mod:`ast` and evaluates only the
  target assignment's string literal -- the module is never imported, so its
  ``from client.unified_llm_client import ...`` line and its heavy transitive
  dependencies are never touched.

Keeping the bodies in ``_vendor/`` means the sync checker can byte-compare
them against upstream and the content rule is honoured (this file contains
loader logic only, no payload text).
"""

from __future__ import annotations

import ast
import textwrap
from functools import cache
from importlib.resources import files
from typing import Any

import yaml

_PKG = "mtjb_coa_optimizer"
_VENDOR_ROOT = ("_vendor", "mtjb", "engine")


def _read_vendored(*parts: str) -> str:
    """Return the text of a vendored file addressed relative to ``engine/``."""
    resource = files(_PKG)
    for part in (*_VENDOR_ROOT, *parts):
        resource = resource.joinpath(part)
    return resource.read_text(encoding="utf-8")


@cache
def load_yaml(*parts: str) -> dict[str, Any]:
    """Load a vendored YAML file (addressed relative to ``engine/``)."""
    return yaml.safe_load(_read_vendored(*parts))


def _eval_string_node(node: ast.expr) -> str:
    """Evaluate an expression node that must yield a ``str`` payload.

    Supports the two shapes upstream uses for prompt constants:

    * ``"..."`` / implicit-concatenated string literals -> a single
      :class:`ast.Constant`.
    * ``textwrap.dedent("...")`` -> a :class:`ast.Call` we unwrap and dedent.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "dedent"
        and len(node.args) == 1
    ):
        return textwrap.dedent(_eval_string_node(node.args[0]))
    raise ValueError(f"unsupported constant expression: {ast.dump(node)[:80]}")


@cache
def load_source_constant(name: str, *parts: str) -> str:
    """Extract a top-level or class-level ``str`` constant from vendored source.

    Args:
        name: The assigned identifier to extract (e.g. ``"SIM_APPROX_PROMPT"``).
        *parts: Path of the vendored ``.py`` file relative to ``engine/``.

    Returns:
        The evaluated string literal, byte-for-byte as upstream wrote it.

    Raises:
        KeyError: If no assignment to ``name`` is found.
    """
    tree = ast.parse(_read_vendored(*parts))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return _eval_string_node(node.value)
    raise KeyError(f"constant {name!r} not found in {'/'.join(parts)}")
