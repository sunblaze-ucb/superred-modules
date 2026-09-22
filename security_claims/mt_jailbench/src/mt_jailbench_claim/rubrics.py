"""Load MT-JailBench judge rubrics from the byte-identical vendored source.

The four rubric prompt templates and ``MAX_JUDGE_RETRY`` live as constants in
``_vendor/mtjb/engine/utils/independent_judge.py`` (vendored verbatim from the
pinned commit). They are extracted here with :mod:`ast` -- the module is never
imported, so its ``from client.unified_llm_client import UnifiedLLMClient`` line
and heavy transitive dependencies are never touched, and no rubric body is
reproduced in authored code (the content rule).

The claim's judge (``judge.py``) reimplements the parse/aggregation logic in
async form over ``litellm``, but the rubric *text* it sends is exactly what
upstream ships.
"""

from __future__ import annotations

import ast
import textwrap
from functools import cache
from importlib.resources import files

_PKG = "mt_jailbench_claim"
_JUDGE_SOURCE = ("_vendor", "mtjb", "engine", "utils", "independent_judge.py")


def _source() -> str:
    resource = files(_PKG)
    for part in _JUDGE_SOURCE:
        resource = resource.joinpath(part)
    return resource.read_text(encoding="utf-8")


def _eval_string_node(node: ast.expr) -> str:
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
def _constants() -> dict[str, object]:
    """Extract every ``NAME = <str|int>`` assignment (module- or class-level)."""
    tree = ast.parse(_source())
    out: dict[str, object] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, int):
                out[target.id] = value.value
            else:
                try:
                    out[target.id] = _eval_string_node(value)
                except ValueError:
                    continue
    return out


def rubric(name: str) -> str:
    """Return a vendored rubric template by constant name (e.g. ``COA``)."""
    key = f"{name}_PROMPT_TEMPLATE"
    value = _constants().get(key)
    if not isinstance(value, str):
        raise KeyError(f"rubric {key!r} not found in vendored independent_judge.py")
    return value


@cache
def max_judge_retry() -> int:
    """Upstream ``JailbreakJudge.MAX_JUDGE_RETRY`` (3 at the pinned commit)."""
    value = _constants().get("MAX_JUDGE_RETRY")
    return value if isinstance(value, int) else 3
