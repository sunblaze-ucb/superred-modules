"""Guard: this package must never specify an LLM temperature.

Reasoning models reject the parameter outright (gpt-5.x answers "gpt-5 models
don't support temperature=0. Only temperature=1 is supported"); Bedrock Claude
rejects temperature combined with top_p. Every component that pinned one here
was a judge, evaluator or scorer that swallows its own exceptions, so the pin
did not fail loudly: it silently disabled the component and reported the attack
as a clean failure. In the RQ1.3-1 2026-06/07 run that cost 56,198 tasks.

The check is structural (AST), not a text search, so a rename, a reformat or a
value moved into a dict cannot slip past it.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def _offences() -> list[str]:
    found: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if "_vendor" in path.parts:
            continue  # vendored upstream code is pinned verbatim by design
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = path.relative_to(SRC)
        for node in ast.walk(tree):
            line = getattr(node, "lineno", 0)
            if isinstance(node, ast.arg) and "temperature" in node.arg.lower():
                found.append(f"{rel}:{line} parameter {node.arg!r}")
            elif (
                isinstance(node, ast.keyword)
                and node.arg
                and "temperature" in node.arg.lower()
            ):
                found.append(f"{rel}:{line} keyword argument {node.arg!r}")
            elif isinstance(node, ast.Dict):
                for key in node.keys:
                    if (
                        isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and "temperature" in key.value.lower()
                    ):
                        found.append(f"{rel}:{line} dict key {key.value!r}")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    name = getattr(target, "attr", None) or getattr(target, "id", None)
                    if name and "temperature" in name.lower():
                        found.append(f"{rel}:{line} assignment to {name!r}")
    return found


def test_package_never_specifies_a_temperature() -> None:
    offences = _offences()
    assert not offences, (
        "This package must not specify an LLM temperature. Found:\n  "
        + "\n  ".join(offences)
        + "\n\nOmit the parameter entirely so each provider uses its own default."
    )
