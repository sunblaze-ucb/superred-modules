"""Guard that Libertas never imposes a temperature on its helper LLM."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def test_package_never_specifies_a_temperature() -> None:
    offences: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            line = getattr(node, "lineno", 0)
            if isinstance(node, ast.arg) and "temperature" in node.arg.casefold():
                offences.append(f"{path.name}:{line} parameter {node.arg!r}")
            elif (
                isinstance(node, ast.keyword)
                and node.arg is not None
                and "temperature" in node.arg.casefold()
            ):
                offences.append(f"{path.name}:{line} keyword {node.arg!r}")
            elif isinstance(node, ast.Dict):
                for key in node.keys:
                    if (
                        isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and "temperature" in key.value.casefold()
                    ):
                        offences.append(f"{path.name}:{line} dict key {key.value!r}")
    assert offences == []
