"""Packaging metadata checks for DRA optimizer dependencies."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_detoxify_is_a_default_dependency() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text())

    dependencies = pyproject["project"]["dependencies"]
    optional_dependencies = pyproject["project"].get("optional-dependencies", {})

    assert any(dependency.startswith("detoxify") for dependency in dependencies)
    assert "detoxify" not in optional_dependencies
