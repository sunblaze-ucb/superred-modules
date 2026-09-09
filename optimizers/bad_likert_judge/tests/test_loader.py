"""Tests for the vendored-template loader's shadowing guard."""

from __future__ import annotations

import importlib.util
import sys

from bad_likert_judge_optimizer._vendor import loader


def test_template_loads_with_all_upstream_methods() -> None:
    t = loader.bad_likert_judge_template()
    assert t.__name__ == "BadLikertJudgeTemplate"
    for method in (
        "get_category_guidelines",
        "likert_setup_prompt",
        "likert_generate_examples",
        "likert_generate_attack_from_example",
        "likert_refine_score_3",
        "likert_judge",
        "extract_score_3_example",
        "non_refusal",
    ):
        assert hasattr(t, method)


def test_guard_checks_install_state_not_import_state(monkeypatch) -> None:
    """An installed-but-unimported deepteam must not be stubbed over.

    `_BASE_MODULE not in sys.modules` is true for a package that is installed
    but simply not imported yet, so the guard has to consult find_spec.
    """
    for name in list(sys.modules):
        if name == "deepteam" or name.startswith("deepteam."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    real = importlib.util.find_spec

    def pretend_installed(name, *a, **k):
        if name == loader._BASE_MODULE:
            return object()  # a truthy spec: "deepteam is installed"
        return real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", pretend_installed)
    assert loader._base_is_resolvable() is True


def test_absent_deepteam_is_reported_unresolvable(monkeypatch) -> None:
    for name in list(sys.modules):
        if name == "deepteam" or name.startswith("deepteam."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: None
    )
    assert loader._base_is_resolvable() is False


def test_already_imported_base_is_resolvable(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, loader._BASE_MODULE, object())
    assert loader._base_is_resolvable() is True
