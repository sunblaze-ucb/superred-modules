"""Factory tests: subset sizes, filtering, composition, validation."""

from __future__ import annotations

import pytest

from xstest_claim import (
    GPTJudgeConfig,
    XSTestTask,
    xstest_claim,
    xstest_full_claim,
    xstest_safe_claim,
    xstest_unsafe_claim,
)


def test_axis_and_full_sizes() -> None:
    assert len(list(xstest_safe_claim())) == 250
    assert len(list(xstest_unsafe_claim())) == 200
    assert len(list(xstest_full_claim())) == 450  # composed from the two axes


def test_full_claim_covers_both_labels() -> None:
    labels = {t.prompt_label for t in xstest_full_claim() if isinstance(t, XSTestTask)}
    assert labels == {"safe", "unsafe"}


def test_type_filter() -> None:
    claim = xstest_claim(types=["homonyms", "contrast_homonyms"])
    tasks = list(claim)
    assert len(tasks) == 50  # 25 each
    assert {t.dimension for t in tasks if isinstance(t, XSTestTask)} == {
        "homonyms",
        "contrast_homonyms",
    }


def test_label_filter() -> None:
    assert len(list(xstest_claim(labels=["safe"]))) == 250


def test_judge_config_threads_through() -> None:
    cfg = GPTJudgeConfig(api_base="x", api_key="y")
    claim = xstest_claim(labels=["safe"], types=["homonyms"], judge=cfg)
    tasks = list(claim)
    assert len(tasks) == 25
    assert all(t._judge is cfg for t in tasks if isinstance(t, XSTestTask))  # type: ignore[attr-defined]


def test_unknown_filters_raise() -> None:
    with pytest.raises(ValueError):
        xstest_claim(labels=["bogus"])
    with pytest.raises(ValueError):
        xstest_claim(types=["not_a_type"])
