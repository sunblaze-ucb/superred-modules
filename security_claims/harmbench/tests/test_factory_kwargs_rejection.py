"""Tests for the hardcoded-kwarg rejection in leaf and composed factories.

Before this fix, calling e.g. ``harmbench_standard_test(functional_categories=...)``
would forward ``functional_categories`` to ``harmbench_claim()`` which already
received it from the leaf factory's hardcoded value, producing a confusing
``TypeError: got multiple values for keyword argument`` from the underlying
function. Now we raise a clear ``TypeError`` at the leaf with a helpful message.
"""

from __future__ import annotations

import pytest

from secclaim_harmbench import (
    harmbench_complete,
    harmbench_contextual_test,
    harmbench_contextual_val,
    harmbench_copyright_test,
    harmbench_copyright_val,
    harmbench_standard_test,
    harmbench_standard_val,
    harmbench_test,
    harmbench_val,
)


_JUDGE_KWARGS = {
    "judge_model": "openai/test",
    "judge_api_base": "https://x",
    "judge_api_key": "sk-test",
}


_ALL_LEAF_AND_COMPOSED = [
    harmbench_standard_test,
    harmbench_contextual_test,
    harmbench_copyright_test,
    harmbench_standard_val,
    harmbench_contextual_val,
    harmbench_copyright_val,
    harmbench_test,
    harmbench_val,
    harmbench_complete,
]


@pytest.mark.parametrize("factory", _ALL_LEAF_AND_COMPOSED)
def test_factory_rejects_split_kwarg(factory) -> None:
    """All leaf/composed factories hardcode ``split`` and must reject it."""
    with pytest.raises(TypeError, match="hardcoded"):
        factory(split="test", **_JUDGE_KWARGS)


@pytest.mark.parametrize("factory", _ALL_LEAF_AND_COMPOSED)
def test_factory_rejects_functional_categories_kwarg(factory) -> None:
    """All leaf/composed factories hardcode ``functional_categories``."""
    with pytest.raises(TypeError, match="hardcoded"):
        factory(functional_categories=("standard",), **_JUDGE_KWARGS)


def test_factory_error_message_names_factory() -> None:
    """The error mentions the specific factory the user called."""
    with pytest.raises(TypeError, match=r"harmbench_standard_test\(\)"):
        harmbench_standard_test(split="test", **_JUDGE_KWARGS)

    with pytest.raises(TypeError, match=r"harmbench_complete\(\)"):
        harmbench_complete(functional_categories=("standard",), **_JUDGE_KWARGS)


def test_factory_accepts_legitimate_kwargs() -> None:
    """Sanity: legitimate kwargs still pass through after the rejection check."""
    claim = harmbench_standard_val(
        semantic_categories=("harmful",),
        clip_tokens=256,
        **_JUDGE_KWARGS,
    )
    tasks = list(claim)
    # 4 'harmful' standard rows in val (verified against the CSV).
    assert len(tasks) == 4
    for t in tasks:
        assert t.semantic_category == "harmful"  # type: ignore[attr-defined]


def test_composed_factory_rejection_blocks_propagation() -> None:
    """Composed factory rejects BEFORE propagating to leaf factories.

    Without the rejection, the bad kwarg would be forwarded to each
    leaf factory and trigger the same error N times (or worse, partial
    work). The composed factory must catch it up front.
    """
    with pytest.raises(TypeError, match=r"harmbench_test\(\)"):
        harmbench_test(split="val", **_JUDGE_KWARGS)


def test_complete_rejects_csv_path(tmp_path) -> None:
    """``harmbench_complete(csv_path=X)`` would load X twice (once via
    the test composition, once via val) and yield every row twice.
    Reject the combination at the boundary."""
    custom_csv = tmp_path / "custom.csv"
    custom_csv.write_text(
        "Behavior,FunctionalCategory,SemanticCategory,Tags,ContextString,BehaviorID\n"
        "test,standard,harmful,,,c1\n",
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match=r"harmbench_complete\(\).*csv_path"):
        harmbench_complete(csv_path=str(custom_csv), **_JUDGE_KWARGS)
