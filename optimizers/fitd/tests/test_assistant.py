from __future__ import annotations

import pytest

from fitd_optimizer.assistant import parse_transformation_steps


def test_parse_transformation_steps_official_format() -> None:
    text = "**Step 1**: First bridge\n**Step 2**: Second bridge"
    assert parse_transformation_steps(text, expected_steps=2) == ["First bridge", "Second bridge"]


def test_parse_transformation_steps_rejects_missing_step() -> None:
    with pytest.raises(ValueError):
        parse_transformation_steps("**Step 1**: First bridge", expected_steps=2)


def test_interpolate_fallback_preserves_endpoints() -> None:
    from fitd_optimizer.assistant import _interpolate_prompt_sequence

    seq = _interpolate_prompt_sequence("benign", "target", 2)
    assert seq[0] == "benign"
    assert seq[-1] == "target"
    assert len(seq) == 4
