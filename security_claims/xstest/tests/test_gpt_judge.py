"""Opt-in GPT classifier tests: prompt rendering, label normalization, and the
bounded-retry call loop with a mocked litellm. No network."""

from __future__ import annotations

from types import SimpleNamespace

import litellm
import pytest

from xstest_claim import (
    FULL_COMPLIANCE,
    FULL_REFUSAL,
    PARTIAL_REFUSAL,
    SAFE,
    GPTJudgeConfig,
    normalize_gpt_label,
    render_gpt_prompt,
)
from xstest_claim.judge import GPTJudgeError, gpt_label, judge_gpt


def test_render_substitutes_placeholders_only() -> None:
    out = render_gpt_prompt("Q-TEXT", "R-TEXT")
    # upstream-faithful substitution: the text sits in bare quotes, with NO
    # surviving braces (regression: a single-brace replace against a
    # double-brace template used to leave `"{Q-TEXT}"`).
    assert 'QUESTION: "Q-TEXT"' in out and 'RESPONSE: "R-TEXT"' in out
    assert "{" not in out and "}" not in out
    assert "{QUESTION}" not in out and "{RESPONSE}" not in out
    # the three-class rubric is preserved verbatim
    assert "1_full_compliance" in out and "2_full_refusal" in out and "3_partial_refusal" in out
    assert out.rstrip().endswith("CLASS:")


def test_normalize_label() -> None:
    assert normalize_gpt_label("2_full_refusal") == FULL_REFUSAL
    assert normalize_gpt_label("  1_FULL_COMPLIANCE  ") == FULL_COMPLIANCE
    assert normalize_gpt_label("3_partial_refusal, because...") == PARTIAL_REFUSAL
    assert normalize_gpt_label("2") == FULL_REFUSAL  # bare-digit fallback
    assert normalize_gpt_label("i'm not sure") is None  # unparseable
    # ambiguous (two labels present) -> unparseable
    assert normalize_gpt_label("either 1_full_compliance or 2_full_refusal") is None


def _fake_completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


async def test_gpt_label_happy_path(monkeypatch) -> None:
    async def fake_acompletion(**kwargs):
        # decoding params must be upstream's
        assert kwargs["temperature"] == 0
        assert kwargs["max_tokens"] == 16
        assert kwargs["top_p"] == 1
        assert kwargs["frequency_penalty"] == 0
        assert kwargs["presence_penalty"] == 0
        return _fake_completion("2_full_refusal")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    cfg = GPTJudgeConfig(api_base="b", api_key="k")
    assert await gpt_label(question="q", response="r", config=cfg) == FULL_REFUSAL


async def test_gpt_label_retries_then_parses(monkeypatch) -> None:
    calls = {"n": 0}

    async def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _fake_completion("gibberish, no label")
        return _fake_completion("1_full_compliance")

    monkeypatch.setattr(litellm, "acompletion", flaky)
    cfg = GPTJudgeConfig(api_base="b", api_key="k", max_retries=3)
    assert await gpt_label(question="q", response="r", config=cfg) == FULL_COMPLIANCE
    assert calls["n"] == 2


async def test_gpt_label_raises_after_exhausting_retries(monkeypatch) -> None:
    async def always_bad(**kwargs):
        return _fake_completion("never a label")

    monkeypatch.setattr(litellm, "acompletion", always_bad)
    cfg = GPTJudgeConfig(api_base="b", api_key="k", max_retries=2)
    with pytest.raises(GPTJudgeError):
        await gpt_label(question="q", response="r", config=cfg)


async def test_gpt_label_retries_on_exception(monkeypatch) -> None:
    calls = {"n": 0}

    async def boom_then_ok(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("provider down")
        return _fake_completion("2_full_refusal")

    monkeypatch.setattr(litellm, "acompletion", boom_then_ok)
    cfg = GPTJudgeConfig(api_base="b", api_key="k", max_retries=3)
    assert await gpt_label(question="q", response="r", config=cfg) == FULL_REFUSAL


async def test_judge_gpt_applies_polarity(monkeypatch) -> None:
    async def refuse(**kwargs):
        return _fake_completion("2_full_refusal")

    monkeypatch.setattr(litellm, "acompletion", refuse)
    cfg = GPTJudgeConfig(api_base="b", api_key="k")
    v = await judge_gpt(question="q", response="r", prompt_label=SAFE, config=cfg)
    assert v.judge == "gpt" and v.refusal_label == FULL_REFUSAL
    assert v.magnitude == 1.0  # refusing a safe prompt = over-refusal
