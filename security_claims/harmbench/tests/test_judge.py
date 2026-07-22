"""Judge tests: verbatim prompt, parser robustness, dispatch logic.

We do not call any real LLM. Network-dependent ``HarmBenchJudge.judge``
behavior is exercised in :mod:`test_task_evaluate` via mocking.
"""

from __future__ import annotations

import pytest

from harmbench_claim.judge import (
    DEFAULT_CLIP_TOKENS,
    HarmBenchJudge,
    LLAMA2_CLS_PROMPT_CONTEXTUAL,
    LLAMA2_CLS_PROMPT_STANDARD,
    parse_judgment,
)


# ---------- Verbatim prompt-template fingerprints ----------
# These are static checks: any change to the prompt template (intentional
# or accidental) must update these expected substrings. They function as
# guardrails against drift from upstream eval_utils.py.


def test_standard_prompt_starts_with_inst() -> None:
    assert LLAMA2_CLS_PROMPT_STANDARD.startswith("[INST] <<SYS>>")


def test_standard_prompt_ends_with_inst() -> None:
    assert LLAMA2_CLS_PROMPT_STANDARD.endswith("Answer: [/INST]")


def test_standard_prompt_uses_bracketed_field_labels() -> None:
    # Verbatim from eval_utils.py uses [BEHAVIOR]: and [GENERATION]:
    # (the notebook uses no brackets; eval_utils.py is canonical).
    assert "[BEHAVIOR]:" in LLAMA2_CLS_PROMPT_STANDARD
    assert "[GENERATION]:" in LLAMA2_CLS_PROMPT_STANDARD


def test_standard_prompt_has_placeholders() -> None:
    assert "{behavior}" in LLAMA2_CLS_PROMPT_STANDARD
    assert "{generation}" in LLAMA2_CLS_PROMPT_STANDARD
    # Critically: no {context} in the standard template.
    assert "{context}" not in LLAMA2_CLS_PROMPT_STANDARD


def test_contextual_prompt_has_three_placeholders() -> None:
    assert "{behavior}" in LLAMA2_CLS_PROMPT_CONTEXTUAL
    assert "{generation}" in LLAMA2_CLS_PROMPT_CONTEXTUAL
    assert "{context}" in LLAMA2_CLS_PROMPT_CONTEXTUAL


def test_contextual_prompt_includes_context_field() -> None:
    assert "[CONTEXT]:" in LLAMA2_CLS_PROMPT_CONTEXTUAL


def test_contextual_prompt_has_english_only_rule() -> None:
    """The contextual prompt has the 'must be in English' rule that the
    standard prompt also carries (upstream eval_utils.py)."""
    assert "Generations must be in English" in LLAMA2_CLS_PROMPT_CONTEXTUAL
    assert "Generations must be in English" in LLAMA2_CLS_PROMPT_STANDARD


# ---------- parse_judgment ----------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Yes", True),
        ("yes", True),
        ("YES", True),
        ("No", False),
        ("no", False),
        ("NO", False),
        ("  Yes  \n", True),
        ("Yes.", True),  # punctuation tolerated
        ("Yes,", True),
        ("**Yes**", True),  # markdown bold
        ('"Yes"', True),
    ],
)
def test_parse_judgment_accepts_canonical_outputs(raw: str, expected: bool) -> None:
    assert parse_judgment(raw) is expected


def test_parse_judgment_takes_first_token() -> None:
    """Some models prepend reasoning before the verdict. We take the first
    whitespace-delimited token to match the spirit of upstream's max_tokens=1."""
    assert parse_judgment("Yes the behavior was elicited") is True
    assert parse_judgment("No, the model refused") is False


def test_parse_judgment_empty_raises() -> None:
    with pytest.raises(RuntimeError):
        parse_judgment("")


def test_parse_judgment_whitespace_only_raises() -> None:
    with pytest.raises(RuntimeError):
        parse_judgment("   \n\n   ")


def test_parse_judgment_unparseable_raises() -> None:
    with pytest.raises(RuntimeError, match="unparseable judge output"):
        parse_judgment("Maybe.")


def test_parse_judgment_nonbinary_word_raises() -> None:
    with pytest.raises(RuntimeError):
        parse_judgment("Definitely!")


# ---------- HarmBenchJudge construction & dispatch ----------


def _judge(**kwargs: object) -> HarmBenchJudge:
    """Construct a judge with stub credentials."""
    return HarmBenchJudge(
        model="openai/test",
        api_base="https://x",
        api_key="sk-test",
        **kwargs,  # type: ignore[arg-type]
    )


def test_judge_construction_no_network() -> None:
    """Construction must not require network; no LLM call happens here."""
    judge = _judge()
    assert judge.model == "openai/test"


def test_judge_default_clip_tokens() -> None:
    judge = _judge()
    assert judge.clip_tokens == DEFAULT_CLIP_TOKENS == 512


def test_judge_clip_tokens_validation() -> None:
    with pytest.raises(ValueError, match="clip_tokens must be >= 1"):
        _judge(clip_tokens=0)
    with pytest.raises(ValueError, match="clip_tokens must be >= 1"):
        _judge(clip_tokens=-5)


def test_judge_env_fallback_for_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """If api_base/api_key are None, env vars are used."""
    monkeypatch.setenv("LITELLM_API_BASE", "https://env-base/")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-env-key")
    judge = HarmBenchJudge(model="openai/test")
    # internal access for verification
    assert judge._api_base == "https://env-base/"  # type: ignore[attr-defined]
    assert judge._api_key == "sk-env-key"  # type: ignore[attr-defined]


def test_judge_format_dispatches_on_context() -> None:
    """Standard prompt for None context, contextual prompt otherwise."""
    judge = _judge()
    standard_text = judge._format(  # type: ignore[attr-defined]
        behavior="b", generation="g", context=None,
    )
    contextual_text = judge._format(  # type: ignore[attr-defined]
        behavior="b", generation="g", context="c",
    )
    assert "[CONTEXT]:" not in standard_text
    assert "[CONTEXT]:" in contextual_text
    # Both must include the (non-templated) [BEHAVIOR]: and [GENERATION]: labels.
    for text in (standard_text, contextual_text):
        assert "[BEHAVIOR]:" in text
        assert "[GENERATION]:" in text


def test_judge_format_substitutes_correctly() -> None:
    judge = _judge()
    formatted = judge._format(  # type: ignore[attr-defined]
        behavior="STUB_BEHAVIOR", generation="STUB_GEN", context=None,
    )
    assert "STUB_BEHAVIOR" in formatted
    assert "STUB_GEN" in formatted


def test_judge_format_clips_long_generations() -> None:
    judge = _judge(clip_tokens=8)
    long_gen = "word " * 1000  # ~1000 tokens
    formatted = judge._format(  # type: ignore[attr-defined]
        behavior="b", generation=long_gen, context=None,
    )
    # Whole prompt under ~5000 chars: clipped generation should be short.
    assert len(formatted) < 5000
