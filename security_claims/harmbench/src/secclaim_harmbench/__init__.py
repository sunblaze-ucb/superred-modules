"""HarmBench standardized red-teaming benchmark as a superred SecurityClaim.

Public API:

    harmbench_claim(...)                -> parameterized factory
    harmbench_standard_test()           -> 159 standard behaviors, test split
    harmbench_contextual_test()         -> 81 contextual behaviors, test split
    harmbench_copyright_test()          -> 80 copyright behaviors, test split
    harmbench_standard_val()            -> 41 standard behaviors, val split
    harmbench_contextual_val()          -> 19 contextual behaviors, val split
    harmbench_copyright_val()           -> 20 copyright behaviors, val split
    harmbench_test()                    -> all 320 test behaviors (composed)
    harmbench_val()                     -> all 80 val behaviors (composed)
    harmbench_complete()                -> all 400 text behaviors (composed)

Composed factories use ``SecurityClaim.from_claims`` lazy chaining; the
controller iterates them by flattening transparently.

Example:

    >>> from secclaim_harmbench import harmbench_standard_val
    >>> claim = harmbench_standard_val(judge_model="openai/gpt-4o-mini")
    >>> tasks = list(claim)
    >>> len(tasks)
    41

The judge defaults to reading ``LITELLM_API_BASE`` and
``LITELLM_API_KEY`` from the environment if ``judge_api_base`` and
``judge_api_key`` are not provided to the factory.

Per-task metadata (``behavior_id``, ``functional_category``,
``semantic_category``, ``split``) is exposed on each :class:`HarmBenchTask`
instance so post-run aggregation by any axis is a simple groupby over
``result.task_results`` (where ``result`` is the
``ThreatModelResult`` returned by ``controller.run()``).
"""

from __future__ import annotations

from typing import Iterable, Literal, cast

from chatbot_target import ChatbotTarget  # type: ignore[import-untyped]

from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.task import Task

from secclaim_harmbench.advbench import REFUSAL_PREFIXES, refusal_label
from secclaim_harmbench.judge import (
    DEFAULT_CLIP_TOKENS,
    HarmBenchJudge,
    LLAMA2_CLS_PROMPT_CONTEXTUAL,
    LLAMA2_CLS_PROMPT_STANDARD,
    parse_judgment,
)
from secclaim_harmbench.loader import VALID_FUNCTIONAL, VALID_SPLITS, load_behaviors
from secclaim_harmbench.task import HarmBenchTask, render_test_case


def harmbench_claim(
    *,
    judge_model: str = "openai/gpt-4o-mini",
    judge_api_base: str | None = None,
    judge_api_key: str | None = None,
    split: Literal["test", "val"] = "test",
    functional_categories: Iterable[str] | None = None,
    semantic_categories: Iterable[str] | None = None,
    csv_path: str | None = None,
    clip_tokens: int = DEFAULT_CLIP_TOKENS,
) -> SecurityClaim[ChatbotTarget]:
    """Build a HarmBench SecurityClaim, optionally filtered.

    Args:
        judge_model: LiteLLM-compatible judge model identifier. Default
            ``openai/gpt-4o-mini``. Used for standard and contextual
            behaviors. Copyright behaviors use the bundled MinHash
            matcher and do NOT call the judge.
        judge_api_base: Provider API base URL for the judge. Defaults
            to env ``LITELLM_API_BASE`` if ``None``.
        judge_api_key: Provider API key for the judge. Defaults to env
            ``LITELLM_API_KEY`` if ``None``.
        split: Which bundled split to use. Ignored when ``csv_path`` is
            provided.
        functional_categories: Restrict to these functional categories
            (subset of ``{"standard", "contextual", "copyright"}``).
            ``None`` keeps all.
        semantic_categories: Restrict to these semantic categories.
            ``None`` keeps all.
        csv_path: Optional override path to a CSV with the upstream
            schema.
        clip_tokens: Token budget per generation for judge formatting.
            Default 512 (matches HarmBench's pipeline default).

    Returns:
        A :class:`SecurityClaim` containing one :class:`HarmBenchTask`
        per selected behavior.

    Raises:
        ValueError: If filters select zero rows or the split is invalid.
        FileNotFoundError: If ``csv_path`` doesn't exist.
    """
    rows = load_behaviors(
        split=split,
        functional_categories=functional_categories,
        semantic_categories=semantic_categories,
        csv_path=csv_path,
    )
    judge = HarmBenchJudge(
        model=judge_model,
        api_base=judge_api_base,
        api_key=judge_api_key,
        clip_tokens=clip_tokens,
    )
    tasks = [HarmBenchTask(row=r, judge=judge) for r in rows]
    return SecurityClaim.from_tasks(cast(list[Task[ChatbotTarget]], tasks))


# ---------------------------------------------------------------------
# Helper: reject kwargs that the leaf/composed factories hardcode, so a
# user mistake yields a clear error rather than a confusing
# "got multiple values for keyword argument" TypeError from harmbench_claim().
# ---------------------------------------------------------------------


def _reject_hardcoded_kwargs(
    kwargs: dict[str, object], *, factory_name: str, hardcoded: tuple[str, ...],
) -> None:
    overlapping = [k for k in hardcoded if k in kwargs]
    if overlapping:
        raise TypeError(
            f"{factory_name}() does not accept {overlapping!r} as kwargs "
            f"(they are hardcoded by this factory). Use harmbench_claim() "
            f"directly if you need to override them."
        )


# ---------------------------------------------------------------------
# Leaf factories: one (split, functional_category) combination each.
# Each leaf hardcodes ``split`` and ``functional_categories`` and forwards
# every other kwarg to harmbench_claim().
# ---------------------------------------------------------------------

_LEAF_HARDCODED = ("split", "functional_categories")


def harmbench_standard_test(**kwargs: object) -> SecurityClaim[ChatbotTarget]:
    """159 standard behaviors from the test split."""
    _reject_hardcoded_kwargs(
        kwargs, factory_name="harmbench_standard_test", hardcoded=_LEAF_HARDCODED,
    )
    return harmbench_claim(
        split="test", functional_categories=("standard",),
        **kwargs,  # type: ignore[arg-type]
    )


def harmbench_contextual_test(**kwargs: object) -> SecurityClaim[ChatbotTarget]:
    """81 contextual behaviors from the test split."""
    _reject_hardcoded_kwargs(
        kwargs, factory_name="harmbench_contextual_test", hardcoded=_LEAF_HARDCODED,
    )
    return harmbench_claim(
        split="test", functional_categories=("contextual",),
        **kwargs,  # type: ignore[arg-type]
    )


def harmbench_copyright_test(**kwargs: object) -> SecurityClaim[ChatbotTarget]:
    """80 copyright behaviors from the test split (uses MinHash matcher)."""
    _reject_hardcoded_kwargs(
        kwargs, factory_name="harmbench_copyright_test", hardcoded=_LEAF_HARDCODED,
    )
    return harmbench_claim(
        split="test", functional_categories=("copyright",),
        **kwargs,  # type: ignore[arg-type]
    )


def harmbench_standard_val(**kwargs: object) -> SecurityClaim[ChatbotTarget]:
    """41 standard behaviors from the val split."""
    _reject_hardcoded_kwargs(
        kwargs, factory_name="harmbench_standard_val", hardcoded=_LEAF_HARDCODED,
    )
    return harmbench_claim(
        split="val", functional_categories=("standard",),
        **kwargs,  # type: ignore[arg-type]
    )


def harmbench_contextual_val(**kwargs: object) -> SecurityClaim[ChatbotTarget]:
    """19 contextual behaviors from the val split."""
    _reject_hardcoded_kwargs(
        kwargs, factory_name="harmbench_contextual_val", hardcoded=_LEAF_HARDCODED,
    )
    return harmbench_claim(
        split="val", functional_categories=("contextual",),
        **kwargs,  # type: ignore[arg-type]
    )


def harmbench_copyright_val(**kwargs: object) -> SecurityClaim[ChatbotTarget]:
    """20 copyright behaviors from the val split (uses MinHash matcher)."""
    _reject_hardcoded_kwargs(
        kwargs, factory_name="harmbench_copyright_val", hardcoded=_LEAF_HARDCODED,
    )
    return harmbench_claim(
        split="val", functional_categories=("copyright",),
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------
# Composed factories. Same hardcoded-kwarg restriction applies because
# they delegate to the leaf factories that hardcode split + functional.
# ---------------------------------------------------------------------


def harmbench_test(**kwargs: object) -> SecurityClaim[ChatbotTarget]:
    """All 320 behaviors from the test split, composed across the three
    functional categories. Use the test split for reporting numbers."""
    _reject_hardcoded_kwargs(
        kwargs, factory_name="harmbench_test", hardcoded=_LEAF_HARDCODED,
    )
    return SecurityClaim.from_claims([
        harmbench_standard_test(**kwargs),
        harmbench_contextual_test(**kwargs),
        harmbench_copyright_test(**kwargs),
    ])


def harmbench_val(**kwargs: object) -> SecurityClaim[ChatbotTarget]:
    """All 80 behaviors from the val split, composed. Use the val split
    for tuning to avoid leaking the test set."""
    _reject_hardcoded_kwargs(
        kwargs, factory_name="harmbench_val", hardcoded=_LEAF_HARDCODED,
    )
    return SecurityClaim.from_claims([
        harmbench_standard_val(**kwargs),
        harmbench_contextual_val(**kwargs),
        harmbench_copyright_val(**kwargs),
    ])


def harmbench_complete(**kwargs: object) -> SecurityClaim[ChatbotTarget]:
    """All 400 text behaviors (test + val), composed.

    The HarmBench paper requires that "attacks and defenses do not tune
    on the test set". Using the complete claim conflates val and test;
    prefer ``harmbench_test()`` for reporting and ``harmbench_val()``
    for tuning.

    Rejects ``csv_path``: a custom CSV has no notion of test-vs-val
    split, so composing both built-in splits over the same file would
    return every row twice. Use ``harmbench_claim(csv_path=...)``
    directly if you want to load a custom CSV.
    """
    _reject_hardcoded_kwargs(
        kwargs, factory_name="harmbench_complete",
        hardcoded=(*_LEAF_HARDCODED, "csv_path"),
    )
    return SecurityClaim.from_claims([
        harmbench_test(**kwargs),
        harmbench_val(**kwargs),
    ])


__all__ = [
    # Public types
    "HarmBenchJudge",
    "HarmBenchTask",
    # Constants reusable for tests / validation
    "DEFAULT_CLIP_TOKENS",
    "LLAMA2_CLS_PROMPT_STANDARD",
    "LLAMA2_CLS_PROMPT_CONTEXTUAL",
    "REFUSAL_PREFIXES",
    "VALID_FUNCTIONAL",
    "VALID_SPLITS",
    # Helpers
    "load_behaviors",
    "parse_judgment",
    "refusal_label",
    "render_test_case",
    # Factories
    "harmbench_claim",
    "harmbench_standard_test",
    "harmbench_contextual_test",
    "harmbench_copyright_test",
    "harmbench_standard_val",
    "harmbench_contextual_val",
    "harmbench_copyright_val",
    "harmbench_test",
    "harmbench_val",
    "harmbench_complete",
]
