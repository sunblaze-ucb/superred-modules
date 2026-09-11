"""Factories for the XSTest claim.

``xstest_claim(...)`` builds a :class:`SecurityClaim` of one
:class:`~xstest_claim.task.XSTestTask` per prompt, optionally filtered by label
or type. Three named wrappers cover the common cuts:

* :func:`xstest_safe_claim` — the 250 ``safe`` prompts (**over-refusal**, the
  distinctive XSTest axis: refusing a safe prompt is the attack);
* :func:`xstest_unsafe_claim` — the 200 ``unsafe`` ``contrast_*`` prompts
  (**unsafe compliance**: complying is the attack);
* :func:`xstest_full_claim` — all 450 prompts.

All default to the deterministic ``strmatch`` judge; pass a
:class:`~xstest_claim.judge.GPTJudgeConfig` to use the opt-in GPT classifier.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from chatbot_target import ChatbotTarget
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.task import Task

from xstest_claim.categories import PROMPT_LABELS, PROMPT_TYPES, SAFE, UNSAFE
from xstest_claim.judge import GPTJudgeConfig
from xstest_claim.loader import load_rows
from xstest_claim.task import XSTestTask

_STRMATCH: Literal["strmatch"] = "strmatch"


def xstest_claim(
    *,
    labels: Iterable[str] | None = None,
    types: Iterable[str] | None = None,
    judge: Literal["strmatch"] | GPTJudgeConfig = _STRMATCH,
    system_prompt: str | None = None,
    success_threshold: float = 1.0,
) -> SecurityClaim[ChatbotTarget]:
    """Build an XSTest claim over a (optionally filtered) subset of the 450 prompts.

    Args:
        labels: restrict to these prompt labels (subset of ``{"safe", "unsafe"}``);
            default both.
        types: restrict to these prompt types (subset of the 18 XSTest types);
            default all.
        judge: ``"strmatch"`` (default, deterministic) or a :class:`GPTJudgeConfig`.
        system_prompt: override the target system prompt, or ``None`` to leave it.
        success_threshold: attack magnitude at/above which a run is a success.

    Raises:
        ValueError: if ``labels`` or ``types`` contains an unknown value.
    """
    rows = load_rows()
    if labels is not None:
        wanted_labels = tuple(labels)
        unknown = [x for x in wanted_labels if x not in PROMPT_LABELS]
        if unknown:
            raise ValueError(f"unknown labels {unknown}; expected subset of {PROMPT_LABELS}")
        labelset = set(wanted_labels)
        rows = [r for r in rows if r["label"] in labelset]
    if types is not None:
        wanted_types = tuple(types)
        unknown = [x for x in wanted_types if x not in PROMPT_TYPES]
        if unknown:
            raise ValueError(f"unknown types {unknown}; expected subset of {PROMPT_TYPES}")
        typeset = set(wanted_types)
        rows = [r for r in rows if r["type"] in typeset]

    tasks: list[Task[ChatbotTarget]] = [
        XSTestTask(
            prompt=r["prompt"],
            prompt_id=r["id"],
            prompt_type=r["type"],
            prompt_label=r["label"],
            focus=r["focus"],
            note=r["note"],
            judge=judge,
            system_prompt=system_prompt,
            success_threshold=success_threshold,
        )
        for r in rows
    ]
    return SecurityClaim.from_tasks(tasks)


def xstest_safe_claim(
    *,
    judge: Literal["strmatch"] | GPTJudgeConfig = _STRMATCH,
    system_prompt: str | None = None,
    success_threshold: float = 1.0,
) -> SecurityClaim[ChatbotTarget]:
    """The 250 ``safe`` prompts — the over-refusal (exaggerated safety) axis."""
    return xstest_claim(
        labels=[SAFE],
        judge=judge,
        system_prompt=system_prompt,
        success_threshold=success_threshold,
    )


def xstest_unsafe_claim(
    *,
    judge: Literal["strmatch"] | GPTJudgeConfig = _STRMATCH,
    system_prompt: str | None = None,
    success_threshold: float = 1.0,
) -> SecurityClaim[ChatbotTarget]:
    """The 200 ``unsafe`` ``contrast_*`` prompts — the unsafe-compliance axis."""
    return xstest_claim(
        labels=[UNSAFE],
        judge=judge,
        system_prompt=system_prompt,
        success_threshold=success_threshold,
    )


def xstest_full_claim(
    *,
    judge: Literal["strmatch"] | GPTJudgeConfig = _STRMATCH,
    system_prompt: str | None = None,
    success_threshold: float = 1.0,
) -> SecurityClaim[ChatbotTarget]:
    """All 450 XSTest prompts, composed from the two axes.

    Built via :meth:`SecurityClaim.from_claims` over the safe (over-refusal) and
    unsafe (unsafe-compliance) sub-claims, so the two axes stay legible and each
    can also be run alone (:func:`xstest_safe_claim` / :func:`xstest_unsafe_claim`).
    """
    return SecurityClaim.from_claims(
        [
            xstest_safe_claim(
                judge=judge,
                system_prompt=system_prompt,
                success_threshold=success_threshold,
            ),
            xstest_unsafe_claim(
                judge=judge,
                system_prompt=system_prompt,
                success_threshold=success_threshold,
            ),
        ]
    )


__all__ = [
    "xstest_claim",
    "xstest_safe_claim",
    "xstest_unsafe_claim",
    "xstest_full_claim",
]
