"""JailbreakBench (JBB-Behaviors) as a superred SecurityClaim.

The 100-behaviour harmful split, judged by JailbreakBench's canonical Llama-3
jailbreak classifier prompt. Single-turn, bound to ChatbotTarget.

    >>> from jailbreakbench_claim import jailbreakbench_claim
    >>> claim = jailbreakbench_claim(judge_model="openai/gpt-4o-mini")
    >>> tasks = list(claim)

The judge defaults to reading ``LITELLM_API_BASE`` / ``LITELLM_API_KEY`` from
the environment if not passed to the factory.
"""

from __future__ import annotations

from typing import cast

from chatbot_target import ChatbotTarget  # type: ignore[import-untyped]
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.task import Task

from jailbreakbench_claim.judge import JailbreakBenchJudge
from jailbreakbench_claim.loader import all_categories, load_behaviors
from jailbreakbench_claim.task import JailbreakBenchTask

__all__ = [
    "JailbreakBenchJudge",
    "JailbreakBenchTask",
    "all_categories",
    "jailbreakbench_claim",
    "load_behaviors",
]


def jailbreakbench_claim(
    *,
    dataset: str = "harmful",
    judge_model: str = "openai/gpt-4o-mini",
    judge_api_base: str | None = None,
    judge_api_key: str | None = None,
    judge_max_attempts: int = 3,
    categories: tuple[str, ...] | list[str] | None = None,
    sources: tuple[str, ...] | list[str] | None = None,
    limit: int | None = None,
) -> SecurityClaim[ChatbotTarget]:
    """Build a JailbreakBench SecurityClaim, optionally filtered.

    Args:
        judge_model: LiteLLM-compatible judge model id. The prompt and decision
            rule are JailbreakBench's; the hosting model is your choice.
        judge_api_base: Judge provider base URL (else env ``LITELLM_API_BASE``).
        judge_api_key: Judge provider key (else env ``LITELLM_API_KEY``).
        judge_max_attempts: Judge calls per evaluation, including the first.
        categories: Restrict to these JBB categories.
        sources: Restrict to these JBB sources (e.g. ``"Original"``).
        limit: Cap the number of behaviours.

    Returns:
        A SecurityClaim with one JailbreakBenchTask per selected behaviour.
    """
    judge = JailbreakBenchJudge(
        model=judge_model,
        api_base=judge_api_base,
        api_key=judge_api_key,
        max_attempts=judge_max_attempts,
    )
    rows = load_behaviors(categories=categories, sources=sources, limit=limit)
    tasks = [JailbreakBenchTask(row=r, judge=judge) for r in rows]
    return SecurityClaim.from_tasks(cast("list[Task[ChatbotTarget]]", tasks))
