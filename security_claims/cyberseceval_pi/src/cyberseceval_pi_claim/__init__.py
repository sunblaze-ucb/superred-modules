"""CyberSecEval prompt-injection benchmark as a superred SecurityClaim.

The textual prompt-injection test cases from Meta's PurpleLlama CyberSecEval,
judged by upstream's Yes/No judge prompt. Each case plants a defended system
prompt and sends an injection; success is the injection overriding it. Bound to
ChatbotTarget.

    >>> from cyberseceval_pi_claim import cyberseceval_pi_claim
    >>> claim = cyberseceval_pi_claim(judge_model="openai/gpt-4o-mini", limit=50)
"""

from __future__ import annotations

from typing import cast

from chatbot_target import ChatbotTarget  # type: ignore[import-untyped]
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.task import Task

from cyberseceval_pi_claim.judge import PromptInjectionJudge
from cyberseceval_pi_claim.loader import DATASET_NAMES, InjectionCase, load_cases
from cyberseceval_pi_claim.task import PromptInjectionTask

__all__ = [
    "DATASET_NAMES",
    "InjectionCase",
    "PromptInjectionJudge",
    "PromptInjectionTask",
    "cyberseceval_pi_claim",
    "load_cases",
]


def cyberseceval_pi_claim(
    *,
    judge_model: str = "openai/gpt-4o-mini",
    judge_api_base: str | None = None,
    judge_api_key: str | None = None,
    judge_max_attempts: int = 3,
    dataset: str = "english",
    injection_types: tuple[str, ...] | list[str] | None = None,
    risk_categories: tuple[str, ...] | list[str] | None = None,
    limit: int | None = None,
) -> SecurityClaim[ChatbotTarget]:
    """Build a CyberSecEval prompt-injection SecurityClaim.

    Args:
        judge_model: LiteLLM judge model id. The prompt and Yes/No rule are
            CyberSecEval's; the hosting model is your choice.
        judge_api_base/judge_api_key: judge provider (else env LITELLM_*).
        judge_max_attempts: judge calls per evaluation.
        dataset: ``"english"`` (251) or ``"multilingual"`` (1004).
        injection_types/risk_categories/limit: restrict the case set.

    Returns:
        A SecurityClaim with one PromptInjectionTask per selected case.
    """
    judge = PromptInjectionJudge(
        model=judge_model,
        api_base=judge_api_base,
        api_key=judge_api_key,
        max_attempts=judge_max_attempts,
    )
    cases = load_cases(
        dataset=dataset,
        injection_types=injection_types,
        risk_categories=risk_categories,
        limit=limit,
    )
    tasks = [PromptInjectionTask(case=c, judge=judge) for c in cases]
    return SecurityClaim.from_tasks(cast("list[Task[ChatbotTarget]]", tasks))
