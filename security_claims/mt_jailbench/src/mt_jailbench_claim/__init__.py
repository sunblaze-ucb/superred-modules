"""MT-JailBench evaluation as a superred SecurityClaim.

Public surface: :func:`mt_jailbench_claim`, a factory returning a
``SecurityClaim[ChatbotTarget]`` with one :class:`MTJailBenchTask` per HarmBench
standard behavior (159 rows). Success is MT-JailBench's **unanimous** 3-judge
vote (XTeaming + Chain-of-Attack + Foot-in-the-Door; threshold 3-of-3).

The behaviors are the HarmBench standard subset, redistributed by MT-JailBench
under the MIT License; HarmBench is attributed separately in ``LICENSES/``.
superred's existing ``harmbench`` claim evaluates the *same* behaviors with the
HarmBench Llama-2 classifier; this claim differs only in the judge.

Citation: "MT-JailBench: A benchmark framework for studying multi-turn jailbreak
attacks", SafetyArena, arXiv:2605.11002 (pinned commit ``cb8184e0``).
"""

from __future__ import annotations

import os

from chatbot_target import ChatbotTarget  # type: ignore[import-untyped]
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.task import Task

from mt_jailbench_claim.judge import (
    DEFAULT_JUDGE_TYPES,
    DEFAULT_THRESHOLD,
    MODEL,
    JudgeConfig,
    JudgeKind,
    JudgeResult,
)
from mt_jailbench_claim.judge_failure import (
    JudgeFailedError,
    JudgeUnavailableError,
    JudgeUnparseableError,
)
from mt_jailbench_claim.loader import load_standard_behaviors
from mt_jailbench_claim.task import MTJailBenchTask


def mt_jailbench_claim(
    *,
    judge_api_base: str | None = None,
    judge_api_key: str | None = None,
    judge_model: str = MODEL,
    judge_max_retries: int = 0,
    number_of_behaviors: int | None = None,
    judge_types: tuple[JudgeKind, ...] = DEFAULT_JUDGE_TYPES,
    threshold: int = DEFAULT_THRESHOLD,
) -> SecurityClaim[ChatbotTarget]:
    """Build the MT-JailBench claim over the HarmBench standard behaviors.

    Args:
        judge_api_base: LiteLLM API base. Falls back to ``LITELLM_API_BASE``.
        judge_api_key: LiteLLM API key. Falls back to ``LITELLM_API_KEY``.
        judge_model: Judge model id (default: the paper's ``openai/gpt-4o``).
        judge_max_retries: Attempts per judge (0 -> upstream ``MAX_JUDGE_RETRY``).
        number_of_behaviors: Keep only the first N standard behaviors; ``None``
            keeps all 159 (upstream default).
        judge_types: Which judges to run (default: unanimous XTeaming/CoA/FITD).
        threshold: Success threshold (default 3 = 3-of-3 unanimous).

    Returns:
        A ``SecurityClaim[ChatbotTarget]`` with one task per behavior.
    """
    def _resolve(explicit: str | None, env_key: str) -> str:
        return explicit if explicit is not None else os.environ.get(env_key, "")

    config = JudgeConfig(
        api_base=_resolve(judge_api_base, "LITELLM_API_BASE"),
        api_key=_resolve(judge_api_key, "LITELLM_API_KEY"),
        model=judge_model,
        max_retries=judge_max_retries,
    )
    rows = load_standard_behaviors(number_of_behaviors=number_of_behaviors)
    tasks: list[Task[ChatbotTarget]] = [
        MTJailBenchTask(
            row=row,
            judge_config=config,
            judge_types=judge_types,
            threshold=threshold,
        )
        for row in rows
    ]
    return SecurityClaim.from_tasks(tasks)


__all__ = [
    "DEFAULT_JUDGE_TYPES",
    "DEFAULT_THRESHOLD",
    "JudgeConfig",
    "JudgeFailedError",
    "JudgeKind",
    "JudgeResult",
    "JudgeUnavailableError",
    "JudgeUnparseableError",
    "MTJailBenchTask",
    "mt_jailbench_claim",
]
