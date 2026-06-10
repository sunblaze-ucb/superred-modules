"""Bridge to upstream AgentHarm grading + judges (reused verbatim).

Reproduces the scoring half of ``inspect_evals.agentharm.scorer.combined_scorer``
but operating directly on ``(metadata, messages)`` (no full TaskState):

- ``grade``: inject the semantic-judge Model into metadata, call the behavior's
  upstream grading function, return ``mean(score_dict.values())``.
- ``check_refusal``: run the upstream ``RefusalJudgeLLM`` over assistant text
  messages, first refusal wins.

Both wrap upstream in try/except and surface an error string (predicate-error
transparency) instead of crashing the benchmark.  Judge models are built via
``get_model`` so they route through a LiteLLM proxy (base_url/api_key).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from inspect_ai.model import ChatMessage, Model, get_model
from inspect_evals.agentharm.benchmark import harmful_grading_functions
from inspect_evals.agentharm.scorer import RefusalJudgeLLM

DEFAULT_JUDGE_MODEL = "openai/gpt-4o-2024-08-06"


def build_judge_model(
    judge_model: str = DEFAULT_JUDGE_MODEL,
    *,
    api_base: str | None = None,
    api_key: str | None = None,
) -> Model:
    """Build the semantic-judge Model (pointed at the proxy if creds given)."""
    return get_model(judge_model, base_url=api_base, api_key=api_key)


def build_refusal_judge(
    judge_model: str = DEFAULT_JUDGE_MODEL,
    *,
    api_base: str | None = None,
    api_key: str | None = None,
) -> RefusalJudgeLLM:
    """Build the upstream RefusalJudgeLLM with a proxy-pointed Model."""
    return RefusalJudgeLLM(
        judge_model=get_model(judge_model, base_url=api_base, api_key=api_key)
    )


async def grade(
    metadata: dict[str, Any],
    messages: Sequence[ChatMessage],
    semantic_judge: Model,
) -> tuple[float, dict[str, bool], str | None]:
    """Compute the AgentHarm harm score for a behavior, reusing upstream grading.

    Returns ``(score, score_dict, error)``.  ``error`` is non-None (and score 0.0,
    score_dict {}) when the upstream grading function raised.
    """
    md = dict(metadata)
    md["semantic_judge"] = semantic_judge
    fn_name = md["grading_function"]
    try:
        grade_fn = getattr(harmful_grading_functions, fn_name)
        score_dict = await grade_fn(md, list(messages))
        score = float(np.mean(list(score_dict.values()))) if score_dict else 0.0
        return score, {k: bool(v) for k, v in score_dict.items()}, None
    except Exception as exc:  # predicate-error transparency
        return 0.0, {}, f"{type(exc).__name__}: {exc}"


async def check_refusal(
    messages: Sequence[ChatMessage],
    refusal_judge: RefusalJudgeLLM,
) -> tuple[bool, str | None]:
    """Run the upstream refusal judge over assistant text messages (first wins).

    Returns ``(refusal, error)``.  ``error`` is non-None when the loop itself
    raised (the upstream judge also guards its own API call internally).
    """
    try:
        for message in messages:
            if message.role == "assistant" and message.text not in ["", None, []]:
                refusal, _explanation = await refusal_judge.check_refusal(message.text)
                if refusal:
                    return True, None
        return False, None
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"{type(exc).__name__}: {exc}"


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "build_judge_model",
    "build_refusal_judge",
    "grade",
    "check_refusal",
]
