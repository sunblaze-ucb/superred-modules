"""Faithfulness: OUR scoring == upstream combined_scorer on identical messages.

This is the precise faithfulness claim.  For a sample behavior we run OUR
rollout to produce the message trace, then score that SAME trace two ways:

1. OUR pipeline: ``grading_bridge.grade`` + ``grading_bridge.check_refusal``.
2. The real upstream ``inspect_evals.agentharm.scorer.combined_scorer`` over a
   constructed ``TaskState`` holding the same messages + metadata.

At temperature 0 with the same judge model these must agree.  Credentialed
(real LLM calls), so skipped without proxy/OpenAI credentials.

Run with::

    LITELLM_API_KEY=... LITELLM_API_BASE=... \
      pytest -m faithfulness security_claims/agentharm/tests/faithfulness
"""

from __future__ import annotations

import os

import pytest
from inspect_ai.model import GenerateConfig, ModelName, get_model
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
from inspect_evals.agentharm.benchmark import harmful_grading_functions
from inspect_evals.agentharm.scorer import RefusalJudgeLLM, combined_scorer

from security_claim_agentharm.agentharm_tools import agentharm_tool_resolver
from security_claim_agentharm.dataset_loader import load_agentharm_dataset
from security_claim_agentharm.grading_bridge import check_refusal, grade
from security_claim_agentharm.prompts import faithful_system_prompt, faithful_user_prompt

pytestmark = pytest.mark.faithfulness

JUDGE = "openai/gpt-4o-2024-08-06"


def _one_behavior_per_category() -> list[str]:
    """One representative behavior id per harm category (8), for broad parity."""
    seen: dict[str, str] = {}
    for sample in load_agentharm_dataset():
        cat = str(sample.metadata["category"])
        if cat not in seen:
            seen[cat] = str(sample.id)
    return [seen[c] for c in sorted(seen)]


BEHAVIORS = _one_behavior_per_category()


def _creds() -> tuple[str, str | None, str | None] | None:
    if "LITELLM_API_KEY" in os.environ and "LITELLM_API_BASE" in os.environ:
        return (
            os.environ.get("LITELLM_MODEL", "openai/gpt-4o-mini"),
            os.environ["LITELLM_API_BASE"],
            os.environ["LITELLM_API_KEY"],
        )
    if "OPENAI_API_KEY" in os.environ:
        return (
            os.environ.get("OPENAI_MODEL", "openai/gpt-4o-mini"),
            None,
            os.environ["OPENAI_API_KEY"],
        )
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize("behavior_id", BEHAVIORS)
async def test_our_scoring_matches_upstream(behavior_id: str) -> None:
    creds = _creds()
    if creds is None:
        pytest.skip("no LLM credentials (LITELLM_API_KEY+BASE or OPENAI_API_KEY)")
    model_id, api_base, api_key = creds

    ds = load_agentharm_dataset(behavior_ids=[behavior_id])
    sample = ds[0]
    md = dict(sample.metadata or {})

    # OUR rollout -> messages.
    from inspect_agent_target.rollout import run_rollout, static_tools_provider

    target_model = get_model(
        model_id, base_url=api_base, api_key=api_key,
        config=GenerateConfig(temperature=0.0, max_tokens=4096),
    )
    tools = [agentharm_tool_resolver(n) for n in md["target_functions"]]
    messages = await run_rollout(
        target_model,
        system_prompt=faithful_system_prompt(model_id),
        user_prompt=faithful_user_prompt(md["prompt"]),
        tools_provider=static_tools_provider(tools),
        tool_choice="auto",
        message_limit=20,
    )

    # OUR scoring.
    semantic_judge = get_model(JUDGE, base_url=api_base, api_key=api_key)
    refusal_judge = RefusalJudgeLLM(
        judge_model=get_model(JUDGE, base_url=api_base, api_key=api_key)
    )
    our_score, _our_dict, our_err = await grade(md, messages, semantic_judge)
    our_refusal, _ = await check_refusal(messages, refusal_judge)
    assert our_err is None

    # Upstream scoring over the SAME messages.
    state = TaskState(
        model=ModelName(model_id),
        sample_id=behavior_id,
        epoch=0,
        input=md["prompt"],
        messages=list(messages),
        metadata=dict(md),
    )
    scorer_fn = combined_scorer(harmful_grading_functions, refusal_judge, semantic_judge)
    upstream = await scorer_fn(state, Target([""]))

    assert our_score == pytest.approx(float(upstream.value["score"]))
    assert float(our_refusal) == pytest.approx(float(upstream.value["refusal"]))
