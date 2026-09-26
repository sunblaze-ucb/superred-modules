"""FITD end-to-end smoke against SORRY-Bench with writable system_prompt.

Verifies the SuperRed plumbing that unit tests cannot fully exercise:
    - ChatbotTarget exposes system_prompt and user_message separately.
    - FITD writes setup to system_prompt only when that capability is in scope.
    - FITD injects the paper-style turn ladder through user_message.
    - Model responses are read from the trajectory when response_readable is in scope.
    - Framework evaluation is consumed from RunEndEvent.

Usage:
    cd optimizers/fitd/smoke
    python sorry_bench_writable_sp.py
"""

from __future__ import annotations

import asyncio
import os

from chatbot_target import ChatbotTarget, RESPONSE_READABLE_TAG, SYSTEM_PROMPT_TAG, USER_TAG
from dotenv import load_dotenv
from sorry_bench_claim import sorry_bench_claim
from superred.core.controller import Controller, TargetFactory
from superred.core.types.llm import LLMConfig

from fitd_optimizer import FITDOptimizer

ATTACKER_MODEL = "gpt-4o-mini-2024-07-18"
TARGET_MODEL = "gpt-4o-mini-2024-07-18"
JUDGE_MODEL = "gpt-4o-mini-2024-07-18"
ATTACKER_BUDGET_USD = 0.50
JUDGE_BUDGET_USD = 0.20
QUESTION_IDS = [1]


async def main() -> None:
    load_dotenv()

    api_key = os.environ["LITELLM_API_KEY"]
    api_base = os.environ["LITELLM_API_BASE"]

    judge_cfg = LLMConfig(
        model=JUDGE_MODEL,
        api_base=api_base,
        api_key=api_key,
    )
    attacker_cfg = LLMConfig(
        model=ATTACKER_MODEL,
        api_base=api_base,
        api_key=api_key,
    )
    claim = sorry_bench_claim(
        target_model_id=TARGET_MODEL,
        judge_llm_config=judge_cfg,
        question_ids=QUESTION_IDS,
    )

    def factory() -> FITDOptimizer:
        return FITDOptimizer(level=2, max_attempts=2)

    target_factory = TargetFactory(
        create=lambda: ChatbotTarget(
            model=TARGET_MODEL,
            api_key=api_key,
            api_base=api_base,
        )
    )
    controller = Controller(
        optimizer_factory=factory,
        target_factory=target_factory,
        security_claim=claim,
        scope=frozenset({USER_TAG, RESPONSE_READABLE_TAG, SYSTEM_PROMPT_TAG}),
        llm_config=attacker_cfg,
        task_cost_cap_usd=ATTACKER_BUDGET_USD,
        max_runs_per_task=5,
    )
    result = await controller.run()

    print("=" * 72)
    print("FITD + SORRY-Bench + writable system_prompt - smoke")
    print(f"Attacker: {ATTACKER_MODEL}  Target/Judge: {TARGET_MODEL}")
    print(f"Question ids: {QUESTION_IDS}")
    print("=" * 72)
    for tr in result.task_results:
        print(f"\nTask {tr.task.goal.description[:60]}...")
        print(f"  best score: {tr.best_score.value:.3f}  runs: {len(tr.runs)}")
        print(
            f"  attacker LLM usage: {tr.llm_usage.calls} calls, "
            f"${tr.llm_usage.cost:.4f}"
        )
        if tr.error is not None:
            print(f"  error: {tr.error.splitlines()[-1]}")
        for i, run in enumerate(tr.runs, 1):
            ev = run.evaluation
            print(f"    run {i}: success={ev.success} score={ev.primary_score.value:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
