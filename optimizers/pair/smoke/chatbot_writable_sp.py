"""PAIR end-to-end smoke against SORRY-Bench with ChatbotTarget.

Verifies the important SuperRed plumbing for PAIR:
    - user_message injection works with ChatbotTarget;
    - writable system_prompt is used only when the attacker returns a
      non-empty optional ``system_prompt`` JSON field;
    - model responses are read from the filtered trajectory;
    - RunEndEvent evaluation is consumed as framework feedback.

Bounded for cost: one question_id, one PAIR stream, one PAIR iteration,
and gpt-4o-mini for attacker, target, and judge.

Usage:
    cd optimizers/pair/smoke
    python chatbot_writable_sp.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from chatbot_target import ChatbotTarget, RESPONSE_READABLE_TAG, SYSTEM_PROMPT_TAG, USER_TAG
from dotenv import load_dotenv
from security_claim_sorry_bench import sorry_bench_claim
from superred.core.controller import Controller, TargetFactory
from superred.core.types.llm import LLMConfig

from pair_optimizer import PAIROptimizer

ATTACKER_MODEL = "gpt-4o-mini-2024-07-18"
TARGET_MODEL = "gpt-4o-mini-2024-07-18"
JUDGE_MODEL = "gpt-4o-mini-2024-07-18"
ATTACKER_BUDGET_USD = 0.50
JUDGE_BUDGET_USD = 0.20
QUESTION_IDS = [1]


async def main() -> None:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[4] / "superred-experiments" / "test_sorry_bench_chatbot" / ".env",
        here.parents[3] / "superred-experiments" / "test_sorry_bench_chatbot" / ".env",
    ]
    for env_path in candidates:
        if env_path.exists():
            load_dotenv(env_path)
            break
    else:
        load_dotenv()

    api_key = os.environ["LITELLM_API_KEY"]
    api_base = os.environ["LITELLM_API_BASE"]

    target_factory = TargetFactory(
        create=lambda: ChatbotTarget(
            model=TARGET_MODEL, api_key=api_key, api_base=api_base,
        ),
    )
    judge_cfg = LLMConfig(
        model=JUDGE_MODEL,
        api_base=api_base,
        api_key=api_key,
        max_cost=JUDGE_BUDGET_USD,
    )
    attacker_cfg = LLMConfig(
        model=ATTACKER_MODEL,
        api_base=api_base,
        api_key=api_key,
        max_cost=ATTACKER_BUDGET_USD,
    )
    claim = sorry_bench_claim(
        target_model_id=TARGET_MODEL,
        judge_llm_config=judge_cfg,
        question_ids=QUESTION_IDS,
    )

    def factory() -> PAIROptimizer:
        return PAIROptimizer(n_streams=1, n_iterations=1, judge_method="gcg")

    controller = Controller(
        optimizer_factory=factory,
        target_factory=target_factory,
        security_claim=claim,
        scope=frozenset({USER_TAG, RESPONSE_READABLE_TAG, SYSTEM_PROMPT_TAG}),
        llm_config=attacker_cfg,
        max_runs_per_task=3,
    )
    result = await controller.run()

    print("=" * 72)
    print("PAIR + SORRY-Bench + ChatbotTarget writable system_prompt - smoke")
    print(f"Attacker: {ATTACKER_MODEL}  Target/Judge: {TARGET_MODEL}")
    print(f"Question ids: {QUESTION_IDS}")
    print("=" * 72)
    for tr in result.task_results:
        print(f"\nTask {tr.task.goal.description[:60]}...")
        print(f"  best score: {tr.best_score.value:.3f}  runs: {len(tr.runs)}")
        print(f"  attacker usage: {tr.llm_usage.calls} calls, ${tr.llm_usage.cost:.4f}")
        for i, run in enumerate(tr.runs, 1):
            ev = run.evaluation
            print(f"    run {i}: success={ev.success} score={ev.primary_score.value:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
