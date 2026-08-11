#!/usr/bin/env python3
"""Full end-to-end live test across the parity mode configs.

Runs the real superred controller + ChatbotTarget + StrongREJECT judge against a
few tasks under three configs -- (a) default separate + assembly, (b)
recursive-leaf, (c) goal-as-root + A/B/C fallback -- to validate that the
harnessing works reliably end to end with a real LLM: units map onto runs, fresh
conversations start new runs, the assembly run is framework-judged, budget/time
are respected, and nothing crashes. Small by design (cents).

Note on the attacker model: a safety-tuned attacker refuses the decomposition
meta-request on harmful goals, so decomposition falls back to the goal (the
vendored behaviour); the scaffolding -- wrappers, fresh-conversation runs, the
assembly run, fallback dispatch, the framework verdict -- is still exercised. Use
a weakly-aligned attacker (the upstream's Qwen) to see multi-sub-task plans.

Example:
    PYTHONPATH="...superred/src:...targets/chatbot/src:...security_claims/strongreject/src:...optimizers/attack_anything/src" \
      python optimizers/attack_anything/smoke/full_parity_live.py \
      --api-base "$BASE" --api-key "$KEY" --model openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import asyncio
import time

import strongreject_claim as sr
from chatbot_target import RESPONSE_READABLE_TAG, USER_TAG, ChatbotTarget
from superred.core.controller import Controller, TargetFactory
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.types.events import ControllableInjection
from superred.core.types.llm import LLMConfig

from attack_anything_optimizer import AttackAnythingConfig, AttackAnythingOptimizer

_CONFIGS = {
    "default": AttackAnythingConfig(
        n_iterations=2, n_steps=3, max_turns=2, n_early_stop_successes=1
    ),
    "recursive_leaf": AttackAnythingConfig(
        recursive_leaf_attack=True,
        recursive_max_depth=1,
        recursive_branch=2,
        recursive_wrappers_per_leaf=2,
        n_steps=2,
        max_target_queries_per_goal=12,
        n_iterations=1,
        n_early_stop_successes=1,
    ),
    "goal_as_root": AttackAnythingConfig(
        goal_as_root=True,
        fallback_enabled=True,
        recursive_max_depth=1,
        recursive_branch=2,
        recursive_wrappers_per_leaf=2,
        max_target_queries_per_goal=15,
        n_early_stop_successes=1,
    ),
}


async def run_config(name: str, cfg: AttackAnythingConfig, args: argparse.Namespace) -> None:
    claim = SecurityClaim.from_tasks(
        list(
            sr.strongreject_small_claim(
                judge_api_base=args.api_base, judge_api_key=args.api_key, judge_model=args.model
            )
        )[: args.tasks]
    )
    target_factory = TargetFactory(
        create=lambda: ChatbotTarget(
            model=args.model, api_key=args.api_key, api_base=args.api_base
        ),
        concurrency=2,
    )
    controller = Controller(
        optimizer_factory=lambda: AttackAnythingOptimizer(config=cfg),
        target_factory=target_factory,
        security_claim=claim,
        scope=frozenset({USER_TAG, RESPONSE_READABLE_TAG}),
        llm_config=LLMConfig(model=args.model, api_base=args.api_base, api_key=args.api_key),
        task_cost_cap_usd=args.max_cost,
        max_runs_per_task=args.max_runs,
        task_time_cap_s=args.max_time,
        include_feedback=True,
        persist=False,
        report=False,
    )
    t0 = time.time()
    result = await controller.run()
    dt = time.time() - t0
    n_success = sum(1 for tr in result.task_results if tr.success)
    cost = sum(tr.llm_usage.cost for tr in result.task_results)
    errors = [tr.stop_reason for tr in result.task_results if tr.stop_reason == "error"]
    print(
        f"[{name:14s}] {dt:5.1f}s  tasks={len(result.task_results)} success={n_success} "
        f"errors={len(errors)} attacker_cost=${cost:.4f} "
        f"stop_reasons={sorted({tr.stop_reason for tr in result.task_results})}"
    )
    # One conversation from task 0, to show the unit->run mapping + assembly.
    tr = result.task_results[0]
    print(f"    runs={len(tr.runs)}  (each run = one fresh victim conversation / assembly)")
    for j, run in enumerate(tr.runs[:12]):
        injects = [
            it.value for it in run.trajectory.snapshot() if isinstance(it, ControllableInjection)
        ]
        is_assembly = any("Original objective" in v for v in injects)
        tag = " <ASSEMBLY>" if is_assembly else ""
        print(
            f"      run {j}: injects={len(injects)} score={run.evaluation.primary_score.value:.2f} "
            f"success={run.evaluation.success}{tag}"
        )
    if errors:
        print(f"    ERROR sample: {result.task_results[0].error!r}")


async def main(args: argparse.Namespace) -> None:
    modes = args.modes.split(",") if args.modes else list(_CONFIGS)
    for name in modes:
        await run_config(name, _CONFIGS[name], args)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api-base", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--tasks", type=int, default=2)
    p.add_argument("--max-cost", type=float, default=0.30)
    p.add_argument("--max-runs", type=int, default=40)
    p.add_argument("--max-time", type=float, default=240.0)
    p.add_argument("--modes", default="", help="comma list; default all three")
    return p


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(_parser().parse_args())))
