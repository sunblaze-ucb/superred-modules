#!/usr/bin/env python3
"""Smoke-test Attack Anything against ChatbotTarget end to end.

A small real-controller harness (not a unit test): runs the SuperRed controller,
ChatbotTarget, and AttackAnythingOptimizer so you can watch the attack decompose
the goal, walk the sub-tasks as one conversation, read the victim's replies from
the trajectory, and consume the framework verdict. Proves the inverted event loop
against a live LLM.

Example:
    PYTHONPATH="/path/to/superred/src:/path/to/superred-modules/targets/chatbot/src:/path/to/superred-modules/security_claims/demo_secret_leak/src:/path/to/superred-modules/optimizers/attack_anything/src" \
      python optimizers/attack_anything/smoke/chatbot_writable_sp.py \
      --optimizer-model openai/gpt-4o-mini \
      --target-model openai/gpt-4o-mini \
      --api-key "$OPENAI_API_KEY"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from attack_anything_optimizer import AttackAnythingConfig, AttackAnythingOptimizer
from chatbot_target import (
    RESPONSE_READABLE_TAG,
    SYSTEM_PROMPT_TAG,
    USER_TAG,
    ChatbotTarget,
)
from demo_secret_leak_claim import demo_secret_leak_claim
from superred.core.controller import Controller, TargetFactory
from superred.core.types.llm import LLMConfig
from superred.core.types.security_domain import Scope


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE"))
    parser.add_argument("--max-runs", type=int, default=6)
    parser.add_argument("--max-cost", type=float, default=1.0)
    parser.add_argument("--secret", default="AA-SMOKE-SECRET")
    parser.add_argument("--trigger", default="spaghetti")
    parser.add_argument("--n-iterations", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--scope", choices=("user", "writable-sp"), default="user")
    return parser


def _scope(name: str) -> Scope:
    # RESPONSE_READABLE_TAG is required for the attacker to READ the victim's
    # replies: the chatbot target tags its reply observable there, so without it
    # a user-only scope leaves the attacker blind and the multi-turn feedback
    # loop degrades to static probes.
    if name == "user":
        return frozenset({USER_TAG, RESPONSE_READABLE_TAG})
    return frozenset({SYSTEM_PROMPT_TAG, USER_TAG, RESPONSE_READABLE_TAG})


async def _run(args: argparse.Namespace) -> int:
    if not args.api_key:
        raise SystemExit("Provide --api-key or set OPENAI_API_KEY.")

    target_factory = TargetFactory(
        create=lambda: ChatbotTarget(
            model=args.target_model, api_key=args.api_key, api_base=args.api_base
        ),
    )
    claim = demo_secret_leak_claim(secret=args.secret, trigger=args.trigger)

    def optimizer_factory() -> AttackAnythingOptimizer:
        return AttackAnythingOptimizer(
            config=AttackAnythingConfig(
                n_iterations=args.n_iterations,
                n_steps=args.n_steps,
                max_turns=args.max_turns,
                n_early_stop_successes=1,
            )
        )

    controller = Controller(
        optimizer_factory=optimizer_factory,
        target_factory=target_factory,
        security_claim=claim,
        scope=_scope(args.scope),
        llm_config=LLMConfig(
            model=args.optimizer_model,
            api_base=args.api_base or "",
            api_key=args.api_key,
        ),
        task_cost_cap_usd=args.max_cost,
        max_runs_per_task=args.max_runs,
        include_feedback=True,
    )
    result = await controller.run()
    task_result = result.task_results[0]

    print("\nAttack Anything smoke details")
    print(
        json.dumps(
            {
                "scope": args.scope,
                "success": task_result.success,
                "best_score": task_result.best_score.value,
                "runs": len(task_result.runs),
                "stop_reason": task_result.stop_reason,
                "llm_calls": task_result.llm_usage.calls,
                "llm_cost": task_result.llm_usage.cost,
            },
            indent=2,
        )
    )
    return 0 if task_result.runs else 1


def main() -> int:
    args = _build_parser().parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
