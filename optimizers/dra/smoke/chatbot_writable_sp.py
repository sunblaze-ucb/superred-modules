#!/usr/bin/env python3
"""Smoke-test DRA against ChatbotTarget with optional writable system_prompt scope.

This is a small end-to-end harness for inspecting SuperRed plumbing. It runs the
real Controller, ChatbotTarget, and DraOptimizer so we can verify that DRA sends
one disguised user prompt per run, optionally injects the official chat-model
system prompt when writable, reads trajectory responses, and consumes framework
feedback.

Example:
    PYTHONPATH="/path/to/superred/src:/path/to/superred-modules/targets/chatbot/src:/path/to/superred-modules/security_claims/demo_secret_leak/src:/path/to/superred-modules/optimizers/dra/src" \
      python optimizers/dra/smoke/chatbot_writable_sp.py \
      --target-model openai/gpt-4o-mini \
      --api-key "$OPENAI_API_KEY"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from demo_secret_leak_claim import demo_secret_leak_claim
from chatbot_target import SYSTEM_PROMPT_TAG, USER_TAG, ChatbotTarget
from dra_optimizer import DraOptimizer
from superred.core.controller import Controller, TargetFactory
from superred.core.types.llm import LLMConfig
from superred.core.types.security_domain import Scope


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE"))
    parser.add_argument("--max-runs", type=int, default=3)
    parser.add_argument("--max-cost", type=float, default=1.0)
    parser.add_argument("--secret", default="DRA-SMOKE-SECRET")
    parser.add_argument("--trigger", default="spaghetti")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--scope",
        choices=("user", "writable-sp"),
        default="writable-sp",
        help="Use user-only paper path or writable system_prompt + user path.",
    )
    return parser


def _scope(name: str) -> Scope:
    if name == "user":
        return frozenset({USER_TAG})
    return frozenset({SYSTEM_PROMPT_TAG, USER_TAG})


async def _run(args: argparse.Namespace) -> int:
    if not args.api_key:
        raise SystemExit("Provide --api-key or set OPENAI_API_KEY.")

    claim = demo_secret_leak_claim(secret=args.secret, trigger=args.trigger)

    def optimizer_factory() -> DraOptimizer:
        return DraOptimizer(max_attempts=args.max_attempts)

    target_factory = TargetFactory(
        create=lambda: ChatbotTarget(
            model=args.target_model,
            api_key=args.api_key,
            api_base=args.api_base,
        )
    )

    controller = Controller(
        optimizer_factory=optimizer_factory,
        target_factory=target_factory,
        security_claim=claim,
        scope=_scope(args.scope),
        llm_config=LLMConfig(
            model=args.target_model,
            api_base=args.api_base or "",
            api_key=args.api_key,
            max_cost=args.max_cost,
        ),
        max_runs_per_task=args.max_runs,
        include_feedback=True,
    )
    result = await controller.run()
    task_result = result.task_results[0]

    print("\nDRA smoke details")
    print(
        json.dumps(
            {
                "scope": args.scope,
                "success": task_result.success,
                "best_score": task_result.best_score.value,
                "runs": len(task_result.runs),
                "llm_calls": task_result.llm_usage.calls,
                "llm_cost": task_result.llm_usage.cost,
            },
            indent=2,
        )
    )

    for idx, run in enumerate(task_result.runs, start=1):
        print(f"\n--- trajectory {idx} ---")
        for item in run.trajectory.snapshot():
            print(type(item).__name__, item)

    return 0 if task_result.runs else 1


def main() -> int:
    args = _build_parser().parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
