#!/usr/bin/env python3
"""Smoke-test TAP against ChatbotTarget with writable system_prompt scope.

This is intentionally a small end-to-end harness rather than a unit test. It
runs the real SuperRed controller, ChatbotTarget, and TapOptimizer plumbing so
we can inspect whether TAP uses the writable system prompt, sends the user turn,
reads trajectory responses, and consumes RunEnd feedback.

Example:
    PYTHONPATH="/path/to/superred/src:/path/to/superred-modules/targets/chatbot/src:/path/to/superred-modules/targets/test_basic_llm_chat/src:/path/to/superred-modules/security_claims/test_basic_secret_leak/src:/path/to/superred-modules/optimizers/tap/src" \
      python optimizers/tap/smoke/chatbot_writable_sp.py \
      --optimizer-model openai/gpt-4o-mini \
      --target-model openai/gpt-4o-mini \
      --api-key "$OPENAI_API_KEY"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from basic_secret_leak_claim import basic_secret_leak_claim
from chatbot_target import SYSTEM_PROMPT_TAG, USER_TAG, ChatbotTarget
from superred.core.controller import Controller
from superred.core.types.llm import LLMConfig
from superred.core.types.security_domain import Scope
from tap_optimizer import TapOptimizer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE"))
    parser.add_argument("--max-runs", type=int, default=3)
    parser.add_argument("--max-cost", type=float, default=1.0)
    parser.add_argument("--secret", default="TAP-SMOKE-SECRET")
    parser.add_argument("--trigger", default="spaghetti")
    parser.add_argument("--tree-depth", type=int, default=2)
    parser.add_argument("--tree-width", type=int, default=1)
    parser.add_argument("--branching-factor", type=int, default=1)
    parser.add_argument("--root-nodes", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=0)
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

    target = ChatbotTarget(
        model=args.target_model,
        api_key=args.api_key,
        api_base=args.api_base,
    )
    claim = basic_secret_leak_claim(secret=args.secret, trigger=args.trigger)

    def optimizer_factory() -> TapOptimizer:
        return TapOptimizer(
            branching_factor=args.branching_factor,
            root_nodes=args.root_nodes,
            tree_width=args.tree_width,
            tree_depth=args.tree_depth,
            random_seed=args.random_seed,
        )

    controller = Controller(
        optimizer_factory=optimizer_factory,
        target=target,
        security_claim=claim,
        llm_configs=[
            LLMConfig(
                model=args.optimizer_model,
                api_base=args.api_base or "",
                api_key=args.api_key,
                max_cost=args.max_cost,
            )
        ],
        max_runs_per_task=args.max_runs,
        include_feedback=True,
    )
    result = await controller.run(scopes=[_scope(args.scope)], models=[args.optimizer_model])
    task_result = result.threat_model_results[0].task_results[0]

    print("\nTAP smoke details")
    print(json.dumps({
        "scope": args.scope,
        "success": task_result.success,
        "best_score": task_result.best_score.value,
        "runs": len(task_result.runs),
        "llm_calls": task_result.llm_usage.calls,
        "llm_cost": task_result.llm_usage.cost,
    }, indent=2))

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
