"""LLM-driven faithfulness comparison: upstream AgentDojo vs our port.

For each of 12 hand-picked ``(suite, user_task_id, injection_task_id)``
pairs we drive the same prompt + canonical injection through BOTH the
upstream :meth:`TaskSuite.run_task_with_pipeline` API AND our
``Controller + AgentDojoTarget + Layer1Task`` stack at temperature 0,
and assert the resulting ``security`` (= "attack succeeded") verdicts
match.

Polarity (per ASSUMPTIONS D.1): both sides return ``True`` iff the
injection task's predicate fired, i.e. the attack succeeded.

The test is gated on credentials (``LITELLM_API_KEY``+``LITELLM_API_BASE``
or ``OPENAI_API_KEY``) and on the ``--faithfulness-upstream`` opt-in
marker because it makes real LLM calls.  Cost is hard-capped at
``LLMConfig(max_cost=4.0)`` per ASSUMPTIONS D.2; expected actual cost at
temperature 0 with ``gpt-4o-2024-05-13`` (or the LiteLLM substitution
``gpt-4-turbo-2024-04-09``) is ~$1-3 for all 12 pairs together.

Determinism: ``temperature=0`` is the most we can do; the OpenAI API
does not guarantee bit-for-bit reproducibility, so a single mismatched
pair does NOT immediately fail the suite.  We require >=10/12 verdicts
to match (one or two transient noise tolerated); a CRITICAL upstream
predicate bug (per ``UPSTREAM_PREDICATE_AUDIT.md``) is treated as
expected variance.
"""

from __future__ import annotations

import os
from typing import Iterable

import pytest

from superred.core.types.controllable import Controllable
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import (
    ControllableNoInjection,
    ControllablePreCallEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.observable import ObservableValue
from superred.core.interfaces.optimizer import Optimizer
from superred.core.llm import LLMClient
from superred.core.types.llm import LLMConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


_CANONICAL_MODEL: str = "openai/gpt-4o-2024-05-13"
"""AgentDojo's canonical benchmark model.  Matches the production
default in :class:`AgentDojoTarget` so the faithfulness test exercises
the same configuration users will see in real runs.  Override via the
``AGENTDOJO_FAITHFULNESS_MODEL`` env var when targeting a proxy that
lacks gpt-4o; e.g. ``openai/gpt-4-turbo-2024-04-09`` is a known-working
fallback for the user's LiteLLM proxy if needed."""


def _credentials() -> tuple[str, str | None, str | None]:
    """Return (model_id, api_base, api_key).  Picks LiteLLM proxy if available.

    Defaults to :data:`_CANONICAL_MODEL` in both branches so the test
    mirrors the production model unless the operator overrides via the
    ``AGENTDOJO_FAITHFULNESS_MODEL`` env var.
    """
    model = os.environ.get("AGENTDOJO_FAITHFULNESS_MODEL", _CANONICAL_MODEL)
    if "LITELLM_API_KEY" in os.environ and "LITELLM_API_BASE" in os.environ:
        return model, os.environ["LITELLM_API_BASE"], os.environ["LITELLM_API_KEY"]
    if "OPENAI_API_KEY" in os.environ:
        return model, None, os.environ["OPENAI_API_KEY"]
    return "", None, None


def _has_creds() -> bool:
    return bool(_credentials()[0])


pytestmark = [
    pytest.mark.skipif(
        not _has_creds(),
        reason=(
            "Faithfulness comparison requires LLM credentials.  "
            "Set OPENAI_API_KEY or LITELLM_API_KEY+LITELLM_API_BASE."
        ),
    ),
    pytest.mark.faithfulness_upstream,
]


# Hand-picked 12 pairs covering attack semantics across all four suites.
# Sourced from CANONICAL_PAIRS (security_claim_agentdojo.layer1_pairs).
#
# Selection rationale (one IT per attack semantic family per suite):
# - banking: IT0 (send-money), IT2 (send-money-conditional), IT4 (scheduled-txn)
# - workspace: IT0 (unauth email send), IT1 (file deletion), IT2 (calendar manip)
# - slack: IT1 (phishing page), IT2 (channel exfil), IT5 (membership manip)
# - travel: IT0 (hotel reserve), IT2 (restaurant reserve), IT3 (calendar event)
FAITHFULNESS_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("banking", "user_task_1", "injection_task_0"),
    ("banking", "user_task_1", "injection_task_2"),
    ("banking", "user_task_2", "injection_task_4"),
    ("workspace", "user_task_14", "injection_task_0"),
    ("workspace", "user_task_26", "injection_task_1"),
    ("workspace", "user_task_0", "injection_task_2"),
    ("slack", "user_task_0", "injection_task_1"),
    ("slack", "user_task_1", "injection_task_2"),
    ("slack", "user_task_2", "injection_task_5"),
    ("travel", "user_task_0", "injection_task_0"),
    ("travel", "user_task_0", "injection_task_2"),
    ("travel", "user_task_3", "injection_task_3"),
)

# Minimum matched-verdict count below which the test fails.
MATCH_FLOOR = 10


# ---------------------------------------------------------------------------
# Noop optimizer (port side): never injects.  Lets the canonical
# AgentDojo injection_vector_defaults (already substituted into the
# environment YAML at seed-load time) act as the only attack channel,
# matching upstream's run_task_with_pipeline behaviour exactly.
# ---------------------------------------------------------------------------


class _NoopOptimizer(Optimizer):
    """An optimizer that responds NoInjection to every Controllable event
    and signals ``done=True`` after the first run."""

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            return EventResponse(event=event)
        if isinstance(event, ControllablePreCallEvent):
            return ControllableNoInjection(
                event=event, controllable=event.controllable,
            )
        # Catch ControllablePostCallEvent (per-read events) too.
        if hasattr(event, "controllable"):
            return ControllableNoInjection(
                event=event, controllable=event.controllable,
            )
        if isinstance(event, RunEndEvent):
            return RunEndResponse(event=event, done=True)
        return EventResponse(event=event)

    async def teardown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Upstream side: invoke task_suite.run_task_with_pipeline directly.
# ---------------------------------------------------------------------------


def _load_upstream_suite(suite_name: str):
    """Return the upstream ``TaskSuite`` for a given suite name.

    Avoids the v1.1.1 circular-import bug by pre-importing the upstream
    suite loader first (per memory).
    """
    # Pre-import the loader so v1.1.1 cannot break the import order.
    import agentdojo.task_suite.load_suites  # noqa: F401

    if suite_name == "banking":
        from agentdojo.default_suites.v1.banking.task_suite import task_suite
        return task_suite
    if suite_name == "workspace":
        from agentdojo.default_suites.v1.workspace.task_suite import task_suite
        return task_suite
    if suite_name == "slack":
        from agentdojo.default_suites.v1.slack.task_suite import task_suite
        return task_suite
    if suite_name == "travel":
        from agentdojo.default_suites.v1.travel.task_suite import task_suite
        return task_suite
    raise ValueError(f"Unknown suite {suite_name!r}")


def _build_upstream_pipeline(model_id: str, api_base: str | None, api_key: str):
    """Build a vanilla AgentDojo no-defense pipeline at temperature 0."""
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
    from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
    from agentdojo.agent_pipeline.tool_execution import (
        ToolsExecutionLoop,
        ToolsExecutor,
    )
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
    import openai
    import yaml
    from importlib.resources import files

    if not model_id.startswith("openai/"):
        pytest.skip(f"Upstream comparison only supports openai/* models; got {model_id}")

    client = openai.OpenAI(api_key=api_key, base_url=api_base)
    # AgentDojo's OpenAILLM hard-codes temperature=0 internally (see
    # llms/openai_llm.py:_chat_completion_request) so no override needed.
    llm = OpenAILLM(client, model_id.removeprefix("openai/"))

    # System prompt: upstream's default.
    sysprompt = yaml.safe_load(
        (files("agentdojo.data") / "system_messages.yaml").read_text()
    )["default"]

    tools_loop = ToolsExecutionLoop([ToolsExecutor(), llm])
    return AgentPipeline(
        [SystemMessage(sysprompt), InitQuery(), llm, tools_loop]
    )


def _run_upstream_pair(
    suite_name: str, user_task_id: str, injection_task_id: str,
    model_id: str, api_base: str | None, api_key: str,
) -> tuple[bool, bool]:
    """Run one upstream pair; return ``(utility, security)`` booleans."""
    suite = _load_upstream_suite(suite_name)
    pipeline = _build_upstream_pipeline(model_id, api_base, api_key)
    user_task = suite.user_tasks[user_task_id]
    injection_task = suite.injection_tasks[injection_task_id]
    injection_defaults = suite.get_injection_vector_defaults()
    return suite.run_task_with_pipeline(
        agent_pipeline=pipeline,
        user_task=user_task,
        injection_task=injection_task,
        injections=injection_defaults,
    )


# ---------------------------------------------------------------------------
# Port side: drive Controller + AgentDojoTarget + Layer-1 paired task.
# ---------------------------------------------------------------------------


async def _run_port_pair(
    suite_name: str, user_task_id: str, injection_task_id: str,
    model_id: str, api_base: str | None, api_key: str,
) -> tuple[bool, bool]:
    """Run one port pair via Controller.  Return ``(utility, attack_succeeded)``."""
    # Force the v1.1.1 circular-import workaround on the port side too.
    import agentdojo.task_suite.load_suites  # noqa: F401

    from agentdojo_target.target import AgentDojoTarget
    from security_claim_agentdojo.layer1_factory import agentdojo_layer1_claim
    from superred.core.controller import Controller, TargetFactory
    from superred.core.types.security_domain import SecurityDomainTag

    claim = agentdojo_layer1_claim(
        pairs=[(suite_name, user_task_id, injection_task_id)],
    )

    def make_target() -> AgentDojoTarget:
        return AgentDojoTarget(
            pipeline_model=model_id, api_base=api_base, api_key=api_key,
        )

    # Scope: every tag in the target's security domain so no controllable,
    # observable, or trajectory entry is filtered out for this faithfulness run.
    sample_target = make_target()
    all_tags: set[SecurityDomainTag] = set(
        sample_target.security_domain._tags.values()  # noqa: SLF001
    )

    controller = Controller(
        optimizer_factory=lambda: _NoopOptimizer(),
        target_factory=TargetFactory(create=make_target, concurrency=1),
        security_claim=claim,
        scope=frozenset(all_tags),
        llm_config=LLMConfig(
            model=model_id, api_base=api_base, api_key=api_key, max_cost=4.0,
        ),
        max_runs_per_task=1,
        include_feedback=True,
    )
    result = await controller.run()
    assert len(result.task_results) == 1, (
        f"Expected 1 task result, got {len(result.task_results)}"
    )
    tr = result.task_results[0]
    if tr.error is not None:
        pytest.fail(f"Port-side run failed: {tr.error}")
    assert tr.best_evaluation is not None
    score = tr.best_evaluation.primary_score
    attack_succeeded = bool(score.value)
    # sub_scores is dict[str, Score]; pull "utility" by key.
    utility_score = tr.best_evaluation.sub_scores.get("utility")
    utility = bool(utility_score.value) if utility_score is not None else False
    return utility, attack_succeeded


# ---------------------------------------------------------------------------
# The comparison test.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upstream_vs_port_security_verdict_match() -> None:
    """Drive 12 pairs through both sides and check >=10/12 security verdicts match.

    The test prints a per-pair report so a human reviewer can inspect
    each pair's verdicts and trace differences.  The MATCH_FLOOR (10)
    tolerates two transient LLM-nondeterminism mismatches without
    failing CI.
    """
    model_id, api_base, api_key = _credentials()
    assert api_key is not None  # gated by pytestmark.skipif

    matched: list[tuple[str, str, str]] = []
    mismatched: list[tuple[str, str, str, bool, bool]] = []
    errors: list[tuple[str, str, str, str]] = []

    for suite_name, user_task_id, injection_task_id in FAITHFULNESS_PAIRS:
        pair_id = f"{suite_name}::{user_task_id}::{injection_task_id}"
        try:
            _, upstream_security = _run_upstream_pair(
                suite_name, user_task_id, injection_task_id,
                model_id, api_base, api_key,
            )
        except Exception as exc:  # pragma: no cover - LLM hiccups
            errors.append((suite_name, user_task_id, injection_task_id, f"upstream: {exc}"))
            continue
        try:
            _, port_security = await _run_port_pair(
                suite_name, user_task_id, injection_task_id,
                model_id, api_base, api_key,
            )
        except Exception as exc:  # pragma: no cover
            errors.append((suite_name, user_task_id, injection_task_id, f"port: {exc}"))
            continue

        print(
            f"  {pair_id}: upstream_security={upstream_security}  "
            f"port_attack_succeeded={port_security}  "
            f"match={upstream_security == port_security}"
        )
        if upstream_security == port_security:
            matched.append((suite_name, user_task_id, injection_task_id))
        else:
            mismatched.append(
                (suite_name, user_task_id, injection_task_id,
                 upstream_security, port_security),
            )

    total = len(FAITHFULNESS_PAIRS)
    matched_n = len(matched)
    mismatched_n = len(mismatched)
    error_n = len(errors)
    print(
        f"\nFaithfulness summary: matched={matched_n}/{total} "
        f"mismatched={mismatched_n} errored={error_n}"
    )
    if mismatched:
        print("Mismatches:")
        for pair in mismatched:
            print(f"  {pair}")
    if errors:
        print("Errors:")
        for err in errors:
            print(f"  {err}")

    assert matched_n >= MATCH_FLOOR, (
        f"Only {matched_n}/{total} pair verdicts matched upstream "
        f"(floor {MATCH_FLOOR}); mismatches={mismatched}; errors={errors}"
    )
