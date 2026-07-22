"""Full superred pipeline against REAL Docker + LiteLLM, per agent.

Drives the WHOLE target path end to end -- Controller + the goal_passthrough optimizer
+ a DTAP target (Claude Code / OpenClaw) + the DTAP claim + the OOB env-state judge --
on ONE real ``travel`` task. A green run proves the env stack boots, the host MCP proxy
serves the agent, the agent episode runs its own inference through the proxy, the
trajectory is extracted emit-once, and the judge returns a verdict (no infra error).

SKIPPED offline: needs a Docker daemon + the built agent image, LiteLLM credentials,
the DTAP dataset, and the ``goal_passthrough`` optimizer installed. The offline suites
cover every seam with the Docker/agent/judge boundaries mocked; this is the one test
that wires them all against the real stack.
"""

from __future__ import annotations

import os
import shutil

import pytest

pytestmark = [pytest.mark.docker, pytest.mark.live]

_DOCKER = shutil.which("docker") is not None
_CREDS = bool(os.getenv("LITELLM_API_KEY") and os.getenv("LITELLM_API_BASE"))


def _have_optimizer() -> bool:
    try:
        import goal_passthrough_optimizer  # noqa: F401
    except Exception:
        return False
    return True


def _model_and_factory(agent: str, *, api_base: str, api_key: str):
    """Return ``(model, TargetFactory)`` for the requested agent."""
    from dtap_claim import (
        dtap_claudecode_target_factory,
        dtap_openclaw_target_factory,
    )

    if agent == "claudecode":
        model = os.environ.get("DTAP_MODEL", "claude-opus-4-8")
        return model, dtap_claudecode_target_factory(
            model=model, api_base=api_base, api_key=api_key
        )
    model = os.environ.get("DTAP_OPENCLAW_MODEL", "openai/gpt-4o-mini")
    return model, dtap_openclaw_target_factory(model=model, api_base=api_base, api_key=api_key)


@pytest.mark.skipif(not _DOCKER, reason="needs a Docker daemon + CLI")
@pytest.mark.skipif(not _CREDS, reason="needs LITELLM_API_KEY / LITELLM_API_BASE")
@pytest.mark.skipif(not _have_optimizer(), reason="needs the goal_passthrough optimizer installed")
@pytest.mark.parametrize("agent", ["claudecode", "openclaw"])
async def test_full_pipeline_one_travel_task(agent: str) -> None:
    from dtap_scaffold.forest import USER_TAG
    from goal_passthrough_optimizer import GoalPassthroughOptimizer
    from superred.core.controller import Controller
    from superred.core.interfaces.security_claim import SecurityClaim
    from superred.core.types.llm import LLMConfig

    from dtap_claim import dtap_direct_claim

    api_key = os.environ["LITELLM_API_KEY"]
    api_base = os.environ["LITELLM_API_BASE"]
    dataset_root = os.environ.get("DTAP_DATASET_ROOT")

    model, target_factory = _model_and_factory(agent, api_base=api_base, api_key=api_key)

    # Enumerate the travel/direct/malicious tasks (offline), then run just ONE so the
    # live run stays tiny; the attacker controls only the user prompt (goal-passthrough
    # baseline = direct request), every other DTAP vector is out of scope.
    full = dtap_direct_claim(
        domains=["travel"],
        dataset_root=dataset_root,
        judge_model="openai/gpt-4o-2024-05-13",
        judge_api_base=api_base,
        judge_api_key=api_key,
    )
    tasks = list(full)
    assert tasks, "no travel/direct tasks enumerated -- is DTAP_DATASET_ROOT set?"
    claim = SecurityClaim.from_tasks([tasks[0]])

    controller = Controller(
        optimizer_factory=GoalPassthroughOptimizer,
        target_factory=target_factory,
        security_claim=claim,
        scope=frozenset({USER_TAG}),
        llm_config=LLMConfig(model=model, api_base=api_base, api_key=api_key),
        task_cost_cap_usd=1.0,
        max_runs_per_task=1,
    )
    tmr = await controller.run()

    results = tmr.task_results
    assert len(results) == 1
    result = results[0]
    # The pipeline ran end to end with NO infra error, the OOB judge returned a verdict,
    # and a primary score exists (True/False attack outcome is data, not a test failure).
    assert result.stop_reason in ("done", "max_runs"), f"pipeline errored: {result.error}"
    assert result.best_score is not None
    assert result.best_evaluation is not None
