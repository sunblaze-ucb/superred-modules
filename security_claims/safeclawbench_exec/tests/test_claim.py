"""SafeClawBench Exec-Balanced claim tests (offline: mock sandbox + stub agent)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from safeclawbench_exec_claim import (
    SafeClawBenchExecTask,
    safeclawbench_exec_claim,
    safeclawbench_exec_combined_claim,
    safeclawbench_exec_family_claim,
)
from safeclawbench_exec_target import SafeClawBenchExecTarget, load_scenarios
from safeclawbench_exec_target._vendor.executable.schema import Scenario
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.types.events import ControllableNoInjection
from superred.core.types.trajectory import Trajectory


def _resp(content: str) -> Any:
    message = type("M", (), {"content": content})
    return type("R", (), {"choices": [type("C", (), {"message": message})]})


class ScriptedAgent:
    def __init__(self, scripts: list[str]) -> None:
        self.scripts = scripts
        self.calls = 0

    async def complete(self, messages: list[dict[str, str]], **kw: Any) -> Any:
        content = (
            self.scripts[self.calls]
            if self.calls < len(self.scripts)
            else '{"tool_calls": [], "final_response": "done"}'
        )
        self.calls += 1
        return _resp(content)


def _handlers():
    async def send_event(ev: Any) -> Any:
        return ControllableNoInjection(event=ev, controllable=ev.controllable)

    return (lambda ev: None), send_event


def _scenario(case_id: str) -> Scenario:
    return next(s for s in load_scenarios() if s.case_id == case_id)


def _plan(tool: str, args: dict[str, Any]) -> str:
    return json.dumps({"tool_calls": [{"name": tool, "arguments": args}], "final_response": "ok"})


async def _run_task(task: SafeClawBenchExecTask, agent: ScriptedAgent) -> Any:
    target = SafeClawBenchExecTarget(agent_client=agent)
    await task.configure_target(target)
    emit, send = _handlers()
    await target.run(emit, send)
    return await task.evaluate(Trajectory(), target)


# -- factory -----------------------------------------------------------------
def test_factory_enumerates_family() -> None:
    claim = safeclawbench_exec_claim(attack_types=["ADI"], limit=5)
    assert isinstance(claim, SecurityClaim)


def test_family_helper() -> None:
    assert isinstance(safeclawbench_exec_family_claim("MPI", limit=1), SecurityClaim)


def test_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="Unknown attack_types"):
        safeclawbench_exec_claim(attack_types=["NOPE"])


def test_empty_selection_raises() -> None:
    with pytest.raises(ValueError, match="no tasks"):
        safeclawbench_exec_claim(task_ids=["does-not-exist"])


def test_combined_claim() -> None:
    a = safeclawbench_exec_claim(attack_types=["ADI"], limit=1)
    b = safeclawbench_exec_claim(attack_types=["DPI"], limit=1)
    assert isinstance(safeclawbench_exec_combined_claim([a, b]), SecurityClaim)


# -- task scoring (real target + stub agent) ---------------------------------
async def test_task_scores_harm_success() -> None:
    sc = _scenario("ADI_3212")
    task = SafeClawBenchExecTask(sc)
    agent = ScriptedAgent(
        [_plan("write_file", {"path": "/work/adi_3212_review.md", "content": "unsafe_default"})]
    )
    result = await _run_task(task, agent)
    assert result.success is True
    assert result.primary_score.value == 1.0
    assert "HarmCompletion-ASR=True" in result.rationale


async def test_task_scores_safe_no_success() -> None:
    sc = _scenario("ADI_3212")
    task = SafeClawBenchExecTask(sc)
    agent = ScriptedAgent(
        [_plan("write_file", {"path": "/work/adi_3212_review.md", "content": "needs_review"})]
    )
    result = await _run_task(task, agent)
    assert result.success is False
    assert result.primary_score.value == 0.0


async def test_task_raises_to_abstain_without_post_state() -> None:
    # No post_state (e.g. unconfigured scenario) must ABSTAIN, not score "safe":
    # raising marks the run errored so the controller excludes it from metrics.
    sc = _scenario("ADI_3212")
    task = SafeClawBenchExecTask(sc)

    class _NoState:
        def set_config(self, *a: Any, **k: Any) -> None: ...
        def query(self, name: str, **k: Any) -> str:
            return ""

    with pytest.raises(RuntimeError, match="did not complete"):
        await task.evaluate(Trajectory(), _NoState())  # type: ignore[arg-type]


async def test_task_raises_to_abstain_on_model_error() -> None:
    # A model failure (target records `error`) must abstain even though a partial
    # post_state exists — otherwise it scores as a spurious safe refusal.
    sc = _scenario("ADI_3212")
    task = SafeClawBenchExecTask(sc)

    class _Errored:
        def set_config(self, *a: Any, **k: Any) -> None: ...
        def query(self, name: str, **k: Any) -> str:
            if name == "error":
                return "RuntimeError"
            if name == "post_state":
                return (
                    '{"initial_world_state":{},"final_world_state":{},'
                    '"trajectory":{"events":[]}}'
                )
            return ""

    with pytest.raises(RuntimeError, match="RuntimeError"):
        await task.evaluate(Trajectory(), _Errored())  # type: ignore[arg-type]


def test_task_goal_carries_user_prompt() -> None:
    sc = _scenario("ADI_3212")
    task = SafeClawBenchExecTask(sc)
    assert task.goal.description == sc.user_prompt
