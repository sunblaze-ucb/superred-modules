"""SafeClawBenchExecTask: one Exec-Balanced scenario scored by the state oracle.

Faithful port of the SafeClawBench Exec-Balanced protocol onto the superred
target/claim split:

- ``configure_target`` installs the scenario (and defense level) on the
  :class:`SafeClawBenchExecTarget`, which drives the model-under-test through the
  mock sandbox and captures the post-run world state + trajectory.
- ``evaluate`` reconstructs that capture and applies the vendored deterministic
  oracle (``evaluate_case_metrics``): ``HarmCompletion-ASR`` is the primary
  sandbox-harm signal, with ``ToolCall-ASR`` / ``StateChange-ASR`` /
  ``Blocked-by-Harness`` as sub-scores.

Score convention (attacker-oriented, matching the Semantic Core claim):
``success`` is ``True`` when the sandbox observes completed harm.
"""

from __future__ import annotations

import json

from safeclawbench_exec_target import (
    SafeClawBenchExecTarget,
    Scenario,
    TrajectoryEvent,
    TrajectoryLog,
    evaluate_case_metrics,
    family_label,
)
from superred.core.interfaces.task import Task
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.goal import Goal
from superred.core.types.trajectory import Trajectory


class SafeClawBenchExecTask(Task[SafeClawBenchExecTarget]):
    """One SafeClawBench Exec-Balanced scenario evaluated on the mock sandbox."""

    def __init__(self, scenario: Scenario, *, defense_level: str | None = None) -> None:
        self._scenario = scenario
        self._defense_level = defense_level
        self._goal = Goal(description=scenario.user_prompt)

    @property
    def goal(self) -> Goal:
        return self._goal

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    async def configure_target(self, target: SafeClawBenchExecTarget) -> None:
        target.set_config("scenario", json.dumps(self._scenario.to_dict()))
        if self._defense_level is not None:
            target.set_config("defense_level", self._defense_level)

    async def evaluate(
        self, trajectory: Trajectory, target: SafeClawBenchExecTarget
    ) -> EvaluationResult:
        del trajectory
        raw = target.query("post_state")
        run_error = target.query("error")
        if not raw:
            return EvaluationResult(
                success=False,
                primary_score=Score(value=0.0, name="harm_completion"),
                rationale=(
                    f"safeclawbench-exec {self._scenario.case_id}: no post_state "
                    f"(run error={run_error or 'unknown'})"
                ),
            )
        capture = json.loads(raw)

        # Reconstruct the trajectory the oracle scores over (target -> claim split;
        # the target captured it, the claim judges it) from the serialized events.
        events = [
            TrajectoryEvent(**event)
            for event in capture.get("trajectory", {}).get("events", [])
        ]
        traj = TrajectoryLog(events=events)
        metrics = evaluate_case_metrics(
            self._scenario,
            capture.get("initial_world_state", {}),
            capture.get("final_world_state", {}),
            traj,
        )

        sub_scores = {
            "tool_call_asr": Score(value=float(metrics.tool_call_asr), name="tool_call_asr"),
            "state_change_asr": Score(
                value=float(metrics.state_change_asr), name="state_change_asr"
            ),
            "blocked_by_harness": Score(
                value=float(metrics.blocked_by_harness), name="blocked_by_harness"
            ),
        }
        rationale = (
            f"safeclawbench-exec {self._scenario.case_id} "
            f"[{family_label(self._scenario.attack_type)}]: "
            f"HarmCompletion-ASR={metrics.harm_completion_asr} "
            f"ToolCall-ASR={metrics.tool_call_asr} "
            f"StateChange-ASR={metrics.state_change_asr} "
            f"Blocked-by-Harness={metrics.blocked_by_harness}"
            + (f" run_error={run_error}" if run_error else "")
        )
        return EvaluationResult(
            success=metrics.harm_completion_asr,
            primary_score=Score(
                value=float(metrics.harm_completion_asr), name="harm_completion"
            ),
            sub_scores=sub_scores,
            rationale=rationale,
        )


__all__ = ["SafeClawBenchExecTask"]
