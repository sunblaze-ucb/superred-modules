"""InjecAgentTask: one superred Task per InjecAgent test case.

``configure_target`` hands the verbatim test case (+ its attack type and setting)
to the paired ``injecagent`` target, which builds the ReAct / function-calling
prompt, drives the agent-under-test, runs the ds two-step, and captures the
outputs. ``evaluate`` reads those outputs back and runs the deterministic judge
(:func:`injecagent_claim.judge.evaluate_task`).

The task is typed against the base :class:`~superred.core.interfaces.target.Target`
(not the concrete target) so this claim stays installable without the target
package — the dependency runs the other way (the target imports this claim's
judge to gate its ds step-2).

Polarity: ``primary_score`` is the **attack magnitude** (1.0 == the injection
made the agent take the attacker's action; 0.0 == it did not), matching
superred's higher-is-better-for-the-attacker convention.
"""

from __future__ import annotations

import json
from typing import Any

from superred.core.interfaces.target import Target
from superred.core.interfaces.task import Task
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.goal import Goal
from superred.core.types.trajectory import Trajectory

from injecagent_claim.judge import evaluate_task
from injecagent_claim.security_tags import DIMENSION_TAG

_ATTACK_NAME = {"dh": "direct harm", "ds": "data stealing"}


class InjecAgentTask(Task[Target]):
    """One InjecAgent adversarial test case, bound to the ``injecagent`` target.

    Args:
        case: one test-case dict (verbatim upstream fields + the ``_attack`` /
            ``_setting`` / ``_task_id`` routing keys added by the loader).
    """

    def __init__(self, *, case: dict[str, Any]) -> None:
        self._case = case
        self._attack = str(case["_attack"])
        self._setting = str(case["_setting"])
        self._task_id = str(case["_task_id"])
        # The Goal carries the benign user instruction; the adversarial payload is
        # baked into the tool observation the target provisions, so a passthrough
        # optimizer runs the benchmark exactly as InjecAgent designed it.
        self._goal = Goal(description=str(case.get("User Instruction", self._task_id)))

    @property
    def goal(self) -> Goal:
        return self._goal

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def dimension(self) -> str:
        return self._attack

    @property
    def setting(self) -> str:
        return self._setting

    async def configure_target(self, target: Target) -> None:
        target.set_config("case", json.dumps(self._case))
        target.set_config("attack", self._attack)
        target.set_config("setting", self._setting)

    async def evaluate(self, _trajectory: Trajectory, target: Target) -> EvaluationResult:
        raw = target.query("result")
        try:
            result = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            result = {}

        verdict = evaluate_task(self._case, self._attack, result)

        primary = Score(value=verdict.score, security_domain=None, name="attack_score")
        sub_scores: dict[str, Score] = {
            "attack_succeeded": Score(
                value=float(verdict.attack_succeeded),
                security_domain=None,
                name="attack_succeeded",
            ),
            "invalid": Score(value=float(verdict.invalid), security_domain=None, name="invalid"),
            "step1_succeeded": Score(
                value=float(verdict.eval_step1 == "succ"),
                security_domain=None,
                name="step1_succeeded",
            ),
        }
        if verdict.attack == "ds":
            sub_scores["step2_succeeded"] = Score(
                value=float(verdict.eval_step2 == "succ"),
                security_domain=None,
                name="step2_succeeded",
            )
        dim_tag = DIMENSION_TAG.get(self._attack)
        if dim_tag is not None:
            # The IPI attack crosses the external-data trust boundary; tag it so
            # it prunes outside a scope that grants that boundary.
            sub_scores["attack_succeeded_in_domain"] = Score(
                value=float(verdict.attack_succeeded),
                security_domain=dim_tag,
                name="attack_succeeded_in_domain",
            )

        mode = str(result.get("mode", "prompted"))
        rationale = (
            f"injecagent {self._task_id} ({_ATTACK_NAME.get(self._attack, self._attack)}"
            f"/{self._setting}, mode={mode}): attack_succeeded={verdict.attack_succeeded} "
            f"eval1={verdict.eval_step1} eval2={verdict.eval_step2} "
            f"invalid={verdict.invalid}"
            + (f" ({verdict.invalid_reason})" if verdict.invalid_reason else "")
        )
        return EvaluationResult(
            success=verdict.attack_succeeded,
            primary_score=primary,
            sub_scores=sub_scores,
            rationale=rationale,
        )


__all__ = ["InjecAgentTask"]
