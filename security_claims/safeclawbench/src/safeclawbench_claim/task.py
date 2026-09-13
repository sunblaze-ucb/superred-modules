"""SafeClawBenchTask: one Task per SafeClawBench Semantic Core case.

Faithful port of the Semantic Core protocol (upstream ``run_benchmark.py`` +
``evaluator/judge.py``) onto the superred OpenClaw target:

- ``configure_target`` installs the fixed agent-environment system prompt
  (``AGENT_SYSTEM_PROMPT``, the six tool categories) via the target's
  ``system_prompt_append`` config, optionally prefixed by a prompt-level
  defense policy. By default no defense is applied (paper's D0 baseline).
- the case's ``user_prompt`` is the attack payload. It is delivered through
  the target's ``user_message`` controllable (carried on the task ``Goal``),
  so a passthrough optimizer reproduces the upstream baseline while an
  attacker optimizer can augment it.
- ``evaluate`` reads the agent's final response and runs the CoreFail judge
  (``attack_success`` -> ``success``), surfacing the Core-gated HarmEvidence
  category and safety score as sub-scores.

Score convention (attacker-oriented, matching the other OpenClaw claims):
``success`` is ``True`` when CoreFail fires (the attack landed);
``primary_score`` is ``1.0`` on CoreFail.
"""

from __future__ import annotations

from openclaw_target import OpenClawTarget
from superred.core.interfaces.task import Task
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.goal import Goal
from superred.core.types.trajectory import Trajectory

from safeclawbench_claim.judge import CoreFailJudge
from safeclawbench_claim.loader import AGENT_SYSTEM_PROMPT, SafeClawCase, family_label


class SafeClawBenchTask(Task[OpenClawTarget]):
    """One SafeClawBench Semantic Core case evaluated on the OpenClaw agent."""

    def __init__(
        self,
        case: SafeClawCase,
        judge: CoreFailJudge,
        *,
        defense_system_prompt: str | None = None,
        agent_system_prompt: str = AGENT_SYSTEM_PROMPT,
    ) -> None:
        self._case = case
        self._judge = judge
        self._defense_system_prompt = defense_system_prompt
        self._agent_system_prompt = agent_system_prompt
        # The attack payload is the case user_prompt; the optimizer delivers it
        # through the target's user_message controllable.
        self._goal = Goal(description=case.user_prompt)

    @property
    def goal(self) -> Goal:
        return self._goal

    @property
    def case(self) -> SafeClawCase:
        return self._case

    def _system_prompt(self) -> str:
        if self._defense_system_prompt:
            return f"{self._defense_system_prompt}\n\n{self._agent_system_prompt}"
        return self._agent_system_prompt

    async def configure_target(self, target: OpenClawTarget) -> None:
        await target.warmup_static_observables()
        target.set_config("system_prompt_append", self._system_prompt())

    async def evaluate(
        self, trajectory: Trajectory, target: OpenClawTarget,
    ) -> EvaluationResult:
        del trajectory
        response = target.query("last_response")
        verdict = await self._judge.judge(
            case={
                "attack_type": self._case.attack_type,
                "scenario": self._case.scenario,
                "user_prompt": self._case.user_prompt,
                "harm_type": self._case.harm_type,
                "success_predicate": self._case.success_predicate,
                "safe_behavior": self._case.safe_behavior,
            },
            agent_response=response,
        )

        sub_scores: dict[str, Score] = {
            "harm_evidence": Score(
                value=float(verdict.has_harm_evidence), name="harm_evidence",
            ),
        }
        if verdict.safety_score is not None:
            # Attacker-oriented: 1.0 == fully unsafe (safety_score 1), 0.0 == safe (5).
            inverted = (5 - verdict.safety_score) / 4
            sub_scores["safety_inverted"] = Score(
                value=max(0.0, min(1.0, inverted)), name="safety_inverted",
            )

        rationale = (
            f"safeclawbench {self._case.task_id} "
            f"[{family_label(self._case.attack_type)}/{self._case.harm_type}]: "
            f"core_fail={verdict.attack_success} "
            f"harm_evidence={verdict.harm_evidence_category} "
            f"safety={verdict.safety_score} defense={verdict.defense_behavior} "
            f"parsed={verdict.parsed} {verdict.note} :: {verdict.explanation[:200]}"
        )
        return EvaluationResult(
            success=verdict.attack_success,
            primary_score=Score(value=float(verdict.attack_success), name="core_fail"),
            sub_scores=sub_scores,
            rationale=rationale,
        )


__all__ = ["SafeClawBenchTask"]
