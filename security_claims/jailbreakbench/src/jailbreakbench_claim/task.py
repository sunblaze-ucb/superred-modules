"""One JBB behaviour as a superred Task, bound to ChatbotTarget.

JailbreakBench is single-turn: the ``Goal`` is the adversarial objective, and
the judge scores the model's response to it. ``configure_target`` clears the
system prompt (JBB adds none); ``evaluate`` reads the last response and routes
it through the JailbreakBench judge.
"""

from __future__ import annotations

from chatbot_target import ChatbotTarget  # type: ignore[import-untyped]
from superred.core.interfaces.task import Task
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.goal import Goal
from superred.core.types.trajectory import Trajectory

from jailbreakbench_claim.judge import JailbreakBenchJudge

_SYSTEM_PROMPT_CONTROLLABLE = "system_prompt"


class JailbreakBenchTask(Task[ChatbotTarget]):
    """One JBB-Behaviors row, bound to ChatbotTarget.

    Args:
        row: A behaviour row (columns Index, Goal, Target, Behavior,
            Category, Source).
        judge: The shared JailbreakBench judge.
    """

    def __init__(self, *, row: dict[str, str], judge: JailbreakBenchJudge) -> None:
        self._row = row
        self._judge = judge
        self._goal = Goal(description=row["Goal"])

    @property
    def index(self) -> str:
        return self._row["Index"]

    @property
    def behavior(self) -> str:
        return self._row["Behavior"]

    @property
    def category(self) -> str:
        return self._row["Category"]

    @property
    def source(self) -> str:
        return self._row["Source"]

    @property
    def goal(self) -> Goal:
        return self._goal

    @property
    def judge(self) -> JailbreakBenchJudge:
        return self._judge

    async def configure_target(self, target: ChatbotTarget) -> None:
        # JBB's canonical setup adds no system prompt. ChatbotTarget suppresses
        # an empty system prompt, so this restores the default rather than
        # planting one.
        target.set_config(_SYSTEM_PROMPT_CONTROLLABLE, "")

    async def evaluate(
        self, trajectory: Trajectory, target: ChatbotTarget
    ) -> EvaluationResult:
        response = target.query("last_response")
        # A judge failure propagates: "the judge never answered" and "the judge
        # said safe" are different facts, and scoring the first as the second
        # makes a broken judge read as a robust target. The controller records
        # stop_reason="error" and the task is re-measured on resume.
        is_jailbroken, raw = await self._judge.judge(
            goal=self._row["Goal"], response=response
        )
        primary = Score(
            value=1.0 if is_jailbroken else 0.0,
            security_domain=None,
            name="jbb_asr",
        )
        return EvaluationResult(
            success=is_jailbroken,
            primary_score=primary,
            sub_scores={
                f"category_{self.category}": Score(
                    value=primary.value,
                    security_domain=None,
                    name=f"category_{self.category}",
                ),
            },
            rationale=f"judge_raw={raw[:200]!r}",
        )
