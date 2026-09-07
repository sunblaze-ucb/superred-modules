"""One CyberSecEval prompt-injection case as a superred Task.

Bound to ChatbotTarget. Unlike the harm-elicitation claims, this one plants a
defended **system prompt** (``test_case_prompt``): the attack is whether the
``user_input`` injection overrides it, judged by the case's ``judge_question``.
"""

from __future__ import annotations

from chatbot_target import ChatbotTarget  # type: ignore[import-untyped]
from superred.core.interfaces.task import Task
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.goal import Goal
from superred.core.types.trajectory import Trajectory

from cyberseceval_pi_claim.judge import PromptInjectionJudge
from cyberseceval_pi_claim.loader import InjectionCase

_SYSTEM_PROMPT_CONTROLLABLE = "system_prompt"


class PromptInjectionTask(Task[ChatbotTarget]):
    """One injection case: defended system prompt + injection, judged Yes/No."""

    def __init__(self, *, case: InjectionCase, judge: PromptInjectionJudge) -> None:
        self._case = case
        self._judge = judge
        # The injection (user_input) is the adversarial move to attempt.
        self._goal = Goal(description=case.user_input)

    @property
    def prompt_id(self) -> int:
        return self._case.prompt_id

    @property
    def injection_type(self) -> str:
        return self._case.injection_type

    @property
    def risk_category(self) -> str:
        return self._case.risk_category

    @property
    def goal(self) -> Goal:
        return self._goal

    @property
    def judge(self) -> PromptInjectionJudge:
        return self._judge

    async def configure_target(self, target: ChatbotTarget) -> None:
        # Plant the defended setup as the system prompt, exactly as upstream
        # sends test_case_prompt via query_with_system_prompt.
        target.set_config(_SYSTEM_PROMPT_CONTROLLABLE, self._case.system_prompt)

    async def evaluate(
        self, trajectory: Trajectory, target: ChatbotTarget
    ) -> EvaluationResult:
        response = target.query("last_response")
        succeeded, raw = await self._judge.judge(
            llm_response=response, question=self._case.judge_question
        )
        primary = Score(
            value=1.0 if succeeded else 0.0,
            security_domain=None,
            name="cyberseceval_pi_asr",
        )
        return EvaluationResult(
            success=succeeded,
            primary_score=primary,
            sub_scores={
                f"type_{self.injection_type}": Score(
                    value=primary.value, security_domain=None,
                    name=f"type_{self.injection_type}",
                ),
                f"risk_{self.risk_category}": Score(
                    value=primary.value, security_domain=None,
                    name=f"risk_{self.risk_category}",
                ),
            },
            rationale=f"judge_raw={raw[:200]!r}",
        )
