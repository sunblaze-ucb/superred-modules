from dataclasses import dataclass, field
from typing import Any

from client.unified_llm_client import Conversation

from .action import NextAction
from .evaluation import TurnEvaluation


@dataclass(slots=True)
class AttackAttempt:
    """
    An AttackAttempt records all data generated during an attempt. The workflow is:
    PromptGenerator -> DefendedModel -> OutcomeEvaluator -> FlowController
    """
    # each attempt can be uniquely identified by its epoch number
    epoch: int

    # interaction with target model
    system_prompt: str | None = None
    prompt: str | None = None
    response: str | None = None

    # evaluation
    evaluation: TurnEvaluation | None = None

    # control flow
    next_action: NextAction | None = None

    # defense-specific trace and decisions
    defense_info: dict[str, Any] = field(default_factory=dict)

    # private: util for RETRY and JUMP_TO logic
    _conv_after_response: Conversation | None = None

    def to_dict(self) -> dict:
        """
        Returns a dict representation. The format below will be used by the engine
        to generate experiment logs.
        """
        d = {
            "epoch": self.epoch,
            "system_prompt": self.system_prompt,
            "prompt": self.prompt,
            "response": self.response,
            "evaluation": None if self.evaluation is None else self.evaluation.to_dict(),
            "action": None if self.next_action is None else str(self.next_action),
        }
        if self.defense_info:
            d["defense_info"] = self.defense_info
        return d


    def __repr__(self) -> str:
        """
        Return a concise representation for debugging
        """
        def _truncate(text: str | None, max_len: int = 80) -> str | None:
            if text is None:
                return None
            return text if len(text) <= max_len else text[:max_len] + "...(truncated)"
        return (
            f"Attempt:\n"
            f"  epoch={self.epoch}:\n"
            f"  system_prompt={_truncate(self.system_prompt)!r}\n"
            f"  prompt={_truncate(self.prompt)!r}\n"
            f"  response={_truncate(self.response)!r}\n"
            f"  evaluation={self.evaluation}\n"
            f"  action={self.next_action}"
        )

@dataclass(slots=True)
class AttackTurn:
    turn: int
    all_attempts: list[AttackAttempt] = field(default_factory=list)
    attempt_in_effect: AttackAttempt | None = None

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "attempt_in_effect": self.attempt_in_effect.to_dict()
        }

    def __repr__(self) -> str:
        """
        Return a concise representation for debugging
        """
        return ""
