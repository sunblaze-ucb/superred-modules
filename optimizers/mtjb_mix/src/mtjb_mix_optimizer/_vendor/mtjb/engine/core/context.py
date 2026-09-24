from dataclasses import dataclass, field

from client.unified_llm_client import Conversation

from .outcome import AttackOutcome
from .turn import AttackAttempt, AttackTurn


@dataclass(slots=True)
class AttackContext:
    # metadata (shouldn't be modified)
    harmful_behavior_id: str
    harmful_behavior: str
    max_turns: int
    max_epochs: int

    # data generated during the attack
    history: list[AttackTurn]
    full_history: list[AttackAttempt]
    current_turn: int
    current_epoch: int
    conversation: Conversation

    # final outcome of the attack
    attack_outcome: AttackOutcome

    # any custom data (a convenient way to pass data between components)
    custom: dict = field(default_factory=dict)

    def get_turn(self, num: int) -> AttackTurn:
        return self.history[num - 1]
    
    def get_current_turn(self) -> AttackTurn:
        return self.history[self.current_turn - 1]

    def get_current_attempt(self) -> AttackAttempt:
        return self.full_history[-1]

    def __repr__(self) -> str:
        """
        Return a concise representation for debugging
        """
        history_str = "\n\n".join(
            "    " + repr(turn).replace("\n", "\n    ")
            for turn in self.history
        )
        full_history_str = "\n\n".join(
            "    " + repr(turn).replace("\n", "\n    ")
            for turn in self.full_history
        )
        return (
            f"AttackContext(\n"
            f"  harmful_behavior_id={self.harmful_behavior_id},\n"
            f"  harmful_behavior={self.harmful_behavior!r},\n"
            f"  max_turns={self.max_turns},\n"
            f"  max_epochs={self.max_epochs},\n"
            f"  current_turn={self.current_turn},\n"
            f"  current_epoch={self.current_epoch},\n"
            f"  attack_outcome={self.attack_outcome},\n"
            f"\n"
            f"  history=[\n"
            f"{history_str}\n"
            f"  ]\n"
            f"\n"
            f"  full_history=[\n"
            f"{full_history_str}\n"
            f"  ]\n"
            f")"
        )
