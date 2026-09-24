from dataclasses import dataclass
from enum import Enum, auto


class OutcomeCode(Enum):
    """
    Enumeration of final attack outcome.
    """
    SUCCESS_BY_SELF_EVAL = auto()
    FAILURE_BY_SELF_EVAL = auto()
    SUCCESS_BY_INDP_EVAL = auto()   # for independent judge
    FAILURE_BY_INDP_EVAL = auto()   # for independent judge
    MAX_TURN_REACHED = auto()
    MAX_EPOCH_REACHED = auto()
    # MAX_TOKEN_REACHED

@dataclass(frozen=True, slots=True)
class AttackOutcome:
    successful: bool
    outcome_code: OutcomeCode

    def __repr__(self) -> str:
        """
        Return a concise representation for debugging
        """
        return (
            f"AttackOutcome(successful={self.successful}, "
            f"outcome_code={self.outcome_code.name})"
        )
