from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class ActionType(Enum):
    """
    Enumeration of control-flow decisions in the engine.
    """
    CONTINUE = auto()       # proceed to the next turn (can optionally specify an attempt)
    RETRY = auto()          # retry the current turn
    END_SUCCESS = auto()    # terminate with attach success
    END_FAILURE = auto()    # terminate with attack failure
    JUMP_TO = auto()        # jump to a previous turn and restart from there


@dataclass(frozen=True, slots=True)
class NextAction:
    """
    Control-flow signal (immutable).

    Attributes:
    - type: The type of action.
    - payload: Required only for JUMP_TO (payload = turn).
    """
    type: ActionType
    payload: Any | None = None

    def __post_init__(self):
        if self.type == ActionType.JUMP_TO and self.payload is None:
            raise ValueError("JUMP_TO must carry a payload")
        if self.type in [ActionType.RETRY, ActionType.END_FAILURE, ActionType.END_SUCCESS] and self.payload is not None:
            raise ValueError(f"{self.type} cannot carry a payload")

    def __str__(self):
        return f"{self.type.name}" if self.payload is None else f"{self.type.name} {self.payload}"

    def __repr__(self) -> str:
        """
        Return a concise representation for debugging
        """
        if self.payload is None:
            return f"NextAction(type={self.type.name})"
        return f"NextAction(type={self.type.name}, payload={self.payload})"
