from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TurnEvaluation:
    """
    Evaluation for a single turn (immutable).

    Attributes:
    - score: Standard metric
    - similarity: Standard metric
    - metrics: Additional custom metrics may be stored here
    """

    score: float | None = None
    similarity: float | None = None
    reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)  # custom metrics if needed

    def to_dict(self) -> dict:
        data = {}
        if self.score is not None:
            data["score"] = self.score
        if self.similarity is not None:
            data["similarity"] = self.similarity
        if self.reason is not None:
            data["reason"] = self.reason
        # merge custom metrics (metrics override standard keys if duplicated)
        if self.metrics:
            data.update(self.metrics)
        return data

    def __repr__(self) -> str:
        """
        Return a concise representation for debugging
        """
        parts = []
        if self.score is not None:
            parts.append(f"score={self.score}")
        if self.similarity is not None:
            parts.append(f"similarity={self.similarity}")
        if self.metrics:
            parts.append(f"metrics={self.metrics}")
        return ", ".join(parts)
        
