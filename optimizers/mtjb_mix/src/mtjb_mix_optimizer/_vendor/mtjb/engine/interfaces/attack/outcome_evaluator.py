from abc import ABC, abstractmethod

from ...core import AttackContext, TurnEvaluation


class OutcomeEvaluator(ABC):
    
    @abstractmethod
    def evaluate(self, context: AttackContext) -> TurnEvaluation:
        pass
