from abc import ABC, abstractmethod

from ...core import AttackContext, NextAction


class FlowController(ABC):
    
    @abstractmethod
    def next_action(self, context: AttackContext) -> NextAction:
        pass
