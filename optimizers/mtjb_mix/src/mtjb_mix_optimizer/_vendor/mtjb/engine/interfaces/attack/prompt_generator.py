from abc import ABC, abstractmethod

from ...core import AttackContext


class PromptGenerator(ABC):
    
    @abstractmethod
    def next_prompt(self, context: AttackContext) -> str:
        pass

    @abstractmethod
    def refine_prompt(self, context: AttackContext) -> str:
        pass

    @abstractmethod
    def system_prompt(self, context: AttackContext) -> str | None:
        return "You're a helpful assistant."
