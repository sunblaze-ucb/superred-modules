"""Core data structures for the GPTFuzzer optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PromptNode:
    """One jailbreak template in the GPTFuzzer seed tree."""

    prompt: str
    response: list[str] | None = None
    results: list[int] = field(default_factory=list)
    parent: "PromptNode | None" = None
    mutator_name: str | None = None
    index: int | None = None
    visited_num: int = 0
    child: list["PromptNode"] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.parent is not None and self.index is not None:
            self.parent.child.append(self)

    def attach_to_parent(self) -> None:
        if self.parent is not None and self not in self.parent.child:
            self.parent.child.append(self)

    @property
    def level(self) -> int:
        return 0 if self.parent is None else self.parent.level + 1

    @property
    def num_jailbreak(self) -> int:
        return sum(self.results)

    @property
    def num_reject(self) -> int:
        return len(self.results) - sum(self.results)

    @property
    def num_query(self) -> int:
        return len(self.results)
