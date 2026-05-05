"""TapNode and TapTree: tree data model for the TAP optimizer."""

from __future__ import annotations

import copy
import random
import uuid
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TapNode:
    """A single node in the TAP search tree."""

    depth: int
    parent_id: str | None
    conversation_history: list[dict[str, str]]
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str | None = None
    system_prompt: str | None = None
    improvement: str | None = None
    target_response: str | None = None
    score: float = 0.0
    is_on_topic: bool = True
    pruned: bool = False


class TapTree:
    """Manages the TAP search tree."""

    def __init__(self, *, rng: random.Random | None = None) -> None:
        self._nodes: list[TapNode] = []
        self._rng = rng if rng is not None else random.Random()

    def create_root_nodes(self, width: int) -> list[TapNode]:
        """Create *width* root nodes with empty conversation history."""
        roots: list[TapNode] = []
        for _ in range(width):
            node = TapNode(depth=0, parent_id=None, conversation_history=[])
            self._nodes.append(node)
            roots.append(node)
        return roots

    def branch(self, parent: TapNode, branching_factor: int) -> list[TapNode]:
        """Create *branching_factor* children from *parent*."""
        children: list[TapNode] = []
        for _ in range(branching_factor):
            child = TapNode(
                depth=parent.depth + 1,
                parent_id=parent.node_id,
                conversation_history=copy.deepcopy(parent.conversation_history),
                target_response=parent.target_response,
                score=parent.score,
            )
            self._nodes.append(child)
            children.append(child)
        return children

    def get_leaves(self) -> list[TapNode]:
        """Return all non-pruned leaf nodes."""
        parent_ids_with_children: set[str | None] = set()
        for node in self._nodes:
            if not node.pruned and node.parent_id is not None:
                parent_ids_with_children.add(node.parent_id)

        return [
            node
            for node in self._nodes
            if not node.pruned and node.node_id not in parent_ids_with_children
        ]

    def prune_off_topic(self, *, width: int) -> None:
        """Prune off-topic leaves with TAP's minimum-retention fallback."""
        leaves = self.get_leaves()
        self._prune_by_score(leaves, width=width, score=lambda node: float(node.is_on_topic))

    def prune_to_width(self, width: int) -> None:
        """Keep top scoring leaves with stochastic tie-breaking and fallback."""
        leaves = self.get_leaves()
        self._prune_by_score(leaves, width=width, score=lambda node: node.score)

    def best_node(self) -> TapNode | None:
        """Return the highest-scoring non-pruned node that has a prompt set."""
        candidates = [
            node for node in self._nodes if not node.pruned and node.prompt is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda n: n.score)

    def _prune_by_score(
        self,
        leaves: list[TapNode],
        *,
        width: int,
        score: Callable[[TapNode], float],
    ) -> None:
        if not leaves or len(leaves) <= width and any(score(node) > 0 for node in leaves):
            return

        shuffled = list(leaves)
        self._rng.shuffle(shuffled)
        shuffled.sort(key=score, reverse=True)

        positive = [node for node in shuffled if score(node) > 0]
        if positive:
            keep_count = min(width, len(positive))
            keep_ids = {node.node_id for node in positive[:keep_count]}
        else:
            fallback_count = min(len(shuffled), 2)
            keep_ids = {node.node_id for node in shuffled[:fallback_count]}

        for leaf in leaves:
            if leaf.node_id not in keep_ids:
                leaf.pruned = True
