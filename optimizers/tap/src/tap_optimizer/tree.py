"""TapNode and TapTree: tree data model for the TAP optimizer."""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field


@dataclass
class TapNode:
    """A single node in the TAP search tree."""

    depth: int
    parent_id: str | None
    conversation_history: list[dict[str, str]]
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str | None = None
    target_response: str | None = None
    score: float = 0.0
    is_on_topic: bool = True
    pruned: bool = False


class TapTree:
    """Manages the TAP search tree."""

    def __init__(self) -> None:
        self._nodes: list[TapNode] = []

    # ── public API ─────────────────────────────────────────────────────

    def create_root_nodes(self, width: int) -> list[TapNode]:
        """Create *width* root nodes with empty conversation history."""
        roots: list[TapNode] = []
        for _ in range(width):
            node = TapNode(depth=0, parent_id=None, conversation_history=[])
            self._nodes.append(node)
            roots.append(node)
        return roots

    def branch(self, parent: TapNode, branching_factor: int) -> list[TapNode]:
        """Create *branching_factor* children from *parent*.

        Each child gets a deep copy of the parent's conversation_history so
        that branches evolve independently.
        """
        children: list[TapNode] = []
        for _ in range(branching_factor):
            child = TapNode(
                depth=parent.depth + 1,
                parent_id=parent.node_id,
                conversation_history=copy.deepcopy(parent.conversation_history),
            )
            self._nodes.append(child)
            children.append(child)
        return children

    def get_leaves(self) -> list[TapNode]:
        """Return all non-pruned leaf nodes.

        A leaf is a non-pruned node that has no non-pruned children.
        """
        # Collect parent_ids of all non-pruned nodes
        parent_ids_with_children: set[str | None] = set()
        for node in self._nodes:
            if not node.pruned and node.parent_id is not None:
                parent_ids_with_children.add(node.parent_id)

        return [
            node
            for node in self._nodes
            if not node.pruned and node.node_id not in parent_ids_with_children
        ]

    def prune_off_topic(self) -> None:
        """Mark off-topic leaf nodes as pruned."""
        for leaf in self.get_leaves():
            if not leaf.is_on_topic:
                leaf.pruned = True

    def prune_to_width(self, width: int) -> None:
        """Keep only the top-*width* scoring leaves; prune the rest."""
        leaves = self.get_leaves()
        if len(leaves) <= width:
            return
        # Sort descending by score
        leaves.sort(key=lambda n: n.score, reverse=True)
        for leaf in leaves[width:]:
            leaf.pruned = True

    def best_node(self) -> TapNode | None:
        """Return the highest-scoring non-pruned node that has a prompt set.

        Searches the entire tree, not just leaves.  Returns ``None`` when no
        eligible node exists.
        """
        candidates = [
            node
            for node in self._nodes
            if not node.pruned and node.prompt is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda n: n.score)
