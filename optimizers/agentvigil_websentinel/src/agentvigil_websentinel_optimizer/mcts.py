"""Small MCTS/UCT selector matching the official AgentVigil fuzzer shape."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from agentvigil_websentinel_optimizer.seeds import Seed


@dataclass(eq=False)
class Node:
    seed: Seed
    parents: list["Node"] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0

    @property
    def average_reward(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def update(self, reward: float) -> None:
        self.visits += 1
        self.total_reward += reward


class MonteCarloTree:
    def __init__(
        self,
        seeds: list[Seed],
        *,
        exploration_factor: float = 1.41,
        random_seed: int | None = None,
    ) -> None:
        self.nodes: list[Node] = [Node(seed=seed) for seed in seeds]
        self.exploration_factor = exploration_factor
        self._random = random.Random(random_seed)

    def select_node(self, n: int = 1) -> list[Node]:
        if n < 1:
            raise ValueError("n must be at least 1")
        if not self.nodes:
            raise ValueError("cannot select from an empty tree")
        total_visits = sum(node.visits for node in self.nodes)
        scored = [
            (self._uct(node, total_visits), self._random.random(), node)
            for node in self.nodes
        ]
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [node for _score, _tie, node in scored[: min(n, len(scored))]]

    def add_child(self, seed: Seed, parents: list[Node]) -> Node:
        node = Node(seed=seed, parents=parents)
        self.nodes.append(node)
        return node

    def backpropagate(self, node: Node, reward: float) -> None:
        node.update(reward)
        for parent in node.parents:
            self.backpropagate(parent, reward)

    def _uct(self, node: Node, total_visits: int) -> float:
        return node.average_reward + self.exploration_factor * math.sqrt(
            math.log(total_visits + 1) / (node.visits + 1e-5)
        )


__all__ = ["MonteCarloTree", "Node"]
