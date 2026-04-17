"""Seed selection strategies ported from GPTFuzzer.

Implements MCTSExploreSelectPolicy (the primary strategy), UCB,
Random, and RoundRobin selection for choosing which seed template
to mutate next.

Port of: https://github.com/sherdencooper/GPTFuzz (MIT license)
"""

from __future__ import annotations

import math
import random


class SeedNode:
    """A node in the seed pool, tracking statistics for selection.

    Mirrors GPTFuzzer's PromptNode with parent/child tree relationships
    needed for MCTS-style selection.

    Attributes:
        prompt: The jailbreak template text (contains [INSERT PROMPT HERE]).
        index: Position in the seed pool (set via property to auto-register with parent).
        visited_num: Number of times this seed has been selected.
        results: Per-run results (1=jailbreak, 0=reject) for computing statistics.
        parent: Reference to the parent SeedNode (None for initial seeds).
        child: List of child SeedNode objects.
        level: Depth in the tree (0 for initial seeds).
    """

    def __init__(
        self,
        prompt: str,
        parent: SeedNode | None = None,
        *,
        results: list[int] | None = None,
    ) -> None:
        self.prompt: str = prompt
        self.results: list[int] = results if results is not None else []
        self.visited_num: int = 0

        self.parent: SeedNode | None = parent
        self.child: list[SeedNode] = []
        self.level: int = 0 if parent is None else parent.level + 1

        self._index: int | None = None

    @property
    def index(self) -> int | None:
        return self._index

    @index.setter
    def index(self, index: int) -> None:
        self._index = index
        if self.parent is not None:
            self.parent.child.append(self)

    @property
    def num_jailbreak(self) -> int:
        return sum(self.results)

    @property
    def num_reject(self) -> int:
        return len(self.results) - sum(self.results)

    @property
    def num_query(self) -> int:
        return len(self.results)


class SelectPolicy:
    """Base class for seed selection policies."""

    def __init__(self) -> None:
        self.seed_pool: list[SeedNode] = []
        self.initial_prompts_nodes: list[SeedNode] = []

    def set_seed_pool(
        self,
        seed_pool: list[SeedNode],
        initial_prompts_nodes: list[SeedNode] | None = None,
    ) -> None:
        self.seed_pool = seed_pool
        self.initial_prompts_nodes = (
            initial_prompts_nodes
            if initial_prompts_nodes is not None
            else list(seed_pool)
        )

    def select(self) -> SeedNode:
        raise NotImplementedError

    def update(self, prompt_nodes: list[SeedNode]) -> None:
        pass


class MCTSExploreSelectPolicy(SelectPolicy):
    """MCTS-based exploration selection policy.

    Walks the seed tree from root to leaf using UCB-like scoring at
    each level, with a random chance of stopping early (alpha). Rewards
    are backpropagated along the selection path with a level-based
    penalty.

    This is the primary selection strategy used in GPTFuzzer.

    Ported from GPTFuzzer's MCTSExploreSelectPolicy.

    Args:
        ratio: Balance between exploration and exploitation in UCB formula.
        alpha: Probability of stopping tree traversal early at each level.
        beta: Minimal reward multiplier after level penalty.
    """

    def __init__(
        self,
        ratio: float = 0.5,
        alpha: float = 0.1,
        beta: float = 0.2,
    ) -> None:
        super().__init__()

        self.step: int = 0
        self.mctc_select_path: list[SeedNode] = []
        self.last_choice_index: int | None = None
        self.rewards: list[float] = []
        self.ratio = ratio
        self.alpha = alpha
        self.beta = beta

    def _ucb_score(self, node: SeedNode) -> float:
        """Compute UCB score for a node."""
        assert node.index is not None
        return self.rewards[node.index] / (
            node.visited_num + 1
        ) + self.ratio * math.sqrt(2 * math.log(self.step) / (node.visited_num + 0.01))

    def select(self) -> SeedNode:
        self.step += 1
        if len(self.seed_pool) > len(self.rewards):
            self.rewards.extend(
                [0.0 for _ in range(len(self.seed_pool) - len(self.rewards))]
            )

        self.mctc_select_path.clear()

        # Start from the best initial/root node
        cur = max(self.initial_prompts_nodes, key=self._ucb_score)
        self.mctc_select_path.append(cur)

        # Walk down the tree
        while len(cur.child) > 0:
            if random.random() < self.alpha:
                break
            cur = max(cur.child, key=self._ucb_score)
            self.mctc_select_path.append(cur)

        # Increment visited_num for all nodes on the path
        for pn in self.mctc_select_path:
            pn.visited_num += 1

        self.last_choice_index = cur.index
        return cur

    def update(self, prompt_nodes: list[SeedNode]) -> None:
        succ_num = sum(pn.num_jailbreak for pn in prompt_nodes)

        assert self.last_choice_index is not None
        last_choice_node = self.seed_pool[self.last_choice_index]
        for prompt_node in reversed(self.mctc_select_path):
            reward = succ_num / (
                1 * len(prompt_nodes)
            )  # 1 question per run in superred
            assert prompt_node.index is not None
            self.rewards[prompt_node.index] += reward * max(
                self.beta, (1 - 0.1 * last_choice_node.level)
            )


class UCBSelectPolicy(SelectPolicy):
    """Upper Confidence Bound seed selection.

    Balances exploitation (picking seeds that have worked before)
    with exploration (trying seeds that haven't been tested much).

    Ported from GPTFuzzer's UCBSelectPolicy.

    Args:
        explore_coeff: Exploration coefficient for UCB formula.
    """

    def __init__(self, explore_coeff: float = 1.0) -> None:
        super().__init__()
        self.explore_coeff = explore_coeff
        self.step = 0
        self.rewards: list[float] = []
        self.last_choice_index: int | None = None

    def set_seed_pool(
        self,
        seed_pool: list[SeedNode],
        initial_prompts_nodes: list[SeedNode] | None = None,
    ) -> None:
        super().set_seed_pool(seed_pool, initial_prompts_nodes)
        self.rewards = [0.0 for _ in range(len(seed_pool))]

    def select(self) -> SeedNode:
        # Extend rewards if seed pool has grown (successful seeds added)
        if len(self.seed_pool) > len(self.rewards):
            self.rewards.extend(
                [0.0 for _ in range(len(self.seed_pool) - len(self.rewards))]
            )

        self.step += 1

        scores = []
        for i, seed_node in enumerate(self.seed_pool):
            smooth_visited_num = seed_node.visited_num + 1
            exploitation = self.rewards[i] / smooth_visited_num
            exploration = self.explore_coeff * math.sqrt(
                2 * math.log(self.step) / smooth_visited_num
            )
            scores.append(exploitation + exploration)

        # argmax
        self.last_choice_index = max(range(len(scores)), key=lambda i: scores[i])
        self.seed_pool[self.last_choice_index].visited_num += 1
        return self.seed_pool[self.last_choice_index]

    def update(self, prompt_nodes: list[SeedNode]) -> None:
        """Update reward for the selected seed based on evaluation results.

        In the original GPTFuzzer, reward is succ_num / len(questions).
        Here we use a single question per run, so reward is num_jailbreak / 1.
        """
        if self.last_choice_index is not None:
            succ_num = sum(pn.num_jailbreak for pn in prompt_nodes)
            self.rewards[self.last_choice_index] += succ_num / 1  # 1 question per run


class RandomSelectPolicy(SelectPolicy):
    """Random seed selection."""

    def select(self) -> SeedNode:
        seed = random.choice(self.seed_pool)
        seed.visited_num += 1
        return seed


class RoundRobinSelectPolicy(SelectPolicy):
    """Round-robin seed selection.

    Ported from GPTFuzzer's RoundRobinSelectPolicy.
    """

    def __init__(self) -> None:
        super().__init__()
        self._index: int = 0

    def select(self) -> SeedNode:
        seed = self.seed_pool[self._index]
        seed.visited_num += 1
        return seed

    def update(self, prompt_nodes: list[SeedNode]) -> None:
        # Original: self.index = (self.index - 1 + len(nodes)) % len(nodes)
        # This effectively moves backward by 1, which with the +1 from
        # new nodes being appended means it cycles through.
        self._index = (self._index - 1 + len(self.seed_pool)) % len(self.seed_pool)
