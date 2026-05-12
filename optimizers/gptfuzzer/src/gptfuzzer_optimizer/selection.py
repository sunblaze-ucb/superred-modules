"""GPTFuzzer seed selection policies."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from gptfuzzer_optimizer.core import PromptNode


class RoundRobinSelectPolicy:
    def __init__(self) -> None:
        self.index = 0

    def select(self, prompt_nodes: Sequence[PromptNode]) -> PromptNode:
        seed = prompt_nodes[self.index]
        seed.visited_num += 1
        return seed

    def update(
        self,
        updated_nodes: Sequence[PromptNode],
        *,
        len_questions: int,
        prompt_nodes: Sequence[PromptNode] | None = None,
    ) -> None:
        all_nodes = prompt_nodes if prompt_nodes is not None else updated_nodes
        self.index = (self.index - 1 + len(all_nodes)) % len(all_nodes)


class RandomSelectPolicy:
    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select(self, prompt_nodes: Sequence[PromptNode]) -> PromptNode:
        seed = self._rng.choice(list(prompt_nodes))
        seed.visited_num += 1
        return seed

    def update(
        self,
        updated_nodes: Sequence[PromptNode],
        *,
        len_questions: int,
        prompt_nodes: Sequence[PromptNode] | None = None,
    ) -> None:
        return None


class UCBSelectPolicy:
    def __init__(self, *, explore_coeff: float = 1.0) -> None:
        self.step = 0
        self.last_choice_index: int | None = None
        self.explore_coeff = explore_coeff
        self.rewards: list[float] = []

    def _extend(self, prompt_nodes: Sequence[PromptNode]) -> None:
        if len(prompt_nodes) > len(self.rewards):
            self.rewards.extend([0.0 for _ in range(len(prompt_nodes) - len(self.rewards))])

    def select(self, prompt_nodes: Sequence[PromptNode]) -> PromptNode:
        self._extend(prompt_nodes)
        self.step += 1
        best_idx = 0
        best_score = -math.inf
        for i, prompt_node in enumerate(prompt_nodes):
            smooth_visited_num = prompt_node.visited_num + 1
            score = self.rewards[i] / smooth_visited_num
            if self.step > 1:
                score += self.explore_coeff * math.sqrt(2 * math.log(self.step) / smooth_visited_num)
            if score > best_score:
                best_idx = i
                best_score = score
        self.last_choice_index = best_idx
        prompt_nodes[best_idx].visited_num += 1
        return prompt_nodes[best_idx]

    def update(
        self,
        updated_nodes: Sequence[PromptNode],
        *,
        len_questions: int,
        prompt_nodes: Sequence[PromptNode] | None = None,
    ) -> None:
        if self.last_choice_index is None:
            return
        all_nodes = prompt_nodes if prompt_nodes is not None else updated_nodes
        self._extend(all_nodes)
        succ_num = sum(prompt_node.num_jailbreak for prompt_node in updated_nodes)
        self.rewards[self.last_choice_index] += succ_num / max(1, len_questions)


class MCTSExploreSelectPolicy:
    def __init__(
        self,
        *,
        ratio: float = 0.5,
        alpha: float = 0.1,
        beta: float = 0.2,
        seed: int | None = None,
    ) -> None:
        self.step = 0
        self.mctc_select_path: list[PromptNode] = []
        self.last_choice_index: int | None = None
        self.rewards: list[float] = []
        self.ratio = ratio
        self.alpha = alpha
        self.beta = beta
        self._rng = random.Random(seed)

    def _extend(self, prompt_nodes: Sequence[PromptNode]) -> None:
        max_index = max((node.index for node in prompt_nodes if node.index is not None), default=-1)
        if max_index + 1 > len(self.rewards):
            self.rewards.extend([0.0 for _ in range(max_index + 1 - len(self.rewards))])

    def _score(self, node: PromptNode) -> float:
        assert node.index is not None
        return self.rewards[node.index] / (node.visited_num + 1) + self.ratio * math.sqrt(
            2 * math.log(max(self.step, 1)) / (node.visited_num + 0.01)
        )

    def select(
        self,
        prompt_nodes: Sequence[PromptNode],
        *,
        initial_nodes: Sequence[PromptNode] | None = None,
    ) -> PromptNode:
        self._extend(prompt_nodes)
        roots = list(initial_nodes if initial_nodes is not None else prompt_nodes)
        if not roots:
            raise ValueError("prompt_nodes must not be empty")
        self.step += 1
        self.mctc_select_path.clear()
        cur = max(roots, key=self._score)
        self.mctc_select_path.append(cur)
        while cur.child:
            if self._rng.random() < self.alpha:
                break
            cur = max(cur.child, key=self._score)
            self.mctc_select_path.append(cur)
        for node in self.mctc_select_path:
            node.visited_num += 1
        assert cur.index is not None
        self.last_choice_index = cur.index
        return cur

    def update(
        self,
        updated_nodes: Sequence[PromptNode],
        *,
        len_questions: int,
        prompt_nodes: Sequence[PromptNode] | None = None,
    ) -> None:
        if self.last_choice_index is None:
            return
        all_nodes = prompt_nodes if prompt_nodes is not None else updated_nodes
        self._extend(all_nodes)
        succ_num = sum(prompt_node.num_jailbreak for prompt_node in updated_nodes)
        reward = succ_num / max(1, len_questions * len(updated_nodes))
        last_choice_node = next(
            (node for node in all_nodes if node.index == self.last_choice_index),
            None,
        )
        level = 0 if last_choice_node is None else last_choice_node.level
        for node in reversed(self.mctc_select_path):
            assert node.index is not None
            self.rewards[node.index] += reward * max(self.beta, (1 - 0.1 * level))
