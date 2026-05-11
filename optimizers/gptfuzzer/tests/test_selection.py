"""Tests for GPTFuzzer seed selection policies."""

from gptfuzzer_optimizer.core import PromptNode
from gptfuzzer_optimizer.selection import MCTSExploreSelectPolicy, UCBSelectPolicy


def test_ucb_selects_each_seed_with_official_smoothing() -> None:
    nodes = [PromptNode(prompt="a", index=0), PromptNode(prompt="b", index=1)]
    policy = UCBSelectPolicy(explore_coeff=1.0)

    first = policy.select(nodes)
    assert first is nodes[0]
    policy.update([PromptNode(prompt="child", results=[1])], len_questions=1, prompt_nodes=nodes)
    second = policy.select(nodes)
    assert second in nodes
    assert policy.rewards[0] == 1.0


def test_mcts_explore_updates_rewards_along_selected_path() -> None:
    root_a = PromptNode(prompt="a", index=0)
    root_b = PromptNode(prompt="b", index=1)
    child = PromptNode(prompt="c", index=2, parent=root_a)
    nodes = [root_a, root_b, child]
    policy = MCTSExploreSelectPolicy(ratio=0.5, alpha=0.0, beta=0.2, seed=0)

    selected = policy.select(nodes, initial_nodes=[root_a, root_b])
    policy.update([PromptNode(prompt="mut", results=[1])], len_questions=1, prompt_nodes=nodes)

    assert selected in nodes
    assert any(reward > 0 for reward in policy.rewards)


def test_unindexed_mutation_is_not_added_to_mcts_tree() -> None:
    root = PromptNode(prompt="seed", index=0)
    failed_child = PromptNode(prompt="failed", parent=root)

    assert failed_child.index is None
    assert root.child == []
