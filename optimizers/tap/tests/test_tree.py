"""Tests for TapNode and TapTree."""

import random

from tap_optimizer.tree import TapNode, TapTree


class TestTapNode:
    def test_creation_with_defaults(self) -> None:
        node = TapNode(depth=0, parent_id=None, conversation_history=[])
        assert node.depth == 0
        assert node.parent_id is None
        assert node.conversation_history == []
        assert node.node_id
        assert node.prompt is None
        assert node.target_response is None
        assert node.score == 0.0
        assert node.is_on_topic is True
        assert node.pruned is False

    def test_unique_auto_generated_node_ids(self) -> None:
        a = TapNode(depth=0, parent_id=None, conversation_history=[])
        b = TapNode(depth=0, parent_id=None, conversation_history=[])
        assert a.node_id != b.node_id


class TestTapTree:
    def test_create_root_nodes(self) -> None:
        tree = TapTree()
        roots = tree.create_root_nodes(width=3)
        assert len(roots) == 3
        for node in roots:
            assert node.depth == 0
            assert node.parent_id is None
            assert node.conversation_history == []

    def test_branch_creates_children_at_correct_depth_with_parent_id(self) -> None:
        tree = TapTree()
        roots = tree.create_root_nodes(width=1)
        parent = roots[0]
        children = tree.branch(parent, branching_factor=4)
        assert len(children) == 4
        for child in children:
            assert child.depth == parent.depth + 1
            assert child.parent_id == parent.node_id

    def test_branch_deep_copies_conversation_history(self) -> None:
        tree = TapTree()
        roots = tree.create_root_nodes(width=1)
        parent = roots[0]
        parent.conversation_history.append({"role": "user", "content": "hello"})

        children = tree.branch(parent, branching_factor=2)
        assert children[0].conversation_history == [{"role": "user", "content": "hello"}]

        children[0].conversation_history.append({"role": "assistant", "content": "hi"})
        assert len(children[1].conversation_history) == 1
        assert len(parent.conversation_history) == 1

    def test_get_leaves_excludes_pruned_nodes(self) -> None:
        tree = TapTree()
        roots = tree.create_root_nodes(width=3)
        roots[0].pruned = True
        leaves = tree.get_leaves()
        assert len(leaves) == 2
        assert roots[0] not in leaves

    def test_get_leaves_returns_only_leaf_nodes(self) -> None:
        tree = TapTree()
        roots = tree.create_root_nodes(width=1)
        parent = roots[0]
        tree.branch(parent, branching_factor=2)
        leaves = tree.get_leaves()
        assert parent not in leaves
        assert len(leaves) == 2

    def test_prune_off_topic_keeps_minimum_candidate_when_all_are_off_topic(self) -> None:
        tree = TapTree(rng=random.Random(0))
        roots = tree.create_root_nodes(width=3)
        for root in roots:
            root.is_on_topic = False

        tree.prune_off_topic(width=2)

        leaves = tree.get_leaves()
        assert len(leaves) == 2

    def test_prune_to_width_keeps_top_w_by_score(self) -> None:
        tree = TapTree(rng=random.Random(0))
        roots = tree.create_root_nodes(width=5)
        for i, node in enumerate(roots):
            node.score = float(i)
        tree.prune_to_width(width=2)
        leaves = tree.get_leaves()
        assert len(leaves) == 2
        scores = sorted(n.score for n in leaves)
        assert scores == [3.0, 4.0]

    def test_prune_to_width_randomizes_ties_deterministically_with_seed(self) -> None:
        tree = TapTree(rng=random.Random(7))
        roots = tree.create_root_nodes(width=4)
        for i, node in enumerate(roots):
            node.score = 5.0
            node.prompt = f"prompt-{i}"

        tree.prune_to_width(width=2)

        kept = sorted(node.prompt for node in tree.get_leaves())
        assert kept == ["prompt-1", "prompt-3"]

    def test_prune_to_width_keeps_minimum_when_all_scores_are_zero(self) -> None:
        tree = TapTree(rng=random.Random(0))
        roots = tree.create_root_nodes(width=4)
        for node in roots:
            node.score = 0.0
        tree.prune_to_width(width=3)

        assert len(tree.get_leaves()) == 2

    def test_best_node_returns_highest_scorer(self) -> None:
        tree = TapTree()
        roots = tree.create_root_nodes(width=3)
        roots[0].score = 1.0
        roots[0].prompt = "a"
        roots[1].score = 5.0
        roots[1].prompt = "b"
        roots[2].score = 3.0
        roots[2].prompt = "c"
        best = tree.best_node()
        assert best is roots[1]

    def test_best_node_returns_none_on_empty_tree(self) -> None:
        tree = TapTree()
        assert tree.best_node() is None

    def test_best_node_ignores_nodes_without_prompt(self) -> None:
        tree = TapTree()
        roots = tree.create_root_nodes(width=2)
        roots[0].score = 10.0
        roots[0].prompt = None
        roots[1].score = 2.0
        roots[1].prompt = "x"
        best = tree.best_node()
        assert best is roots[1]

    def test_best_node_ignores_pruned_nodes(self) -> None:
        tree = TapTree()
        roots = tree.create_root_nodes(width=2)
        roots[0].score = 10.0
        roots[0].prompt = "a"
        roots[0].pruned = True
        roots[1].score = 2.0
        roots[1].prompt = "b"
        best = tree.best_node()
        assert best is roots[1]
