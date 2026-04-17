"""Tests for TapNode and TapTree."""

import pytest

from tap_optimizer.tree import TapNode, TapTree


# ── TapNode tests ──────────────────────────────────────────────────────────


class TestTapNode:
    def test_creation_with_defaults(self):
        node = TapNode(depth=0, parent_id=None, conversation_history=[])
        assert node.depth == 0
        assert node.parent_id is None
        assert node.conversation_history == []
        assert node.node_id  # non-empty string
        assert node.prompt is None
        assert node.target_response is None
        assert node.score == 0.0
        assert node.is_on_topic is True
        assert node.pruned is False

    def test_unique_auto_generated_node_ids(self):
        a = TapNode(depth=0, parent_id=None, conversation_history=[])
        b = TapNode(depth=0, parent_id=None, conversation_history=[])
        assert a.node_id != b.node_id


# ── TapTree tests ──────────────────────────────────────────────────────────


class TestTapTree:
    def test_create_root_nodes(self):
        tree = TapTree()
        roots = tree.create_root_nodes(width=3)
        assert len(roots) == 3
        for node in roots:
            assert node.depth == 0
            assert node.parent_id is None
            assert node.conversation_history == []

    def test_branch_creates_children_at_correct_depth_with_parent_id(self):
        tree = TapTree()
        roots = tree.create_root_nodes(width=1)
        parent = roots[0]
        children = tree.branch(parent, branching_factor=4)
        assert len(children) == 4
        for child in children:
            assert child.depth == parent.depth + 1
            assert child.parent_id == parent.node_id

    def test_branch_deep_copies_conversation_history(self):
        tree = TapTree()
        roots = tree.create_root_nodes(width=1)
        parent = roots[0]
        parent.conversation_history.append({"role": "user", "content": "hello"})

        children = tree.branch(parent, branching_factor=2)
        # Children start with a copy of parent's history
        assert children[0].conversation_history == [{"role": "user", "content": "hello"}]

        # Mutating one child's history must not affect the other or the parent
        children[0].conversation_history.append({"role": "assistant", "content": "hi"})
        assert len(children[1].conversation_history) == 1
        assert len(parent.conversation_history) == 1

    def test_get_leaves_excludes_pruned_nodes(self):
        tree = TapTree()
        roots = tree.create_root_nodes(width=3)
        roots[0].pruned = True
        leaves = tree.get_leaves()
        assert len(leaves) == 2
        assert roots[0] not in leaves

    def test_get_leaves_returns_only_leaf_nodes(self):
        """A node that has non-pruned children is not a leaf."""
        tree = TapTree()
        roots = tree.create_root_nodes(width=1)
        parent = roots[0]
        tree.branch(parent, branching_factor=2)
        leaves = tree.get_leaves()
        # parent should no longer be a leaf
        assert parent not in leaves
        assert len(leaves) == 2

    def test_prune_off_topic(self):
        tree = TapTree()
        roots = tree.create_root_nodes(width=3)
        roots[0].is_on_topic = False
        roots[1].is_on_topic = True
        roots[2].is_on_topic = False
        tree.prune_off_topic()
        assert roots[0].pruned is True
        assert roots[1].pruned is False
        assert roots[2].pruned is True

    def test_prune_to_width_keeps_top_w_by_score(self):
        tree = TapTree()
        roots = tree.create_root_nodes(width=5)
        for i, node in enumerate(roots):
            node.score = float(i)  # scores: 0, 1, 2, 3, 4
        tree.prune_to_width(width=2)
        leaves = tree.get_leaves()
        assert len(leaves) == 2
        scores = sorted(n.score for n in leaves)
        assert scores == [3.0, 4.0]

    def test_best_node_returns_highest_scorer(self):
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

    def test_best_node_returns_none_on_empty_tree(self):
        tree = TapTree()
        assert tree.best_node() is None

    def test_best_node_ignores_nodes_without_prompt(self):
        tree = TapTree()
        roots = tree.create_root_nodes(width=2)
        roots[0].score = 10.0
        roots[0].prompt = None  # no prompt set
        roots[1].score = 2.0
        roots[1].prompt = "x"
        best = tree.best_node()
        assert best is roots[1]

    def test_best_node_ignores_pruned_nodes(self):
        tree = TapTree()
        roots = tree.create_root_nodes(width=2)
        roots[0].score = 10.0
        roots[0].prompt = "a"
        roots[0].pruned = True
        roots[1].score = 2.0
        roots[1].prompt = "b"
        best = tree.best_node()
        assert best is roots[1]
