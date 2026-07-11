"""Tests for the per-server authorization tool trees (``tool_trees.py`` + data).

Covers: the data file's structural integrity across all 24 servers, the built
tag subtree (single-parent, identity-stable, correct parent links), the
tool->node resolution + root fallback, and the scope semantics the split exists
for (an ancestor node grants its descendants; a sibling does not; the server root
grants the whole subtree).
"""

from __future__ import annotations

import pytest
from superred.core.types.security_domain import scope_includes

from dtap_scaffold.forest import TOOLS_TAG, build_domain, tools_server_tag
from dtap_scaffold.tool_trees import build_server_tree, known_servers


def test_all_servers_build_and_are_single_parent_trees():
    for server in known_servers():
        tree = build_server_tree(server)
        keys = set(tree.nodes)
        roots = [k for k, tag in tree.nodes.items() if tag.parent is tree.root]
        assert roots, f"{server}: no root node parents to tools.{server}"
        for key, tag in tree.nodes.items():
            # every node's parent is either the server root or exactly one other node
            if tag.parent is tree.root:
                continue
            parent_keys = [k for k in keys if tree.nodes[k] is tag.parent]
            assert len(parent_keys) == 1, f"{server}.{key}: parent is not a single tree node"


def test_every_node_key_is_a_valid_tag_segment():
    # keys become tools.<server>.<key>; a dot in a key would corrupt the tag path.
    for server in known_servers():
        for key in build_server_tree(server).nodes:
            assert "." not in key, f"{server}: node key {key!r} contains a dot"


def test_no_tool_belongs_to_two_nodes_within_a_server():
    # tool_to_key must be unambiguous: a tool sits at exactly one authorization node.
    for server in known_servers():
        spec_tree = build_server_tree(server)
        seen: dict[str, str] = {}
        for tool, key in spec_tree.tool_to_key.items():
            assert tool not in seen or seen[tool] == key
            seen[tool] = key
        # tag_for_tool resolves each tool to its node tag (via tool_to_key + nodes)
        for tool, key in spec_tree.tool_to_key.items():
            assert spec_tree.tag_for_tool(tool) is spec_tree.nodes[key]


def test_root_tag_identity_is_stable_and_shared_with_forest():
    tree = build_server_tree("telecom")
    assert tree.root is tools_server_tag("telecom")
    assert tree.root is build_server_tree("telecom").root  # cached
    assert tree.root.parent is TOOLS_TAG
    assert tree.all_tags[0] is tree.root
    assert set(tree.all_tags[1:]) == set(tree.nodes.values())


def test_tag_for_tool_falls_back_to_root():
    tree = build_server_tree("legal")
    assert tree.tag_for_tool("no_such_tool_xyz") is tree.root
    # a known tool resolves to its node, not the root
    known_tool = next(iter(tree.tool_to_key))
    assert tree.tag_for_tool(known_tool) is not tree.root


def test_scope_semantics_ancestor_grants_descendant_sibling_does_not():
    # travel-suite: system(root node) -> customer -> public ; admin is a sibling leaf
    tree = build_server_tree("travel-suite")
    public, customer, admin, root = (
        tree.nodes["public"],
        tree.nodes["customer"],
        tree.nodes["admin"],
        tree.root,
    )
    # the whole-server grant reaches every node
    assert scope_includes(frozenset({root}), public)
    assert scope_includes(frozenset({root}), admin)
    # an ancestor grants its descendants (customer dominates public)
    assert scope_includes(frozenset({customer}), public)
    # the reverse does not hold (holding public cannot reach customer)
    assert not scope_includes(frozenset({public}), customer)
    # siblings are isolated (admin cannot tamper public-tier returns, and vice versa)
    assert not scope_includes(frozenset({admin}), public)
    assert not scope_includes(frozenset({public}), admin)


def test_legal_is_multi_root_under_the_server_root():
    # legal has no single apex tool: two peer planes hang directly under tools.legal.
    tree = build_server_tree("legal")
    roots = sorted(k for k, tag in tree.nodes.items() if tag.parent is tree.root)
    assert roots == ["firm_workspace", "public_records"]
    # neither peer dominates the other
    assert not scope_includes(
        frozenset({tree.nodes["public_records"]}), tree.nodes["firm_workspace"]
    )
    assert not scope_includes(
        frozenset({tree.nodes["firm_workspace"]}), tree.nodes["public_records"]
    )


def test_domain_assembles_for_every_server():
    # the whole subtree of each server must form a valid SecurityDomain (unique
    # names, every parent present) once dropped under the fixed forest.
    for server in known_servers():
        build_domain(build_server_tree(server).all_tags, [])  # raises on a bad tree


def test_full_forest_assembles_with_all_servers_at_once():
    all_tags = [tag for s in known_servers() for tag in build_server_tree(s).all_tags]
    build_domain(all_tags, [])  # raises on any cross-server name collision


@pytest.mark.parametrize("server", sorted(known_servers()))
def test_each_tool_resolves_to_a_node_tag_in_that_servers_subtree(server: str):
    tree = build_server_tree(server)
    subtree = set(tree.all_tags)
    for tool in tree.tool_to_key:
        assert tree.tag_for_tool(tool) in subtree


# (server, tool) -> the node key it MUST sit at. A guard against a control-of-return
# placement being silently de-escalated by a data edit (the subtree-membership and
# uniqueness tests above would not catch a wrong-node move). Covers recognizably
# high-privilege apex tools (must stay HIGH) and public/leaf reads (must stay LOW),
# in both directions, across a spread of servers. Update deliberately if a
# placement is intentionally changed.
_PINNED_PLACEMENTS = {
    # apex / high-trust tools -- a downgrade to a weaker node is a mis-scope
    ("telecom", "reload_all_tables"): "system",  # reload-every-table DBA apex
    ("gmail", "send_email"): "account",  # outbound send (control plane)
    ("OS-filesystem", "execute_command"): "system",  # arbitrary command apex
    ("paypal", "paypal_login"): "login",  # session/identity apex
    ("salesforce", "logout"): "session",  # session apex
    ("whatsapp", "delete_whatsapp_message"): "account",  # destructive, owner-only
    ("legal", "get_matter"): "firm_workspace",  # authoritative firm record read
    # public / leaf tools -- an upgrade to a stronger node is equally a mis-scope
    ("gmail", "list_messages"): "public",  # inbound read
    ("OS-filesystem", "read_file"): "read",  # read-only file access
    ("legal", "search"): "public_records",  # public case-law read
    ("legal", "add_matter_note"): "add_matter_note",  # low-trust append leaf
    ("whatsapp", "send_whatsapp_message"): "messages",  # send is NOT the account apex
}


@pytest.mark.parametrize(
    ("server", "tool", "node_key"), [(s, t, n) for (s, t), n in _PINNED_PLACEMENTS.items()]
)
def test_high_and_low_trust_tools_are_pinned_to_their_nodes(server: str, tool: str, node_key: str):
    tree = build_server_tree(server)
    assert tool in tree.tool_to_key, f"{server}: pinned tool {tool!r} no longer exists in the tree"
    assert tree.tool_to_key[tool] == node_key, (
        f"{server}.{tool} moved from node {node_key!r} to {tree.tool_to_key[tool]!r} -- "
        "a control-of-return placement changed; update the pin only if intended"
    )
