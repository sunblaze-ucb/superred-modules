"""Per-server authorization TREE for the env-tool return-tamper surface.

Each MCP server's single ``tools.<server>`` tag is the ROOT of a small
single-parent authorization tree. Every env tool sits at the node whose trust
boundary an attacker must compromise to CONTROL that tool's RETURN
(control-of-return): a node is the tag ``tools.<server>.<node>`` whose parent is
the dominating node (or ``tools.<server>`` for a tree root). Scoping a node
grants it plus its descendants (the lower-trust surfaces it dominates), so
``{tools.<server>}`` still grants the whole server -- backward-compatible.

The trees live in :data:`data/tool_trees.json` (one per server, derived from the
real service's authorization structure). Edit that file to refine a placement --
no code change. A tool NOT present in its server's tree (version drift, or a
dynamically-registered server such as ``legal``) falls back to the
``tools.<server>`` root tag: the conservative whole-server grant.

Every tag is built ONCE per server and cached (:func:`build_server_tree` is
``lru_cache``-d, and :func:`~dtap_scaffold.forest.tools_server_tag` is cached),
so the identical tag object is reused in the domain, the controllables, and the
fired events -- the identity the forest's ``includes`` check requires.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any

from superred.core.types.security_domain import SecurityDomainTag

from dtap_scaffold.forest import tools_server_tag


def _load_trees() -> dict[str, Any]:
    text = (
        resources.files("dtap_scaffold")
        .joinpath("data/tool_trees.json")
        .read_text(encoding="utf-8")
    )
    data: dict[str, Any] = json.loads(text)
    return data


_TREES: dict[str, Any] = _load_trees()


def known_servers() -> frozenset[str]:
    """Servers that ship a static tool tree."""
    return frozenset(_TREES)


@dataclass(frozen=True)
class ServerToolTree:
    """The built (identity-stable) tag subtree for one MCP server.

    ``root`` is ``tools.<server>``. ``nodes`` maps each node key to its tag,
    ``labels`` its human label. ``tool_to_tag`` / ``tool_to_key`` map a tool name
    to its node's tag / key. Tools absent from the tree resolve to ``root`` via
    :meth:`tag_for_tool` (the whole-server fallback).
    """

    server: str
    root: SecurityDomainTag
    nodes: dict[str, SecurityDomainTag]
    labels: dict[str, str]
    tool_to_tag: dict[str, SecurityDomainTag]
    tool_to_key: dict[str, str]

    @property
    def all_tags(self) -> list[SecurityDomainTag]:
        """The server root plus every node tag (what ``build_domain`` needs)."""
        return [self.root, *self.nodes.values()]

    def tag_for_tool(self, tool: str) -> SecurityDomainTag:
        """The node tag governing *tool*'s return, or the root tag as a fallback."""
        return self.tool_to_tag.get(tool, self.root)


@cache
def build_server_tree(server: str) -> ServerToolTree:
    """Build (once, cached) the authorization tag subtree for *server*.

    Reads the server's tree from :data:`data/tool_trees.json` and materializes a
    :class:`SecurityDomainTag` per node under the cached ``tools.<server>`` root,
    resolving each node's single parent (a root node parents to ``tools.<server>``;
    otherwise to its parent node's tag). A server with no shipped tree yields an
    empty subtree (just the root); every tool then falls back to the root.
    """
    root = tools_server_tag(server)
    spec = _TREES.get(server)
    nodes: dict[str, SecurityDomainTag] = {}
    labels: dict[str, str] = {}
    tool_to_tag: dict[str, SecurityDomainTag] = {}
    tool_to_key: dict[str, str] = {}
    if spec is None:
        return ServerToolTree(server, root, nodes, labels, tool_to_tag, tool_to_key)

    remaining: list[dict[str, Any]] = list(spec["nodes"])
    # Resolve parents in passes: a node is placed once its parent is (roots first).
    for _ in range(len(remaining) + 1):
        if not remaining:
            break
        still: list[dict[str, Any]] = []
        for node in remaining:
            parent_key = node.get("parent")
            if parent_key is None:
                parent_tag: SecurityDomainTag = root
            elif parent_key in nodes:
                parent_tag = nodes[parent_key]
            else:
                still.append(node)
                continue
            key = str(node["key"])
            tag = SecurityDomainTag(f"tools.{server}.{key}", parent=parent_tag)
            nodes[key] = tag
            labels[key] = str(node.get("label", key))
            for tool in node.get("tools", ()):
                tool_to_tag[str(tool)] = tag
                tool_to_key[str(tool)] = key
        remaining = still
    if remaining:  # a broken parent chain in the data file; fail loud rather than mis-scope
        bad = ", ".join(str(n.get("key")) for n in remaining)
        raise ValueError(f"tool_trees.json: unresolved parent(s) for {server!r} nodes: {bad}")

    return ServerToolTree(server, root, nodes, labels, tool_to_tag, tool_to_key)


__all__ = ["ServerToolTree", "build_server_tree", "known_servers"]
