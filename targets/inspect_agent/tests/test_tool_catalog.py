"""ToolCatalog: seed + register/replace/unregister/rewrite + malformed handling."""

from __future__ import annotations

import pytest
from inspect_ai.tool import Tool, tool

from inspect_agent_target.tool_catalog import ToolCatalog


@tool
def _sample_tool() -> Tool:
    async def execute(query: str) -> str:
        """A sample tool.

        Args:
            query: the input query
        """
        return f"real:{query}"

    return execute


def _resolver(_name: str) -> Tool:
    return _sample_tool()


def test_seed_and_snapshot() -> None:
    cat = ToolCatalog.seed(_resolver, ["_sample_tool"])
    snap = cat.snapshot()
    assert len(snap) == 1
    assert snap[0]["name"] == "_sample_tool"
    assert "query" in snap[0]["parameters_schema"]["properties"]
    assert len(cat.tools()) == 1
    assert cat.names() == ["_sample_tool"]


def test_register_adds_tool() -> None:
    cat = ToolCatalog.seed(_resolver, [])
    cat.apply_register(
        {
            "name": "evil_tool",
            "description": "does evil",
            "parameters_schema": {
                "type": "object",
                "properties": {"x": {"type": "string", "description": "x"}},
                "required": ["x"],
            },
            "fake_return": "pwned",
        }
    )
    assert "evil_tool" in cat.names()
    snap = {s["name"]: s for s in cat.snapshot()}
    assert snap["evil_tool"]["description"] == "does evil"
    assert len(cat.tools()) == 1


@pytest.mark.asyncio
async def test_registered_tool_returns_canned() -> None:
    cat = ToolCatalog.seed(_resolver, [])
    cat.apply_register({"name": "evil", "description": "d", "fake_return": "PWNED"})
    out = await cat._defs["evil"].tool(x="anything")  # canned fn ignores kwargs
    assert out == "PWNED"


@pytest.mark.asyncio
async def test_replace_shadows_existing_keeps_schema() -> None:
    cat = ToolCatalog.seed(_resolver, ["_sample_tool"])
    cat.apply_replace({"name": "_sample_tool", "fake_return": "shadowed"})
    snap = {s["name"]: s for s in cat.snapshot()}
    assert "query" in snap["_sample_tool"]["parameters_schema"]["properties"]
    out = await cat._defs["_sample_tool"].tool(query="hi")
    assert out == "shadowed"


def test_unregister_removes() -> None:
    cat = ToolCatalog.seed(_resolver, ["_sample_tool"])
    cat.apply_unregister({"name": "_sample_tool"})
    assert cat.names() == []


def test_rewrite_doc_changes_description() -> None:
    cat = ToolCatalog.seed(_resolver, ["_sample_tool"])
    cat.apply_rewrite_doc({"name": "_sample_tool", "description": "NEW DESC"})
    snap = {s["name"]: s for s in cat.snapshot()}
    assert snap["_sample_tool"]["description"] == "NEW DESC"


def test_malformed_payloads_are_ignored() -> None:
    cat = ToolCatalog.seed(_resolver, ["_sample_tool"])
    cat.apply_register({})  # missing name
    cat.apply_register({"name": "bad", "parameters_schema": {"type": "nope", "properties": "x"}})
    cat.apply_replace({"name": "missing", "fake_return": "x"})
    cat.apply_unregister({"name": "missing"})
    cat.apply_rewrite_doc({"name": "_sample_tool"})  # missing description
    cat.apply_rewrite_doc({"name": "missing", "description": "x"})
    assert cat.names() == ["_sample_tool"]  # original intact, nothing added
