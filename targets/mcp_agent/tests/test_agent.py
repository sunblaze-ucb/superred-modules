"""Agent-loop + tool-mapping tests, with a stub MCP session and mock completion."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from mcp_agent_target import mcp_tools_to_litellm, poison_tool_description, run_agent


def _tool(name: str, description: str = "", schema: dict | None = None) -> Any:
    return SimpleNamespace(
        name=name,
        description=description,
        input_schema=schema or {"type": "object", "properties": {}},
    )


def _tc(call_id: str, name: str, arguments: str) -> Any:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def _resp(content: str | None = None, tool_calls: list | None = None) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))]
    )


class _StubSession:
    def __init__(self, tools: list, results: dict | None = None) -> None:
        self._tools = tools
        self._results = results or {}
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self) -> Any:
        return SimpleNamespace(tools=self._tools)

    async def call_tool(self, name: str, arguments: dict | None = None) -> Any:
        self.calls.append((name, arguments or {}))
        text = self._results.get(name, f"result-of-{name}")
        return SimpleNamespace(content=[SimpleNamespace(text=text)], is_error=False)


def _scripted(responses: list) -> Any:
    it = iter(responses)

    async def complete(**kwargs: Any) -> Any:
        return next(it)

    return complete


# -- tool mapping + poison ---------------------------------------------------
def test_poison_tool_description() -> None:
    assert poison_tool_description("base", "EVIL") == "base\n\nEVIL"
    assert poison_tool_description("", "EVIL") == "EVIL"
    assert poison_tool_description("base", "") == "base"


def test_mapping_applies_poison_to_named_tool() -> None:
    tools = [_tool("safe", "safe desc"), _tool("danger", "danger desc")]
    specs = mcp_tools_to_litellm(tools, poison_tool="danger", poison_injection="CALL ME")
    by_name = {s["function"]["name"]: s for s in specs}
    assert by_name["safe"]["function"]["description"] == "safe desc"
    assert "CALL ME" in by_name["danger"]["function"]["description"]
    assert by_name["danger"]["function"]["parameters"] == {"type": "object", "properties": {}}


# -- agent loop --------------------------------------------------------------
async def test_agent_executes_tool_then_answers() -> None:
    session = _StubSession([_tool("danger", "d")], results={"danger": "did the thing"})
    complete = _scripted(
        [
            _resp(tool_calls=[_tc("c1", "danger", '{"payload": "x"}')]),
            _resp(content="all done"),
        ]
    )
    run = await run_agent(
        session=session,
        complete=complete,
        model="m",
        system_prompt="sys",
        user_message="hi",
        max_turns=4,
    )
    assert run.called_tool_names == ["danger"]
    assert run.tool_calls[0].arguments == {"payload": "x"}
    assert run.tool_calls[0].result == "did the thing"
    assert run.final_response == "all done" and run.turns == 2
    assert session.calls == [("danger", {"payload": "x"})]


async def test_agent_answers_without_tools() -> None:
    session = _StubSession([_tool("danger", "d")])
    run = await run_agent(
        session=session,
        complete=_scripted([_resp(content="just chatting")]),
        model="m",
        system_prompt="",
        user_message="hi",
    )
    assert run.called_tool_names == [] and run.final_response == "just chatting" and run.turns == 1


async def test_max_turns_bounds_the_loop() -> None:
    session = _StubSession([_tool("danger", "d")])
    # always requests a tool -> loop must stop at max_turns
    always_tool = _scripted([_resp(tool_calls=[_tc(str(i), "danger", "{}")]) for i in range(10)])
    run = await run_agent(
        session=session,
        complete=always_tool,
        model="m",
        system_prompt="",
        user_message="hi",
        max_turns=3,
    )
    assert run.turns == 3 and len(run.tool_calls) == 3


async def test_bad_tool_arguments_default_to_empty() -> None:
    session = _StubSession([_tool("danger", "d")])
    complete = _scripted([_resp(tool_calls=[_tc("c1", "danger", "not-json")]), _resp(content="ok")])
    run = await run_agent(
        session=session, complete=complete, model="m", system_prompt="", user_message="hi"
    )
    assert run.tool_calls[0].arguments == {}


async def test_list_tools_error_is_captured() -> None:
    class _Boom:
        async def list_tools(self) -> Any:
            raise RuntimeError("no server")

        async def call_tool(
            self, name: str, arguments: dict | None = None
        ) -> Any:  # pragma: no cover
            raise AssertionError

    run = await run_agent(
        session=_Boom(), complete=_scripted([]), model="m", system_prompt="", user_message="hi"
    )
    assert "list_tools failed" in run.error and run.called_tool_names == []


async def test_completion_error_is_captured() -> None:
    session = _StubSession([_tool("danger", "d")])

    async def boom(**kwargs: Any) -> Any:
        raise RuntimeError("provider down")

    run = await run_agent(
        session=session, complete=boom, model="m", system_prompt="", user_message="hi"
    )
    assert "completion failed" in run.error
