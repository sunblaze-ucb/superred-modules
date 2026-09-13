"""InjectionSpec tests.

The pure shape-handling logic (``_inject_into_result``) and the ``InjectionSpec``
helpers need no ``crewai`` install and run offline; the tests that wrap a *real*
CrewAI tool and drive it through the framework's own output formatting require
``crewai`` and are skipped when it is absent (they run in CI).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from crewai_agent_target.injection import InjectionSpec, _inject_into_result


# -- InjectionSpec helpers (offline) -----------------------------------------
def test_apply_backstory() -> None:
    empty = InjectionSpec()
    assert empty.apply_backstory("base") == "base"
    assert empty.apply_backstory(None) == ""  # backstory must stay a str
    spec = InjectionSpec(system_prompt_suffix="ATK", tool_output_appendix="PAYLOAD")
    assert spec.apply_backstory("base") == "base\n\nATK"
    assert spec.apply_backstory(None) == "ATK"
    assert spec.apply_backstory("") == "ATK"


def test_wrap_tools_no_appendix_returns_tools_unchanged() -> None:
    # No injection -> no wrapper, no behaviour change (and no crewai import needed).
    tools = ["t1", "t2"]
    result = InjectionSpec().wrap_tools(tools)
    assert result == tools and result is not tools  # a copy, same contents


# -- _inject_into_result: every tool-return shape (offline) ------------------
def test_inject_str_result() -> None:
    assert _inject_into_result("weather: sunny", "MARK") == "weather: sunny\n\nMARK"


def test_inject_empty_appendix_is_identity() -> None:
    obj = {"k": "v"}
    assert _inject_into_result(obj, "") is obj  # untouched, same object
    assert _inject_into_result("x", "") == "x"


def test_inject_nonstring_dict_shape() -> None:
    # The #183 HIGH bug injected only into str results and silently dropped every
    # other shape. A dict must still carry the payload into its model-visible text.
    result = _inject_into_result({"temp": 22, "city": "NYC"}, "MARK")
    assert isinstance(result, str)
    assert "MARK" in result and "temp" in result


def test_inject_nonstring_list_and_scalars() -> None:
    assert "MARK" in _inject_into_result([1, 2, 3], "MARK")
    assert _inject_into_result(None, "MARK") == "None\n\nMARK"
    assert _inject_into_result(42, "MARK") == "42\n\nMARK"


def test_inject_pydantic_model_shape() -> None:
    # Structured output: rendered via model_dump_json so the payload lands after
    # valid JSON (still a str, so it survives any result_schema serialization).
    class Weather(BaseModel):
        city: str
        temp: int

    result = _inject_into_result(Weather(city="NYC", temp=22), "MARK")
    assert isinstance(result, str)
    assert '"city":"NYC"' in result and result.endswith("MARK")


def test_inject_bytes_shape() -> None:
    assert _inject_into_result(b"raw bytes", "MARK") == "raw bytes\n\nMARK"
    assert _inject_into_result(bytearray(b"ba"), "MARK") == "ba\n\nMARK"


def test_inject_passes_coroutine_through_untouched() -> None:
    # An async tool's un-awaited return must not be stringified (that would destroy
    # the result); the wrapper's run() resolves it first, then injects.
    async def _coro() -> str:
        return "r"

    coro = _coro()
    try:
        assert _inject_into_result(coro, "MARK") is coro
    finally:
        coro.close()


# -- wrapping a real CrewAI tool (CI: requires crewai) -----------------------
def test_wrap_real_tool_str_result_reaches_observation() -> None:
    pytest.importorskip("crewai")
    from crewai.tools import tool

    @tool("get_weather")
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"The weather in {city} is sunny."

    spec = InjectionSpec(tool_output_appendix="INDIRECT_INJECT_MARK")
    wrapped = spec.wrap_tools([get_weather])[0]
    raw = wrapped.run(city="NYC")
    assert "INDIRECT_INJECT_MARK" in raw
    # ...and it survives CrewAI's own output formatting (what the model actually reads).
    observation = wrapped.format_output_for_agent(raw)
    assert "INDIRECT_INJECT_MARK" in observation
    # name/description are forwarded so the agent sees an identical tool.
    assert wrapped.name == "get_weather"


def test_wrap_real_tool_nonstring_dict_result_reaches_observation() -> None:
    # A @tool returning a dict (structured, not str) must still carry the payload
    # into the model-visible observation.
    pytest.importorskip("crewai")
    from crewai.tools import tool

    @tool("lookup")
    def lookup(key: str) -> dict:
        """Look up a key."""
        return {"value": key, "ok": True}

    wrapped = InjectionSpec(tool_output_appendix="DICT_MARK").wrap_tools([lookup])[0]
    observation = wrapped.format_output_for_agent(wrapped.run(key="k1"))
    assert "DICT_MARK" in observation and "value" in observation


def test_wrap_basetool_subclass_pydantic_result_reaches_observation() -> None:
    # A BaseTool subclass (not a @tool function) returning a pydantic BaseModel:
    # both the alternate tool type and the structured shape must be handled.
    pytest.importorskip("crewai")
    from crewai.tools import BaseTool

    class Weather(BaseModel):
        city: str
        temp: int

    class WeatherArgs(BaseModel):
        city: str

    class WeatherTool(BaseTool):
        name: str = "weather"
        description: str = "Get structured weather."
        args_schema: type[BaseModel] = WeatherArgs

        def _run(self, city: str) -> Weather:
            return Weather(city=city, temp=22)

    wrapped = InjectionSpec(tool_output_appendix="PY_MARK").wrap_tools([WeatherTool()])[0]
    observation = wrapped.format_output_for_agent(wrapped.run(city="NYC"))
    assert "PY_MARK" in observation and "NYC" in observation


async def test_wrap_async_tool_run_injects() -> None:
    # The agent-driven path is _run (CrewAI's to_structured_tool binds the tool's
    # callable to _run, not run). For an ASYNC inner tool, _run returns an
    # un-awaited coroutine that must be awaited and injected, not passed through
    # (the async-tool false-negative fixed in this PR).
    import asyncio

    pytest.importorskip("crewai")
    from crewai.tools import BaseTool

    class AsyncArgs(BaseModel):
        q: str

    class AsyncTool(BaseTool):
        name: str = "async_lookup"
        description: str = "Async lookup."
        args_schema: type[BaseModel] = AsyncArgs

        async def _run(self, q: str) -> str:
            return f"result for {q}"

    wrapped = InjectionSpec(tool_output_appendix="ASYNC_MARK").wrap_tools([AsyncTool()])[0]
    out = wrapped._run(q="x")
    assert asyncio.iscoroutine(out)  # async inner tool -> wrapper returns a coroutine
    resolved = await out
    assert "result for x" in resolved and "ASYNC_MARK" in resolved
