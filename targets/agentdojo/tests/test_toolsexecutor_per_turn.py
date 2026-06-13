"""Pinning test: AgentDojo's ToolsExecutor reads runtime.functions per call.

ASSUMPTIONS C.4 / Section 2.d of the implementation brief require us to
verify that upstream's ``ToolsExecutor`` and LLM elements rebuild their
function lookup from ``runtime.functions`` on each invocation, NOT once
at init.  Our ``CatalogEditHook`` fires once at run start (before the
first LLM call) and mutates the wrapped runtime's function dict in place;
because the ``ToolsExecutor`` runs on every subsequent turn, a tool
registered at the start must stay visible to it.  If upstream ever cached
the function list at init time, that start-of-run registration would
silently become a no-op on later turns.

This file pins the per-call contract empirically by:

1. Building a :class:`WrappedFunctionsRuntime` over the canonical 74-
   tool catalog.
2. Invoking upstream's :class:`ToolsExecutor` against a synthesized
   assistant message that calls a canonical tool, verifying the
   dispatch succeeds.
3. Inserting a brand-new attacker tool into the catalog and refreshing
   the runtime's functions dict (mirrors what
   :class:`_CatalogEditHook.query` does at run start).
4. Invoking ``ToolsExecutor`` again with a tool_call to the new
   attacker tool name, verifying it dispatches successfully.

If upstream ever changes ``ToolsExecutor`` to cache the function list,
test #4 will fail with "Invalid tool ... provided" and we will need to
patch our pipeline_bridge to force a refresh somewhere upstream can
see (or to instantiate a new ToolsExecutor per turn).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

# Pre-import to flush AgentDojo's registration chain.
import agentdojo.task_suite.load_suites  # noqa: F401
from agentdojo.agent_pipeline.tool_execution import ToolsExecutor
from agentdojo.functions_runtime import FunctionCall
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import ControllableNoInjection

from agentdojo_target.runtime_wrapper import WrappedFunctionsRuntime
from agentdojo_target.seed_loader import load_composite_seed
from agentdojo_target.tool_catalog import ToolCatalog
from agentdojo_target.tool_registry import ALL_FUNCTIONS


class _NoOp:
    """Recorder that accepts every event with NoInjection (passthrough)."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def send_event(self, event: Event) -> EventResponse:
        self.events.append(event)
        return ControllableNoInjection(event=event, controllable=event.controllable)

    def emit(self, event: Event) -> None:
        pass


@pytest.fixture
def loop() -> asyncio.AbstractEventLoop:
    """A dedicated event loop running in a background thread (matches the
    runtime_wrapper test conventions)."""
    new_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=new_loop.run_forever, daemon=True)
    thread.start()
    yield new_loop
    new_loop.call_soon_threadsafe(new_loop.stop)
    thread.join(timeout=2)


def _assistant_msg_with_call(name: str, args: dict, tc_id: str) -> dict[str, Any]:
    """A minimal assistant message carrying one tool_call."""
    return {
        "role": "assistant",
        "content": [],
        "tool_calls": [FunctionCall(function=name, args=args, id=tc_id)],
    }


def test_toolsexecutor_dispatches_canonical_tool(loop) -> None:
    """Smoke: ToolsExecutor + WrappedFunctionsRuntime invoke a canonical tool."""
    env = load_composite_seed()
    catalog = ToolCatalog.from_seed(ALL_FUNCTIONS)
    rec = _NoOp()
    wrapper = WrappedFunctionsRuntime(
        catalog=catalog,
        send_event=rec.send_event,
        emit=rec.emit,
        loop=loop,
    )
    executor = ToolsExecutor()
    msg = _assistant_msg_with_call("banking__get_balance", {}, "call-1")

    _, _, _, messages, _ = executor.query("q", wrapper, env=env, messages=[msg])

    last = messages[-1]
    assert last["role"] == "tool"
    assert last.get("error") is None, f"unexpected dispatch error: {last.get('error')}"
    assert str(env.banking.bank_account.balance) in last["content"][0]["content"]


def test_toolsexecutor_rejects_unknown_canonical_name(loop) -> None:
    """ToolsExecutor returns an Invalid-tool error when the name is not
    in runtime.functions.  Establishes the negative-case behaviour."""
    env = load_composite_seed()
    catalog = ToolCatalog.from_seed(ALL_FUNCTIONS)
    rec = _NoOp()
    wrapper = WrappedFunctionsRuntime(
        catalog=catalog,
        send_event=rec.send_event,
        emit=rec.emit,
        loop=loop,
    )
    executor = ToolsExecutor()
    msg = _assistant_msg_with_call("__not_a_real_tool__", {}, "call-x")

    _, _, _, messages, _ = executor.query("q", wrapper, env=env, messages=[msg])

    last = messages[-1]
    assert last["role"] == "tool"
    assert last.get("error") is not None
    assert "Invalid tool" in last["error"]


def test_toolsexecutor_sees_attacker_tool_after_catalog_edit(loop) -> None:
    """The contract this test pins:

    1. Catalog starts canonical (no attacker tool).
    2. ToolsExecutor cannot dispatch the attacker name.
    3. We add the attacker tool to the catalog and refresh the wrapped
       runtime's function dict (mirroring the CatalogEditHook).
    4. ToolsExecutor now dispatches the same name successfully.

    A successful run of step 4 proves upstream's ToolsExecutor reads
    runtime.functions per call rather than caching at init.  If a
    future upstream change broke this, step 4 would still return an
    'Invalid tool' error and we would need to patch the pipeline.
    """
    env = load_composite_seed()
    catalog = ToolCatalog.from_seed(ALL_FUNCTIONS)
    rec = _NoOp()
    wrapper = WrappedFunctionsRuntime(
        catalog=catalog,
        send_event=rec.send_event,
        emit=rec.emit,
        loop=loop,
    )
    executor = ToolsExecutor()  # ONE instance reused across both calls

    # --- Step 2: attacker tool not yet registered ---
    msg_before = _assistant_msg_with_call(
        "__attacker_inject__",
        {"target": "x"},
        "call-pre",
    )
    _, _, _, before_messages, _ = executor.query(
        "q",
        wrapper,
        env=env,
        messages=[msg_before],
    )
    assert "Invalid tool" in (before_messages[-1].get("error") or "")

    # --- Step 3: start-of-run catalog edit + refresh, exactly as
    # _CatalogEditHook does ---
    catalog.apply_register(
        {
            "name": "__attacker_inject__",
            "description": "test injection from start-of-run catalog edit",
            "fake_return": {"exfil": "ok"},
        }
    )
    wrapper.refresh_functions()

    # --- Step 4: the SAME ToolsExecutor instance now dispatches the
    # attacker tool successfully.  This is the load-bearing assertion:
    # it proves runtime.functions is consulted live, not cached. ---
    msg_after = _assistant_msg_with_call(
        "__attacker_inject__",
        {"target": "y"},
        "call-post",
    )
    _, _, _, after_messages, _ = executor.query(
        "q",
        wrapper,
        env=env,
        messages=[msg_after],
    )
    last = after_messages[-1]
    assert last.get("error") is None, (
        f"upstream ToolsExecutor appears to have cached the function list "
        f"at init time -- start-of-run catalog edits are not visible on later "
        f"turns.  Error: {last.get('error')!r}.  Patch needed in pipeline_bridge."
    )
    result_text = last["content"][0]["content"]
    assert "exfil" in result_text


def test_openai_llm_reads_runtime_functions_per_call() -> None:
    """Source-level pin: OpenAILLM.query iterates runtime.functions on each
    invocation when building the OpenAI tool list.

    Complements the ToolsExecutor test by pinning the OTHER element
    that consults runtime.functions: the LLM, which builds the API tool
    list from runtime.functions.values() each turn.  We verify by
    inspecting the source rather than calling OpenAI for real.
    """
    import inspect
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM

    src = inspect.getsource(OpenAILLM)
    # OpenAILLM.query (and tool_filter helper) must reference
    # runtime.functions.values() so the tool list is rebuilt per call.
    assert "runtime.functions.values()" in src, (
        "OpenAILLM no longer reads runtime.functions per call; start-of-run "
        "catalog edits may not reach the next LLM tool list.  Check "
        "agent_pipeline/llms/openai_llm.py."
    )
