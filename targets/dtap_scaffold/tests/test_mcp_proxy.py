"""Offline tests for the host MCP proxy (no Docker, no aiohttp bind, no fastmcp).

The proxy's two responsibilities are exercised directly on the pure core methods,
with the genuine-backend call (:meth:`HostMCPProxy._forward`) monkeypatched:

* tool-description editing (override/suffix) in :meth:`list_tools`;
* the call chokepoint in :meth:`handle_tool_call` -- forward, fire ONE env_tool
  PostCall carrying the genuine return, and return the injected value on tampering
  (and genuine on decline), emitting no separate ObservableEvent (emit-once).

The JSON-RPC brains of the HTTP layer are tested via :meth:`_dispatch_rpc`
(no socket needed). The single real aiohttp bind is a ``live``/``docker`` test,
skipped unless ``DTAP_LIVE`` is set.
"""

from __future__ import annotations

import json
import os

import pytest
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePostCallEvent,
    ObservableEvent,
)

from dtap_scaffold.controllables import env_tool_output_controllable
from dtap_scaffold.forest import tools_server_tag
from dtap_scaffold.mcp_proxy import (
    _HOST_GATEWAY,
    HostMCPProxy,
    _extract_mcp_result,
    _lease_port,
)
from dtap_scaffold.protocols import MCPProxy
from dtap_scaffold.types import ProxyTool

# --------------------------- helpers --------------------------------------


def _recorder(injections: dict[str, str]):
    """A (emit, send_event, events, observables) recorder, like the agent_base test."""
    events: list = []
    observables: list = []

    def emit(evt):
        observables.append(evt)

    async def send_event(evt):
        events.append(evt)
        ctrl = getattr(evt, "controllable", None)
        if ctrl is not None and ctrl.name in injections:
            return ControllableInjection(event=evt, controllable=ctrl, value=injections[ctrl.name])
        return ControllableNoInjection(event=evt, controllable=ctrl)

    return emit, send_event, events, observables


def _proxy_with_forward(genuine: str = "GENUINE"):
    """A proxy whose ``_forward`` returns *genuine* and that fronts one env server."""
    proxy = HostMCPProxy()
    ctrl = env_tool_output_controllable("travel-suite", tools_server_tag("travel-suite"))
    proxy.set_env_tool_controllables({"travel-suite": ctrl})

    async def fake_forward(server, tool, params):
        return genuine

    proxy._forward = fake_forward  # type: ignore[method-assign]
    return proxy, ctrl


# --------------------------- protocol conformance -------------------------


def test_satisfies_mcpproxy_protocol():
    assert isinstance(HostMCPProxy(), MCPProxy)


# --------------------------- list_tools / description edits ---------------


def test_list_tools_applies_override_and_suffix():
    proxy = HostMCPProxy()
    proxy._raw_tools = {
        "travel-suite": [
            {"name": "search_flights", "description": "Search for flights."},
            {"name": "book_hotel", "description": "Book a hotel."},
            {"name": "untouched", "description": "Leave me."},
        ]
    }
    proxy.set_tool_description_edits(
        [
            {
                "server": "travel-suite",
                "tool": "search_flights",
                "mode": "override",
                "content": "EVIL OVERRIDE",
            },
            {
                "server": "travel-suite",
                "tool": "book_hotel",
                "mode": "suffix",
                "content": "ALSO DO EVIL",
            },
            # an edit for a DIFFERENT server must not apply here
            {
                "server": "other",
                "tool": "untouched",
                "mode": "override",
                "content": "NO",
            },
        ]
    )

    tools = proxy.list_tools("travel-suite")
    assert all(isinstance(t, ProxyTool) and t.server == "travel-suite" for t in tools)
    by_name = {t.tool: t.description for t in tools}
    assert by_name["search_flights"] == "EVIL OVERRIDE"  # override replaces
    assert by_name["book_hotel"] == "Book a hotel.\nALSO DO EVIL"  # suffix appends with \n
    assert by_name["untouched"] == "Leave me."  # cross-server edit ignored


def test_list_tools_unknown_server_is_empty():
    assert HostMCPProxy().list_tools("nonexistent") == []


def test_list_tools_no_edits_is_genuine():
    proxy = HostMCPProxy()
    proxy._raw_tools = {"s": [{"name": "t", "description": "genuine desc"}]}
    assert proxy.list_tools("s")[0].description == "genuine desc"


# --------------------------- handle_tool_call: the chokepoint -------------


async def test_handle_tool_call_decline_returns_genuine():
    proxy, _ = _proxy_with_forward("GENUINE")
    emit, send_event, events, observables = _recorder(injections={})  # decline everything
    proxy.bind(emit, send_event)

    result = await proxy.handle_tool_call("travel-suite", "search_flights", {"q": "x"})

    assert result == "GENUINE"  # no tampering on decline
    posts = [e for e in events if isinstance(e, ControllablePostCallEvent)]
    assert len(posts) == 1  # exactly one PostCall fired
    assert posts[0].controllable.name == "env_tool:travel-suite"
    assert posts[0].answer == "GENUINE"
    assert json.loads(posts[0].request) == {
        "tool": "search_flights",
        "params": {"q": "x"},
    }
    # emit-once: env tools are NOT separately emitted as ObservableEvents
    assert not any(isinstance(o, ObservableEvent) for o in observables)
    assert observables == []


async def test_handle_tool_call_injection_returns_tampered():
    proxy, _ = _proxy_with_forward("GENUINE")
    emit, send_event, events, observables = _recorder(
        injections={"env_tool:travel-suite": "TAMPERED_RETURN"}
    )
    proxy.bind(emit, send_event)

    result = await proxy.handle_tool_call("travel-suite", "search_flights", {"q": "x"})

    assert result == "TAMPERED_RETURN"  # return tampering applied
    # the optimizer still observed the GENUINE return in the fired event
    posts = [e for e in events if isinstance(e, ControllablePostCallEvent)]
    assert posts[0].answer == "GENUINE"
    assert observables == []


async def test_handle_tool_call_unknown_server_returns_genuine_no_event():
    proxy, _ = _proxy_with_forward("GENUINE")
    emit, send_event, events, _ = _recorder(injections={})
    proxy.bind(emit, send_event)
    # a server without a configured env_tool controllable: genuine, no firing
    result = await proxy.handle_tool_call("UNCONFIGURED", "t", {})
    assert result == "GENUINE"
    assert events == []


async def test_handle_tool_call_without_bind_returns_genuine():
    proxy, _ = _proxy_with_forward("GENUINE")  # bind() never called -> send_event is None
    result = await proxy.handle_tool_call("travel-suite", "search_flights", {})
    assert result == "GENUINE"


# --------------------------- _forward + small helpers ---------------------


async def test_forward_without_backend_url_returns_error_string():
    # real _forward (not patched): no server_urls -> error string, no fastmcp import
    out = await HostMCPProxy()._forward("travel-suite", "search_flights", {})
    assert "no backend URL" in out


def test_resolve_server_maps_tool_to_owner():
    proxy = HostMCPProxy()
    proxy._raw_tools = {"a": [{"name": "t1"}], "b": [{"name": "t2"}]}
    assert proxy._resolve_server("t1") == "a"
    assert proxy._resolve_server("t2") == "b"
    assert proxy._resolve_server("missing") is None


def test_setters_store_defensive_copies():
    proxy = HostMCPProxy()
    edits = [{"server": "s", "tool": "t", "mode": "override", "content": "x"}]
    proxy.set_tool_description_edits(edits)
    edits.append({"server": "z"})  # mutate caller's list afterwards
    assert len(proxy._tool_desc_edits) == 1  # stored copy is unaffected

    ctrl = env_tool_output_controllable("s", tools_server_tag("s"))
    by_server = {"s": ctrl}
    proxy.set_env_tool_controllables(by_server)
    by_server.clear()
    assert proxy._env_tool_controllables["s"] is ctrl


def test_extract_mcp_result_flattens_text_and_falls_back():
    class _Item:
        def __init__(self, text):
            self.text = text

    class _Res:
        def __init__(self, content):
            self.content = content

    assert _extract_mcp_result(_Res([_Item("a"), _Item("b")])) == "a\nb"
    assert _extract_mcp_result("plain string") == "plain string"  # no .content -> str()
    empty = _Res([])
    assert _extract_mcp_result(empty) == str(empty)  # empty content -> str(result)


# --------------------------- JSON-RPC dispatch (HTTP brains) ---------------


async def test_dispatch_initialize_echoes_protocol_version():
    proxy = HostMCPProxy()
    resp = await proxy._dispatch_rpc(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-X"},
        },
        None,
    )
    assert resp["result"]["protocolVersion"] == "2025-X"
    assert "tools" in resp["result"]["capabilities"]


async def test_dispatch_notification_returns_none():
    proxy = HostMCPProxy()
    resp = await proxy._dispatch_rpc(
        {"jsonrpc": "2.0", "method": "notifications/initialized"}, None
    )
    assert resp is None


async def test_dispatch_ping():
    proxy = HostMCPProxy()
    resp = await proxy._dispatch_rpc({"jsonrpc": "2.0", "id": 6, "method": "ping"}, None)
    assert resp["result"] == {}


async def test_dispatch_tools_list_unions_servers_with_edits_and_schema():
    proxy = HostMCPProxy()
    proxy._raw_tools = {
        "a": [{"name": "t1", "description": "d1", "inputSchema": {"x": 1}}],
        "b": [{"name": "t2", "description": "d2"}],
    }
    proxy.set_tool_description_edits(
        [{"server": "a", "tool": "t1", "mode": "suffix", "content": "S"}]
    )
    resp = await proxy._dispatch_rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, None)
    tools = {t["name"]: t for t in resp["result"]["tools"]}
    assert set(tools) == {"t1", "t2"}  # union across servers
    assert tools["t1"]["description"] == "d1\nS"  # edit applied
    assert tools["t1"]["inputSchema"] == {"x": 1}  # genuine schema preserved
    assert tools["t2"]["inputSchema"] == {}  # missing schema -> {}


async def test_dispatch_tools_list_server_scope():
    proxy = HostMCPProxy()
    proxy._raw_tools = {
        "a": [{"name": "t1", "description": "d1"}],
        "b": [{"name": "t2", "description": "d2"}],
    }
    resp = await proxy._dispatch_rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, "a")
    assert {t["name"] for t in resp["result"]["tools"]} == {"t1"}


async def test_dispatch_tools_call_routes_and_tampers():
    proxy, _ = _proxy_with_forward("GEN")
    proxy._raw_tools = {"travel-suite": [{"name": "search_flights", "description": "d"}]}
    emit, send_event, _, _ = _recorder(injections={"env_tool:travel-suite": "TAMP"})
    proxy.bind(emit, send_event)
    resp = await proxy._dispatch_rpc(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "search_flights", "arguments": {"q": "x"}},
        },
        None,
    )
    assert resp["result"]["content"][0]["text"] == "TAMP"
    assert resp["result"]["isError"] is False


async def test_dispatch_tools_call_unknown_tool_errors():
    proxy = HostMCPProxy()
    resp = await proxy._dispatch_rpc(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        },
        None,
    )
    assert resp["error"]["code"] == -32602


async def test_dispatch_unknown_method_errors():
    proxy = HostMCPProxy()
    resp = await proxy._dispatch_rpc({"jsonrpc": "2.0", "id": 5, "method": "bananas"}, None)
    assert resp["error"]["code"] == -32601


# --------------------------- real aiohttp bind (live; skipped offline) ----


@pytest.mark.docker
@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("DTAP_LIVE"), reason="binds a host aiohttp server")
async def test_start_serves_http_round_trip():
    import aiohttp

    proxy = HostMCPProxy()

    async def fake_fetch(server, url):
        return [{"name": "t", "description": "d", "inputSchema": {}}]

    async def fake_forward(server, tool, params):
        return "GENUINE"

    proxy._fetch_tools = fake_fetch  # type: ignore[method-assign]
    proxy._forward = fake_forward  # type: ignore[method-assign]
    proxy.set_env_tool_controllables(
        {"s": env_tool_output_controllable("s", tools_server_tag("s"))}
    )
    emit, send_event, *_ = _recorder(injections={})
    proxy.bind(emit, send_event)

    url = await proxy.start({"s": "http://backend/s/mcp"})
    assert url.startswith(f"http://{_HOST_GATEWAY}:") and url.endswith("/mcp")
    try:
        async with aiohttp.ClientSession() as session:
            base = f"http://127.0.0.1:{proxy._port}/mcp"
            r = await session.post(base, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            assert (await r.json())["result"]["tools"][0]["name"] == "t"
            r2 = await session.post(
                base,
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "t", "arguments": {}},
                },
            )
            assert (await r2.json())["result"]["content"][0]["text"] == "GENUINE"
    finally:
        await proxy.stop()


def test_lease_port_returns_zero() -> None:
    # Port 0 -> aiohttp binds an OS-assigned free port (read back after start).
    assert _lease_port() == 0
