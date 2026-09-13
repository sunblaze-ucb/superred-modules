# superred-target-mcp-agent

An **MCP (Model Context Protocol) tool-using LLM agent** as a superred target —
built to red-team the **tool supply chain**.

A malicious or compromised MCP server can poison a tool's *advertised
description* with hidden instructions the agent reads and follows ("tool
poisoning" / indirect injection via the tool supply chain). This target connects
an LLM agent to an MCP server, lists its tools, runs a bounded tool-using loop,
and exposes the poison as an attacker-controllable surface. The paired
[`superred-claim-mcp-tool-injection`](../../security_claims/mcp_tool_injection)
scores whether the poison makes the agent call a sensitive tool it should not.

## What it adds, and what the CI test proves

The target speaks the **real MCP protocol** via the SDK's public `mcp.Client`, so
it can point at real third-party MCP servers (stdio / streamable-HTTP) with
server-advertised, dynamically-discovered tools — the actual supply chain an
MCP-using agent trusts. That real-protocol connection is the distinctive part
(tool-description rewriting on its own is also expressible on the shared
`inspect_agent` target; the MCP wire protocol and external servers are what this
adds).

It takes a **session provider** (a zero-arg callable returning an async context
manager — a connected `mcp.Client`). For CI that is an **in-memory** MCP server
(a real `MCPServer` connected in-process, no network/subprocess): the whole
*plumbing* path — connect → discover tools → apply poison → agent loop →
`call_tool` → record — runs offline and is asserted end-to-end. Be precise about
what that verifies: it proves the **plumbing**, not the security outcome — the
agent's decision to follow the poison is made by a mock LLM in tests and needs a
real LLM (and, for the real supply chain, a real server) to measure an actual
attack-success rate.

## Usage

```python
from mcp_agent_target import mcp_agent_target_factory, build_demo_server, in_memory_session_provider

# Offline / CI: the bundled demo server (benign get_weather + sensitive exfiltrate_data)
factory = mcp_agent_target_factory(
    model="openai/gpt-4o-mini",
    session_provider=in_memory_session_provider(build_demo_server()),
    api_base=..., api_key=...,   # key held privately, never emitted
)

# Real MCP server over stdio:
from mcp_agent_target import stdio_session_provider
factory = mcp_agent_target_factory(
    model="openai/gpt-4o-mini",
    session_provider=stdio_session_provider("python", ["-m", "my_mcp_server"]),
    api_base=..., api_key=...,
)
```

Attacker surfaces (controllables): `tool_poison` (the injected tool-description
instruction — offered first, so `goal_passthrough` fills it) and `user_message`.
Config: `system_prompt`, `user_task` (the benign request), `poison_tool` (which
tool's description is poisoned).

## Queries

`last_response`, `tool_calls` (JSON of `{name, arguments, result, is_error}`),
`called_tool_names`, `transcript`, `turns`, `error`. HTTP/LLM/MCP failures are
recorded in `error` (never raised), so a claim can abstain.

## License

MIT. Uses the MIT-licensed [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
and litellm as dependencies; no third-party code is vendored.
