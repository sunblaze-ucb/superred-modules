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

## Uniquely e2e-verifiable in CI

The target takes a **session provider** (an async context manager yielding an
initialized `mcp.ClientSession`). For CI it is an **in-memory** MCP server — a
real `MCPServer` wired to a real `ClientSession` over in-process streams, no
network and no subprocess — so the whole path (connect → list tools → poison →
agent loop → `call_tool` → record) runs offline against a mock LLM. For live runs
it is a stdio or streamable-HTTP connection to a real server.

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
