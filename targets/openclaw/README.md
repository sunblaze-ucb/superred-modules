# OpenClaw target

A superred [`Target`](../../../superred/src/superred/core/interfaces/target.py)
that wraps an **OpenClaw** agent (the personal-assistant / "CLI Claw" agent
driven over its Gateway WebSocket protocol) so optimizers can red-team it:
craft user messages, inject adversarial tool output, optionally intercept
model calls, and observe the agent's behavior.

Scope: code/terminal and Browser-Use scenarios. Computer Use is out of scope.
Security claims (e.g. SafeClawBench-grounded tasks) live in a **separate**
package/PR; this module is the target only.

## Architecture

```
                          superred Controller
                                  │  emit / send_event
                                  ▼
   ┌──────────────────────────  OpenClawTarget  ──────────────────────────┐
   │  controllables / observables · run() · reset · teardown               │
   │                                                                       │
   │   ws_client.py      runtime.py        injection_server.py   proxy_llm │
   │   (Gateway proto)   (managed daemon)  (plugin callbacks)    (model    │
   │        │                 │                    ▲             proxy)    │
   └────────┼─────────────────┼────────────────────┼────────────────┼─────┘
            │ ws://loopback    │ spawns             │ HTTP /hook      │ HTTP
            ▼                  ▼                    │                 ▼
     OpenClaw Gateway  ◄── openclaw gateway   plugin/index.js   upstream LLM
     (agent runtime)       (Node process)     (in the gateway)   provider
```

Components (`src/openclaw_target/`):

| File | Responsibility |
|------|----------------|
| `target.py` | The `Target`: declares controllables/observables, runs one agent turn, maps plugin hooks to events, owns lifecycle. |
| `ws_client.py` | Async client for the Gateway protocol: connect handshake, request/response dispatch, event streaming, the two-stage agent run. |
| `runtime.py` | Optional **managed** mode: starts/stops a local `openclaw gateway` Node process on a loopback port. |
| `factory.py` | `openclaw_target_factory(...)` → a `TargetFactory` that builds one target (and, when managed, one gateway) per task. |
| `plugin/index.js` | OpenClaw extension (`definePluginEntry`) that consults superred for tool-output injection. |
| `injection_server.py` | Python HTTP endpoint the plugin POSTs to; bridges into the live optimizer. |
| `proxy_llm.py` | Optional OpenAI-compatible proxy in front of the provider: records model calls and can inject the system prompt. |
| `constants.py` | Security-domain tags and defaults. |

## Run flow

1. `run()` ensures a connection (`_ensure_connected`): start injection server +
   LLM proxy (if enabled), start the managed gateway (if `managed`), connect,
   cache the tool catalog.
2. Apply pre-run config: `system_prompt_append` (→ `AGENTS.md`) and
   `workspace_files` via `agents.files.set`.
3. Optimizer controllables, in order: optional `model_system_prompt`
   (PreCall, proxy only), then `user_message` (PreCall).
4. Send the message via the **two-stage agent flow**: the `agent` RPC returns
   an `accepted` ack with a `runId`; we then block on `agent.wait` for the
   terminal result while forwarding live `chat` (assistant deltas) and
   `session.tool` (tool calls) events. Final text falls back to `chat.history`.
5. Mid-run, each intercepted tool call fires a `ControllablePostCallEvent`
   (tool-output injection) through the plugin → injection-server → optimizer
   bridge; the injected value is spliced into the persisted tool result.
6. Record `last_response` / `tool_calls` / `events` for the evaluator.

## Capabilities

Controllables (injection points):

| Controllable | Domain tag | When | Notes |
|--------------|-----------|------|-------|
| `user_message` | `user_input` | PreCall, once per run | Always on. |
| `web_content` | `external_data` | PostCall, per web_fetch/web_search call | Needs `enable_tool_injection`. |
| `file_content` | `external_data` | PostCall, per file read | Needs `enable_tool_injection`. |
| `model_system_prompt` | `model` | PreCall, pre-run | Needs the LLM proxy. |

Observables: `model_identity`, `system_description`, `tool_list`,
`system_prompt` (static, populated at `initialize()`); `assistant_stream`,
`agent_response`, `tool_call`, `model_request`, `model_response` (emitted
during a run). User messages and injection-point tool calls are recorded via
their controllables and are **not** double-emitted as observables.

Security domains (`constants.py`): `system` ⊃ {`user_input`, `external_data`,
`internal_context`, `tool_catalog`, `model`}.

## Usage

Managed gateway (one isolated gateway per task), driving the prompt-list baseline:

```python
from openclaw_target import openclaw_target_factory
from superred.core.controller import Controller

controller = Controller(
    optimizer_factory=lambda: MyOptimizer(),
    target_factory=openclaw_target_factory(
        managed=True,                # start a local `openclaw gateway` per task
        model_id="my-model",         # surfaced as a static observable
        enable_tool_injection=True,  # web/file tool-output injection
        provider_base_url="https://api.provider.com",  # enables the LLM proxy
        provider_api_key="...",
    ),
    security_claim=[...],            # tasks live in the claims package
    scope=frozenset({...}),
)
result = await controller.run()
```

External gateway: omit `managed=True`, pass `gateway_url=` and `auth_token=`.

Key knobs (constructor / factory):

- `agent_timeout_s` (default 600s) — max wall-clock per agent run.
- `enable_tool_injection` — expose the web/file tool-output controllables.
- `provider_base_url` / `provider_api_key` — enable + configure the LLM proxy.
- `reset_session_between_runs` (default `False`) — opt into per-run
  conversation isolation (see Lifecycle).
- `tool_policy` (config slot) — restrict the agent's tools via the gateway tool
  profile (applied at startup, not a runtime RPC).

## Lifecycle

The controller drives `configure_target` → (`run` → `reset_ephemeral_state`)\*
→ `teardown` per task, with a fresh target (and managed gateway) per task.

- `reset_ephemeral_state` clears **only** ephemeral per-run buffers
  (`last_response` / `tool_calls` / `events`, proxy records). Durable task
  state — planted files/`AGENTS.md` and the OpenClaw conversation/session — is
  **preserved**, per the `Target` contract (durable state is discarded only via
  a fresh `TargetFactory` instance between tasks). This keeps OpenClaw's durable
  session intact and enables poison-then-trigger attacks. Set
  `reset_session_between_runs=True` to wipe the conversation each run.
- `teardown` best-effort clears planted files, then stops the proxy, injection
  server, WebSocket client, and managed gateway.

## Permissions

The connect handshake requests `operator.read|write|admin`. `admin` is required
for `agents.files.set` (planting the system prompt / workspace files) and
`sessions.reset`.

## Testing

```bash
pytest                      # from this directory
```

Tests use an in-process `MockGateway` (no Node/Docker), exercising the full
`Controller` pipeline, the injection bridge, ws session helpers, and the reset
lifecycle. The managed `runtime.py` path is not covered by the mock suite.

## Known limitations / planned follow-ups

- **Isolation/parallelism**: the gateway currently runs as a local Node daemon.
  Running the whole gateway inside a fresh Docker container per run (full
  isolation + safe `concurrency>1`) is a planned follow-up; until then the
  managed runtime uses a fixed port and shared workspace, so prefer
  `concurrency=1`.
- **LLM proxy** is wired only in managed mode (the managed gateway is pointed at
  the proxy's base URL); an external gateway must be pointed at it manually.
- **Capability coverage**: messaging, memory, skill/plugin, and exec/shell
  controllables/observables are planned before experiments; the hook/registry
  structure is designed to add them additively.
- **Live tool catalog** populates on connect; static observables otherwise come
  from config so they are non-empty at `initialize()`.
