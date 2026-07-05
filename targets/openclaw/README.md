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
| `runtime.py` | Optional **managed** mode: starts/stops a local `openclaw gateway` Node process on a loopback port. Shared port/readiness helpers. |
| `docker_runtime.py` | Managed mode in a **container**: runs the whole gateway in a fresh Docker container per task (full isolation, dynamic port). |
| `config.py` | Builds the grounded `openclaw.json` (model/provider routing, `tools.profile`, `plugins.allow`) and materializes the per-instance state dir + extension. |
| `factory.py` | `openclaw_target_factory(...)` → a `TargetFactory` that builds one target (and, when managed, one gateway) per task. |
| `plugin/index.js` | OpenClaw extension (`definePluginEntry`) that consults superred for tool-output injection. |
| `injection_server.py` | Python HTTP endpoint the plugin POSTs to; bridges into the live optimizer. |
| `proxy_llm.py` | Optional OpenAI-compatible proxy in front of the provider: records model calls and can inject the system prompt. |
| `constants.py` | Security-domain tags and defaults. |

The managed gateway is configured the way OpenClaw really expects: a per-instance
state dir (`OPENCLAW_STATE_DIR`) holds an `openclaw.json` whose
`models.providers.<name>` block points model calls at the LLM proxy
(`baseUrl` + `/v1`, `api: "openai-completions"`, `request.allowPrivateNetwork`),
plus `tools.profile` and `plugins.allow`; the injection extension is installed
under `<stateDir>/extensions/<plugin>`. (Provider/extension routing is **config**,
not env vars.)

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
   bridge; the injected value is spliced into the tool result as it is
   persisted to the session transcript. The real tool always executes for
   real. **Verified live against a real gateway (no mocks):** OpenClaw's
   embedded agent runner drives a same-turn tool-calling continuation (the
   provider call that immediately follows a `tool_calls` response) from its
   own in-memory buffer, not the transcript, so that in-flight continuation
   still sees the real tool output; the injected content is what every
   *subsequent* prompt submission in the same session (the next turn, a
   session resume, etc.) loads as history. This is tool-result poisoning
   that surfaces on a later turn — there is no documented OpenClaw hook that
   rewrites a tool result before the same tool-calling loop's own next
   provider call.
6. Record `last_response` / `tool_calls` / `events` for the evaluator.

## Capabilities

Controllables (injection points):

| Controllable | Domain tag | When | Notes |
|--------------|-----------|------|-------|
| `user_message` | `user_input` | PreCall, once per run | Always on. |
| `web_content` | `external_data` | PostCall, per `web_fetch`/`web_search` call | Needs `enable_tool_injection`. |
| `file_content` | `external_data` | PostCall, per `read` call | Needs `enable_tool_injection`. |
| `shell_output` | `external_data` | PostCall, per `bash`/`exec`/`process` call | Needs `enable_tool_injection`. |
| `message_content` | `external_data` | PostCall, per `message` call | Needs `enable_tool_injection`. |
| `model_system_prompt` | `model` | PreCall, pre-run | Needs the LLM proxy. |

Observables: `model_identity`, `system_description`, `tool_list`,
`system_prompt` (static, populated at `initialize()`); `assistant_stream`,
`agent_response`, `tool_call`, `model_request`, `model_response` (emitted
during a run). User messages and injection-point tool calls are recorded via
their controllables and are **not** double-emitted as observables.

Security domains (`constants.py`): `system` ⊃ {`user_input`, `external_data`,
`internal_context`, `tool_catalog`, `model`}.

**Adding capabilities.** Tool-output injection points are declared in the
`TOOL_OUTPUT_CONTROLLABLES` registry (gateway tool name → `Controllable`).
`get_controllables` and the plugin bridge both derive from it, so a new
injection point is a single entry (define a `Controllable` with the right
security domain and map its real gateway tool name(s)). Mapped tool names are
the verified OpenClaw identifiers: `web_fetch`/`web_search`, `read`,
`bash`/`exec`/`process`, `message`. Note: **memory** is an OpenClaw *plugin
slot* (`plugins.slots.memory`), not a tool, so it is configured via config
rather than registered as a tool-output controllable.

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

Containerised gateway (full isolation; safe to raise `concurrency`):

```python
target_factory=openclaw_target_factory(
    managed=True,
    managed_runtime="docker",     # whole gateway in a fresh container per task
    model_id="openai/gpt-5",      # provider-qualified id for the config
    enable_tool_injection=True,
    provider_base_url="https://api.provider.com",
    provider_api_key="...",
    concurrency=4,                 # each task gets its own container + port
)
```

Docker mode uses the official OpenClaw release image by default
(`ghcr.io/openclaw/openclaw:latest`; Docker Hub mirror: `openclaw/openclaw:latest`).
The runtime auto-pulls on first start when Docker is installed — no manual
`docker build` from an OpenClaw checkout is required. Pin a version with
`managed_kwargs={"image": "ghcr.io/openclaw/openclaw:2026.6.11"}` or set
`OPENCLAW_DOCKER_IMAGE`. Dev builds from source can still use `openclaw:local`.

In Docker mode the gateway runs `--bind lan` with a generated token, is reached
on a dynamic published port, and reaches the host injection server + LLM proxy
via `host.docker.internal` (the host servers bind `0.0.0.0`).

External gateway: omit `managed=True`, pass `gateway_url=` and `auth_token=`.

Key knobs (constructor / factory):

- `managed_runtime` — `"local"` (loopback Node subprocess) or `"docker"`.
- `agent_timeout_s` (default 600s) — max wall-clock per agent run.
- `enable_tool_injection` — expose the web/file/shell/message tool-output controllables.
- `provider_base_url` / `provider_api_key` — enable + configure the LLM proxy
  (written into `models.providers.*` in `openclaw.json`).
- `reset_session_between_runs` (default `False`) — opt into per-run
  conversation isolation (see Lifecycle).
- `tool_policy` (config slot) — restrict the agent's tools via `tools.profile`
  (config, applied at startup, not a runtime RPC).

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

Tests use an in-process `MockGateway` (no Node/Docker) in
`test_openclaw_integration.py`, exercising the full `Controller` pipeline,
the injection bridge, ws session helpers, and the reset lifecycle.

**Live** tests in `test_openclaw_live.py` spawn the real `openclaw gateway`
CLI (skipped when the CLI is missing) and drive RPCs plus a full managed
`OpenClawTarget` run against a stub LLM upstream — no API keys required. This
includes `test_live_tool_injection_round_trip_through_real_plugin`, which
drives the real Node plugin (not `MockGateway`) through an actual tool call:
a stub LLM response with an OpenAI-style `tool_calls` payload makes the real
agent loop invoke the real `read` tool, the real plugin's `before_tool_call`
hook POSTs to a real `InjectionServer`, and `tool_result_persist` splices the
injected content into the persisted transcript — verified by asserting a
*second* `target.run()` in the same session sees the poisoned tool result on
its next prompt submission (see the run-flow note above on why this doesn't
show up in the same tool-calling loop's own continuation).
`test_live_model_response_injection_through_real_proxy` and
`test_live_reset_and_teardown_against_real_gateway` cover model-response
injection and reset/teardown against the same real CLI process.

**Provider** tests in `test_openclaw_provider_live.py` are opt-in only: they
call a real Gemini endpoint and are skipped unless ``GEMINI_API_KEY`` is set.
Run with ``GEMINI_API_KEY=... pytest test_openclaw_provider_live.py -v``.
These cover direct provider turns, proxy response/system injection, live SSE
streaming, and a real model-driven tool-call plugin hook — no stub upstream.

**Tier 3** tests in `test_controller_provider_e2e.py` drive the same paths through
the full ``Controller`` → factory → managed target stack with a real Gemini
upstream (also opt-in). Run with
``GEMINI_API_KEY=... pytest test_controller_provider_e2e.py -v``.

Shared live-test helpers (stub LLM servers, Gemini/Docker factories, send-event
builders) live in ``test_support/``.

Config/command builders are covered in `test_runtime_config.py`. When Docker
is available, `test_docker_smoke.py` exercises real containers: operator
scopes over a published port (Ed25519 device identity + pre-seeded pairing),
injection plugin boot (`openclaw.plugin.json`), RPCs, a stub-LLM agent turn,
and (`test_docker_tool_injection_round_trip_through_real_plugin`) the same
real tool-call + plugin-hook round trip as the live suite, but with the
plugin executing *inside the container* and POSTing back to the host over
`host.docker.internal`. ``test_docker_openclaw_target_managed_run_with_stub_llm``
and ``test_docker_openclaw_target_tool_injection_round_trip`` exercise the
same Docker path through :class:`OpenClawTarget` / ``managed_runtime=\"docker\"``
(including the host-side LLM proxy wiring), not just
:class:`OpenClawDockerRuntime` directly. With ``GEMINI_API_KEY`` set,
``test_docker_openclaw_target_real_gemini_turn`` runs the same Target Docker
path against real Gemini. ``test_docker_concurrency.py`` exercises
``TargetFactory.concurrency=2`` with two parallel containerised gateways.

**Live** tests in ``test_openclaw_live.py`` also cover ``web_search`` and
``process`` tool-name aliases for the ``web_content`` and ``shell_output``
controllables (alongside ``web_fetch`` and ``exec``).

## Known limitations / notes

- **Isolation/parallelism**: `managed_runtime="docker"` runs the whole gateway
  in a fresh container per task with a dynamic host port and private state dir,
  so `concurrency>1` is safe. The default `"local"` runtime now also uses a
  dynamic port + private state dir, but shares host state/network, so keep
  local managed runs at `concurrency=1`.
- **Docker image**: defaults to the official release
  `ghcr.io/openclaw/openclaw:latest` (auto-pulled on first start). Override via
  `managed_kwargs={"image": ...}` or `OPENCLAW_DOCKER_IMAGE`. Pin a version tag
  for reproducibility. On Linux, bind-mounted state must be writable by the
  container's `node` (uid 1000) user.
- **LLM proxy** is wired in managed mode (the gateway's provider `baseUrl` in
  `openclaw.json` points at the proxy); an external gateway must be pointed at
  it manually.
- **Live tool catalog** populates on connect. Call
  ``await target.warmup_static_observables()`` from ``Task.configure_target``
  so it is available in ``get_observables()`` at optimizer init; it is also
  emitted at the start of each ``run()``. Other static observables come from
  config so they are non-empty at ``initialize()``.
