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
| `web_content` | `external_data` | PostCall, per `web_fetch`/`web_search` call | Live same-turn (`registerAgentToolResultMiddleware`). |
| `file_content` | `external_data` | PostCall, per `read` call | Live same-turn. |
| `shell_output` | `external_data` | PostCall, per `exec`/`process` call | Live same-turn. |
| `message_content` | `external_data` | PostCall, per `message` call | Live same-turn. |
| `web_content_transcript` | `external_data` | PostCall, per `web_fetch`/`web_search` call | Transcript/memory poison (next prompt). |
| `file_content_transcript` | `external_data` | PostCall, per `read` call | Transcript/memory poison (next prompt). |
| `shell_output_transcript` | `external_data` | PostCall, per `exec`/`process` call | Transcript/memory poison (next prompt). |
| `message_content_transcript` | `external_data` | PostCall, per `message` call | Transcript/memory poison (next prompt). |
| `persistent_memory` | `external_data` | PostCall, once per run (end-of-run) | Cross-session `MEMORY.md` write via `agents.files.set`. Needs `enable_persistent_memory=True`. |
| `model_system_prompt` | `model` | PreCall, pre-run | Needs the LLM proxy. Same-turn (applied before the run's first model call). |
| `model_response_injection` | `model` | PreCall, pre-run, applied to every model response | Needs the LLM proxy. **Same-turn** — spliced onto the wire before the agent sees the reply (see Design decisions). |

Live tool controllables (`web_content`, `file_content`, …) and transcript-poison
controllables (`*_transcript`) are separate threat models — see Design decisions.
`persistent_memory` is a third, opt-in cross-session threat model (real
`MEMORY.md` writes, not prompt hooks). All tool controllables need
`enable_tool_injection`.

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

## Design decisions

**Tool-output injection: live same-turn vs transcript poison.** OpenClaw exposes
two plugin seams (verified in `openclaw/openclaw`):

- **Live same-turn** — `registerAgentToolResultMiddleware` on the embedded
  `tool_result` path (`src/agents/embedded-agent-runner/extensions.ts`). The
  bundled `tokenjuice` plugin uses the same API. Mapped to `web_content`,
  `file_content`, `shell_output`, `message_content`.
- **Transcript / memory poisoning (next prompt)** — async `before_tool_call`
  stashes the optimizer decision; sync `tool_result_persist` splices it into
  the persisted transcript only (`session-tool-result-guard-wrapper.ts` →
  `transformToolResultForPersistence`). Mapped to `*_transcript` controllables.
  Surfaces on the next `target.run()` in the same session — poison-then-trigger
  (`assert_tool_injection_persisted_on_next_run`).

**Cross-session memory poisoning (`persistent_memory`).** Simon's grouped
end-of-run edit idea maps here: after each successful `run()`, the target
emits one `ControllablePostCallEvent` with the full session `chat.history`
(plus current `MEMORY.md` as `answer`). The optimizer's returned value is
written via the real `agents.files.set` RPC into `MEMORY.md` — a durable
file that survives `sessions.reset` and loads as system context in a fresh
session. This is **not** `before_prompt_build` / `enqueueNextTurnInjection`
(ephemeral prompt context). Distinct from `*_transcript` (same-session JSONL
tool-row poison). Aligned with SafeClawArena SPE and CIK-Bench `mem-long`.
Opt in with `enable_persistent_memory=True`.

For same-turn *model* output (not tool output), use `model_response_injection`.

**Model streaming: live relay + append, not buffer-then-forward.** OpenClaw's
real provider client always sends `stream: true`
(`packages/ai/src/providers/openai-completions.ts`) and expects
genuine incremental SSE from a real provider. `LLMProxy` relays each upstream
delta to the gateway live (not buffered-then-replayed) so the
`assistant_stream` observable keeps real token-by-token granularity — verified
live against Gemini: 3 incremental deltas direct vs. 1 batched delta when an
earlier version forced non-streaming upstream. `model_response_injection` is
appended as a trailing delta chunk before the stream's finish frame, so
injection is still same-turn even in the live-relay path. A buffer-then-forward
design would be simpler to implement (reuses the existing SSE assembler) and
would allow full response *replace* instead of append-only, at the cost of
time-to-first-token and `assistant_stream` granularity; not implemented since
no current use case needs full replace.

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
pytest                      # everything (mock + local live + docker, when available)
pytest -m docker            # Docker only — the production isolation path
pytest -m local_gateway     # local `openclaw gateway` CLI only — fast dev path
pytest -m "not docker and not local_gateway and not provider and not controller_e2e"  # mock only
```

`docker` and `local_gateway` (`pyproject.toml` markers) exercise the *same*
real gateway + real Node plugin + real host-side proxy/injection-server stack
against **both** managed runtimes; almost every scenario below is duplicated
across both suites via shared helpers in `test_support/scenarios.py` so
adding a new controllable's test coverage once covers both lanes. Treat
`docker` as the production/CI path (full isolation, matches how the target
should actually be deployed) and `local_gateway` as the fast local-dev path
(no container/pairing/volume-seed overhead; historically caught real bugs —
SSE relay, upstream URL normalization — before they were reproduced in
Docker).

Tests use an in-process `MockGateway` (no Node/Docker) in
`test_openclaw_integration.py`, exercising the full `Controller` pipeline,
the injection bridge, ws session helpers, and the reset lifecycle.

**Live** (`local_gateway`) tests in `test_openclaw_live.py` spawn the real
`openclaw gateway` CLI (skipped when the CLI is missing) and drive RPCs plus
a full managed `OpenClawTarget` run against a stub LLM upstream — no API keys
required. This includes `test_live_tool_injection_round_trip_through_real_plugin`,
which drives the real Node plugin (not `MockGateway`) through an actual tool call:
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

**Provider** tests are opt-in (``GEMINI_API_KEY`` required). Docker is the
canonical path for real-provider runs:

- ``test_docker_provider_live.py`` — full provider parity in Docker
  (agent turn, proxy injection, streaming, system prompt, live middleware).
- ``test_controller_provider_e2e.py`` — full Controller stack (Tier 3) in Docker.
- ``test_openclaw_provider_live.py`` — local CLI gateway (fast dev path only).

Run e.g. ``GEMINI_API_KEY=... pytest -m "provider and docker" -v``.

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

**Local/Docker parity.** Every stub-LLM gateway scenario in
``test_openclaw_live.py`` has a Docker counterpart in ``test_docker_smoke.py``
via shared helpers in ``test_support/scenarios.py`` (file/shell/web/message
injection — live same-turn and transcript poison — model response injection,
reset/teardown, managed Target run, alias round-trips, all-controllables, and
system-prompt proxy injection). The LAN device-identity remote-connect path
(``test_live_remote_path_grants_operator_scopes_via_device_identity``) stays
local-only: ``test_docker_connect_grants_operator_scopes`` covers the same
auth behaviour in the production container path.

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
