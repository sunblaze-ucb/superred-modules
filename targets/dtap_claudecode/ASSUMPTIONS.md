# ASSUMPTIONS: dtap-claudecode-target

The Claude Code concrete agent for the DTAP port. Source paper /
implementation: **DecodingTrust-Agent** (`AI-secure/DecodingTrust-Agent`), agent
backend `agent/claudesdk/` (`ClaudeSDKAgent`, `ClaudeSDKTraceProcessor`,
`ClaudeSDKTrajectoryConverter`, `MCPProxyServer`) and
`utils/agent_helpers.get_default_disallowed_tools`. The agent loop itself is the
Claude Agent SDK's; only the superred wiring and the transcript normalization are
ours.

The agent-agnostic surface (forest, controllables, observables, specs,
Docker/proxy/injection lifecycle, the query surface) lives in `dtap-scaffold` and
its `DtapAgentTarget` base; this package implements only the four per-agent hooks.
Deviations specific to those hooks are below.

## A. Faithfulness to the upstream Claude SDK backend

- **A.1 SDK driver.** `driver.py` reproduces the upstream `ClaudeSDKAgent` run
  loop: `ClaudeSDKClient(options=ClaudeAgentOptions(...))`, `connect()`, then per
  turn `await client.query(turn)` and `async for message in
  client.receive_response()`, recording each message. Option values match
  upstream where they apply: `permission_mode="bypassPermissions"`,
  `max_turns`, `model`, `system_prompt`, `cwd` (workspace), and
  `disallowed_tools` (the native deny list).
- **A.2 Transcript schema is byte-compatible.** The transcript records
  (`trace_start` / `user_input` / `message` / `error` / `trace_end`) and the
  serialized message/block shapes (`assistant`/`user`/`system`/`result`;
  `text`/`tool_use`/`tool_result`/`thinking`) reproduce
  `ClaudeSDKTraceProcessor`. The serializers use class-name duck typing rather
  than `isinstance` against the SDK types, so they are import-safe and unit-tested
  on the host without the SDK; the field names and shapes are identical.
- **A.3 Trajectory conversion mirrors `ClaudeSDKTrajectoryConverter`.** The
  DTAP-schema `trajectory_json` (`task_info` / `traj_info` / `trajectory`, with
  `user` / `agent` / `tool` steps, `send_message_to_user` for text blocks, and the
  `tool(k="v")` action string) reproduces the upstream converter, including
  `_parse_tool_name` (`mcp__server__tool`) and `_parse_tool_result`
  (json-decode + single-text unwrap).

## B. Deviations adapting the SDK backend to superred

- **B.1 Single MCP proxy server (`dtap_proxy`).** Upstream registers one in-SDK
  `MCPProxyServer` per env MCP server. In superred the env tools are fronted by
  the scaffold's **host** MCP proxy, exposed to the container as one HTTP MCP
  server named `dtap_proxy` (`mcp_servers={"dtap_proxy": {"type": "http", "url":
  proxy_url}}`). So in the transcript every env tool is `mcp__dtap_proxy__<tool>`.
  The agent reaches the host proxy via `host.docker.internal` (the container is
  started with `--add-host host.docker.internal:host-gateway`).
- **B.2 Proxied env tools are SKIPPED in the artifact.** Because the host proxy
  already observed each env-tool call (ObservableEvent) and fired the per-server
  env-tool PostCall controllable, `trajectory.convert` drops every
  `mcp__`-prefixed call from BOTH `native_tool_calls` and the `trajectory` steps
  (and the matching `tool_result`, paired by `tool_use_id`). Only the agent's
  NATIVE container tools and its non-tool messages remain -- the emit-once rule
  the base relies on. Consequence: the DTAP-schema `trajectory_json` produced
  here is the agent's own native trace; the env-tool trace lives on the superred
  trajectory (proxy events) and is recovered by the OOB judge from live env state
  (`env_ports`), not from this `trajectory_json`. (Upstream's converter keeps env
  calls because there is no separate proxy seam.)
- **B.3 `tool_use_id` pairing instead of positional.** The upstream converter
  pairs a `tool_result` with the immediately-preceding agent step. Because we drop
  proxy steps, that positional heuristic would mis-pair; we index every
  `tool_use` by id and look results up by `tool_use_id`, which is correct under
  interleaving and skipping. Behaviour is identical to upstream when no proxy
  tools are present.
- **B.4 Containerized, per-instance isolation.** Each episode runs in a fresh
  `docker run --rm` container with a per-instance mounted dir (task.json in,
  transcript + result out), instead of the upstream per-process Claude home
  (`CLAUDE_CONFIG_DIR` temp dir). This gives the stronger isolation the base's
  parallelism assumes (independent target instances), and credentials reach the
  CLI as `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_MODEL` env
  vars. The single `docker run` is isolated in the overridable `_docker_run` so
  the whole lifecycle is testable offline with a fake.
- **B.5 Per-turn `agent_responses`.** The base's query surface and the DTAP judge
  take a per-turn final-output list. The upstream converter exposes only a single
  `final_response`; we segment the transcript by `user_input` turn and take each
  turn's last assistant text (falling back to that turn's `ResultMessage.result`,
  then `""`). `final_response` is the last turn's entry, so it equals the upstream
  single final for the common case.

## C. The native-tool menu (calibration point)

`_native_tool_deny` returns the deny list; `driver._build_options` enables the
native menu via `allowed_tools = DEFAULT_NATIVE_TOOLS + ["mcp__dtap_proxy__*"]`
with the deny list subtracted through `disallowed_tools`. The upstream backend
leaves native tools at the CLI default (allow-all) and only sets
`disallowed_tools`; we list the full menu explicitly so the proxy glob can be
allowed alongside it. `DEFAULT_NATIVE_TOOLS` mirrors Claude Code's built-in
toolset and is the single place to update if the CLI's native tool names change.
The `"disabled"` deny list is upstream's
`OS_FILESYSTEM_CLAUDE_SDK_DISALLOWED_TOOLS` verbatim.

## D. Skill injection (known limitation)

The `skill` controllable still fires (the base handles it) and the chosen skills
are passed through in `task.json`, but `driver.py` does **not** yet materialize
them into the container's `.claude/skills/` (upstream's
`create_injected_skills_directory`). Skill-vector attacks against this target are
therefore not yet exercised; the other four DTAP vectors (system prompt, user
prompt, tool-description, environment) work fully. This is a deliberate scoped
limitation, not a silent gap.

## E. Out-of-band model client / budget

Like the AgentDojo and inspect-agent ports, the agent runs inference with its own
provider client (the Claude Code CLI talking to `ANTHROPIC_BASE_URL`), **not**
superred's `LLMClient`. Its token spend is out of band and uncounted against the
optimizer's budget. The model id is fixed at construction (not a config slot), so
neither the Task nor the attacker can change the agent's model; it is exposed
read-only via the `model_identity` observable (base).
