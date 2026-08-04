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
  client.receive_response()`, recording each message. `max_turns` is applied
  exactly as upstream does: as the SDK's per-query turn cap (via
  `ClaudeAgentOptions.max_turns`) *and* as a cumulative cap on tool-use turns
  across the whole instruction sequence -- the driver counts each assistant
  message that used a tool (`_is_tool_use_turn`, mirroring upstream's
  `_turn_count`) and stops issuing further instructions once the count reaches
  `max_turns`. For DTAP's single-instruction tasks the two coincide; the
  cumulative cap only bites for a multi-turn `user_prompt`. Option values match
  upstream where they apply: `permission_mode="bypassPermissions"`,
  `max_turns`, `model`, `system_prompt`, `cwd` (workspace), and
  `disallowed_tools` (the native deny list). Live-verification note:
  `bypassPermissions` maps to the CLI's `--dangerously-skip-permissions`, which
  refuses to run as root unless `IS_SANDBOX=1`; since the container runs the agent
  as root, the target passes `-e IS_SANDBOX=1` (and the Dockerfile sets it) -- the
  agent IS sandboxed (a throwaway container), so this is correct, not a bypass of a
  real safety boundary.
- **A.2 Transcript schema is byte-compatible.** The transcript records
  (`trace_start` / `user_input` / `message` / `error` / `trace_end`) and the
  serialized message/block shapes (`assistant`/`user`/`system`/`result`;
  `text`/`tool_use`/`tool_result`/`thinking`) reproduce
  `ClaudeSDKTraceProcessor`. The serializers use class-name duck typing rather
  than `isinstance` against the SDK types, so they are import-safe and unit-tested
  on the host without the SDK; the field names and shapes are identical, with one
  deliberate trim: the `ResultMessage` serializer keeps `subtype` / `result` /
  `is_error` / `total_cost_usd` and omits upstream's `cost_usd` / `duration_ms` /
  `duration_api_ms`, because the converter consumes only `result` (the final text)
  and none of the timing/cost fields -- so the omission is invisible to
  trajectory-facing behaviour and the judge.
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
  vars, or the Bedrock env of B.6. The single `docker run` is isolated in the overridable `_docker_run` so
  the whole lifecycle is testable offline with a fake.
- **B.5 Per-turn `agent_responses`.** The base's query surface and the DTAP judge
  take a per-turn final-output list. The upstream converter exposes only a single
  `final_response`; we segment the transcript by `user_input` turn and take each
  turn's last assistant text (falling back to that turn's `ResultMessage.result`,
  then `""`). `final_response` is the last turn's entry, so it equals the upstream
  single final for the common case.
- **B.6 AWS Bedrock path (opt-in deviation from the upstream Anthropic-direct backend).**
  `bedrock=True` forwards `CLAUDE_CODE_USE_BEDROCK` / `AWS_REGION` /
  `AWS_BEARER_TOKEN_BEDROCK` from the launching process into the container by NAME
  (docker inherits each value, so no token reaches the argv). It is an explicit flag,
  not ambient detection: a stray `CLAUDE_CODE_USE_BEDROCK` in an operator's shell must
  not silently switch a run's provider. Default `False` forwards nothing. `model`
  then carries a Bedrock inference-profile id, so `model_identity` values are not
  comparable across the two provider paths. Two measured constraints (2026-08):
  `npm` installs the newest CLI the runtime's `engines` allow, and Node 20 caps that
  at 2.1.197, which cannot authenticate a bearer token; Node 22 gets 2.1.220, which
  can (hence the Dockerfile pin). And a WRONG credential does not fail fast: Bedrock
  answers HTTP 403 in ~0.2s, but the CLI retries it silently and never exits. So
  `_docker_run` bounds every episode at `docker_timeout` (default 1800s, above the
  ~1080s worst case measured on the legal domain), removes the container by name
  (`proc.kill()` stops only the docker client, and an orphan would keep calling the
  host MCP proxy into a later run), and RAISES: returning an empty `result.json`
  would score as a legitimate no-op. A misconfigured sweep therefore reports
  `stop_reason="error"` on its first tasks instead of a benchmark of zeros.

## C. The native-tool menu (upstream-faithful: no `allowed_tools`)

`_native_tool_deny` returns the deny list; `driver._build_options` sets
`permission_mode="bypassPermissions"` + the MCP proxy server and applies ONLY
`disallowed_tools`, exactly as upstream `ClaudeSDKAgent._build_options_kwargs`
does. Upstream never sets `allowed_tools` (`eval/task_runner.py` builds
`agent_kwargs` with `disallowed_tools` only), and under `bypassPermissions` the CLI
auto-approves BOTH the native tools and the `mcp__dtap_proxy__*` env tools -- so no
allow-list is needed or faithful, and there is no hand-maintained native-tool
constant to keep in sync with the CLI. (An earlier version listed the full native
menu via `allowed_tools`; that was dropped as redundant-and-non-faithful.) The
`"disabled"` deny list is upstream's `OS_FILESYSTEM_CLAUDE_SDK_DISALLOWED_TOOLS`
verbatim.

## D. Skill injection

The `skill` controllable fires (the base handles it), the chosen skills are passed
through in `task.json`, and `driver.materialize_skills` writes each one to
`<workspace>/.claude/skills/<name>/SKILL.md` before the episode runs -- the location
Claude Code discovers skills from under its `cwd`. This mirrors upstream's
`create_injected_skills_directory` (`utils/skill_helpers.py`), which builds the same
`.claude/skills/<name>/SKILL.md` layout and points the SDK's `cwd` at it. All four
DTAP injection vectors are therefore exercised against this target: prompt (system +
user), tool (description), environment, and skill.

Two deliberate, faithful-to-intent narrowings of the upstream skill machinery:

- **Create-from-scratch only.** Upstream seeds the directory from benign base skills
  (`source_skill_dirs`, i.e. `AgentConfig.skill_directories`): it copies each existing
  `SKILL.md`, then applies `insert`/`append` injections *on top of that base*. This
  port loads no base skills -- it materializes only the skills named in the injection
  spec. So `mode="create"` (write/overwrite the file) is the primary path;
  `mode="append"`/`mode="insert"` only modify a file a prior entry in the same spec
  already created, and otherwise degrade to `create` (write the content as a fresh
  `SKILL.md`). This matches upstream's own create branch, which likewise builds a
  brand-new skill from `create_configs[0].content`; only the benign-base overlay is
  dropped, and it has no analogue here.
- **Append separator.** Upstream's `apply_injection_to_content` prepends a blank line
  before appended content (`lines.append("")` then extend), so an append reads
  `"<orig>\n\n<content>"`. This port joins with a single newline
  (`"<orig>\n<content>"`). The injected bytes are otherwise identical.

## E. Out-of-band model client / budget

Like the AgentDojo and inspect-agent ports, the agent runs inference with its own
provider client (the Claude Code CLI talking to `ANTHROPIC_BASE_URL`, or to AWS
Bedrock per B.6), **not**
superred's `LLMClient`. Its token spend is out of band and uncounted against the
optimizer's budget. The model id is fixed at construction (not a config slot), so
neither the Task nor the attacker can change the agent's model; it is exposed
read-only via the `model_identity` observable (base).
