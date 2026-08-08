# ASSUMPTIONS: dtap_openclaw_target

Every deviation from upstream DecodingTrust-Agent (DTAP, `AI-secure/DecodingTrust-Agent`,
pinned commit `e0323a521ba4ef88f8e14c1eccf68d0a3d19a458`, Apache-2.0), specifically
its OpenClaw agent adapter (`agent/openclaw/src/{agent.py,utils.py,plugin_generator.py}`).

This module is **only the OpenClaw-specific slice** of the port. The agent-agnostic
machinery -- the security-domain forest, the five DTAP injection vectors, the env
activation by config, the host MCP proxy / Docker / injection lifecycle, the
emit-once observables, and the query surface the claim's out-of-band judge reads --
lives in the shared, frozen `dtap_scaffold` base
(`dtap_scaffold.agent_base.DtapAgentTarget`). `DtapOpenClawTarget` subclasses it and
implements only the four abstract hooks (`_agent_kind`, `_native_tool_deny`,
`_run_episode`, `_extract_trajectory`). The faithfulness claims here therefore cover
just: how OpenClaw is launched, how it is wired to the proxy/provider, the
native-tool policy, and how its session transcript is parsed.

This is a **bare runtime**: it exposes injection opportunities (via the base's
controllables) but performs **no injection by default** -- with no attacker, every
run is a clean OpenClaw episode. Attacks are an optimizer's concern.

## A. Execution model: host CLI -> isolated Docker container

- **A.1** Upstream runs OpenClaw as a **host subprocess**: `npm i -g openclaw`,
  then per turn `openclaw --profile <iid> agent --local --message <turn>
  --thinking <level> --session-id <id>` with `OPENCLAW_TRAJECTORY=1` and a
  per-profile `openclaw.json` (`agent.py:_run_openclaw_cli`,
  `_configure_openclaw_with_proxies`). The port preserves that **exact invocation**
  but runs it **inside a `node:24` container** (`docker/Dockerfile` +
  `docker/run_turns.mjs`). Reason: this port **enables** the agent's native
  `exec`/`fs` tools (B), which must execute against a disposable filesystem, never
  the host. The state directory is bind-mounted at `/state`; the host writes
  `openclaw.json`, `AGENTS.md`, skills, and `task.json` into it, then `docker run`
  lets the container entrypoint drive the turns. `HOME=/state`, so OpenClaw's
  per-profile config dir (`$HOME/.openclaw-<profile>/openclaw.json`, upstream's
  layout) lands under the bound volume.
- **A.2** One container **per episode** (one `_run_episode`/`docker run`), in a
  fresh `episode-<uuid>` subdir, so concurrent runs of one task never share state
  (upstream isolates with a per-task `--profile`; the container is a stronger
  boundary). The base discards and rebuilds the target per task, so no cross-task
  leakage.
- **A.3** The single Docker boundary is `driver._run_docker` (a thin
  `subprocess.run`), so the whole module is import- and unit-testable with **no**
  Node/OpenClaw/Docker present; the real `docker run` executes only on the
  `@pytest.mark.docker` path. The blocking call is offloaded with
  `asyncio.to_thread` so the controller's event loop is never blocked.
- **A.4** No Python from `agent/openclaw/` is vendored. Upstream's
  `OpenClawAgent`/`MCPProxyManager`/`StaticPluginGenerator` are **host-CLI
  orchestration** glued to the un-packaged `dt_arena` tree; the port reimplements
  only the parts it needs (config wiring + trajectory parse) against the scaffold
  contracts, so it depends on neither `dt_arena` nor a published OpenClaw package.
- **A.5** The OpenClaw npm version is **pinned** to `2026.6.10` (image tag
  `dtap-openclaw:openclaw-2026.6.10`; `driver.DEFAULT_IMAGE` + the Dockerfile
  `ARG OPENCLAW_VERSION`). The `--thinking` level defaults to **`off`**: live
  verification showed some models (e.g. Claude via the litellm provider) reject
  `medium` (`Thinking level "medium" is not supported ... Use one of: off`). `off`
  is the safe cross-model default; override via the constructor `thinking=` arg
  (validated against `off`/`minimal`/`low`/`medium`/`high`).
- **A.6** **Per-turn failures are non-fatal; the episode always completes.**
  `docker/run_turns.mjs` runs every turn and **logs-and-continues** on a failed turn
  (spawn error, non-zero openclaw exit), exiting `0` once the episode finishes --
  mirroring upstream `OpenClawAgent.run`, whose loop swallows a failed turn into
  `final_output` (`_run_openclaw_cli` returns `success:False` on a non-zero exit or
  timeout, never raising) and **always** calls `_generate_trajectory` afterwards.
  The driver (`run_openclaw_container`) likewise **never raises** on a non-zero or
  timed-out container exit: it logs and still returns the episode dir, so the
  (possibly partial) trajectory is extracted and `evaluate()` runs. This matters
  because DTAP judges re-query the **live** environment state: an attack that mutated
  state and then crashed or timed out is a success upstream, and would otherwise be
  lost here as a hard task-error (understated attack-success-rate). **Timeout
  granularity is a documented deviation:** upstream applies `OPENCLAW_TIMEOUT_SECONDS`
  (default 1000s) **per turn** (`agent.py:_run_openclaw_cli`); the port applies its
  `docker_timeout` (default 1000s) as a **whole-episode** backstop on the single
  `docker run`. For single-turn DTAP tasks these coincide; a multi-turn episode is
  bounded more tightly (all turns share one budget), and when the backstop fires the
  partial trace is still extracted.

## B. Native tools ENABLED (the headline deviation)

- **B.1** Upstream keeps OpenClaw's native tools **off for the `os-filesystem`
  domain** (`utils/agent_helpers.py:get_default_disallowed_tools` ->
  `OS_FILESYSTEM_OPENCLAW_DISALLOWED_TOOLS = ["group:fs", "group:runtime",
  "group:web", "group:memory", ...]`), applied via `tools.deny`
  (`agent.py:419-428`). This port **enables** native `exec`/`fs` by default via the
  `tools` **profile** (`openclaw.json` `tools.profile = "full"`), because it runs in
  a throwaway container (A) and because a real DTAP threat model treats the agent's
  own tools as a genuine attack surface. The disable is preserved as an explicit
  **config policy**, not a hidden default. (Live-verified against OpenClaw
  `2026.6.10`: the granular per-tool `{security, ask}` shape is rejected as invalid
  -- `tools.fs: Invalid input` -- so the profile form is the correct one.)
- **B.2** `_native_tool_deny(policy)` maps the base's `native_tools_policy`
  ConfigSpec: `"enabled"` (default) -> `[]` (deny nothing); `"disabled"` ->
  `OS_FILESYSTEM_DISALLOWED_TOOLS` = upstream's twelve-entry
  `OS_FILESYSTEM_OPENCLAW_DISALLOWED_TOOLS` (`utils/agent_helpers.py:30`, copied
  byte-faithfully as `UPSTREAM_OS_FILESYSTEM_DISALLOWED_TOOLS`) **plus four
  image-specific additions** (B.2.4); anything else -> a JSON list of explicit deny
  entries, passed to OpenClaw verbatim (B.2.2). The deny list feeds `openclaw.json`
  `tools.deny`, mirroring upstream's mechanism.
- **B.2.1 Fixed: the `"disabled"` list was mistranscribed and denied the wrong
  half.** It previously read `("exec", "fs")`. OpenClaw's enforcing deny matcher
  (`makeToolPolicyMatcher` over `CORE_TOOL_GROUPS`, `dist/tool-policy-match-*.js` and
  `dist/tool-catalog-*.js` in the pinned image `dtap-openclaw:openclaw-2026.6.10`)
  expands group keys (every key is prefixed `group:`) and glob-matches the result
  against each tool name. There is **no prefix matching and no validation error
  path**, so an unrecognised entry is silently ignored. Measured by running the image
  against a recording endpoint and reading the tool list it hands the model:

  | `tools.deny` | file / shell tools left alive |
  | --- | --- |
  | `[]` | `read write edit apply_patch exec process file_fetch file_write dir_list dir_fetch` |
  | `["exec", "fs"]` (the bug) | all of the above except `exec` |
  | upstream's twelve | `file_fetch file_write dir_list dir_fetch` (B.2.4) |
  | the shipped list | none |

  So the shipped setting stripped the shell and left **every file tool live** while
  looking configured, and the failure was silent in both directions (no error, and
  the missing shell made the setting look applied). `tests/test_target.py` resolves
  every entry against `tests/fixtures/openclaw_tool_policy.json` -- the image's
  ENFORCING group table plus the tool list it really offers -- rather than against a
  literal expected string, and keeps the `["exec", "fs"]` case as a regression guard.
  Refresh that fixture when the image is repinned. Note the image also carries a
  second, already-drifted group table used only by its policy-conformance doctor;
  asserting against that one would let a rename in the enforcing table go unnoticed.
  The sibling Claude Code target's list was always a faithful copy; only OpenClaw was
  affected.
- **B.2.2 Fixed: the advertised JSON deny list denied nothing.**
  `config_specs.NATIVE_TOOLS_POLICY` documents the slot as accepting `"enabled"`,
  `"disabled"`, **or a JSON deny list of native tool names**. The OpenClaw
  implementation was `if policy == "disabled": ... ; return []`, so a JSON deny list
  fell through to "deny nothing" without complaint. The JSON branch is now
  implemented, matching the Claude Code target, so the advertised contract holds on
  both. A value that is neither keyword nor a JSON list raises: this slot is set by
  the claim, not by an attacker, so a malformed one is a configuration error and
  must fail loudly rather than silently disable the denial. Entries are passed to
  OpenClaw verbatim, so each must be a form its matcher recognises (B.2.1); the
  caller owns that choice, and superred does not second-guess it.
- **B.2.3 Who sets the policy, and what it costs.** The DTAP-BENCH claim now sets
  this slot per task instead of leaving it at the construction default: `"disabled"`
  on `os-filesystem` **and `code`**, `"enabled"` on the other nine domains. Extending
  it to `code` is a deliberate divergence from upstream, which applies its deny list
  only to `os-filesystem`. Two consequences, both owned by
  `security_claim_dtap` ASSUMPTIONS D.5 to D.7: `code`-domain scores stop being
  comparable to published DTAP `code` numbers, and the superred `host_filesystem` /
  `host_code_execution` surfaces become structurally unusable on those two domains,
  so their cells must be reported NOT-APPLICABLE rather than 0 percent. DTAP's own
  four vectors are unaffected, because they inject into the environment container or
  the model's context, not into the agent container's filesystem.
- **B.2.4 Divergence: four image-specific entries on top of upstream's list.** The
  pinned image ships the `file-transfer` plugin (enabled by default,
  `dist/extensions/file-transfer/openclaw.plugin.json`) whose tools `file_fetch`,
  `dir_list`, `dir_fetch` and `file_write` belong to **no** core tool group, so all
  twelve upstream entries leave them alive: a complete read/write file toolset
  against the agent's own container. Upstream never had to cover them because it
  applied the list only on `os-filesystem`, where the residue changed nothing;
  applying it on `code` too (B.2.3) is what exposes the gap. Measured with only the
  twelve denied: the agent reached for `file_fetch` instead of `read`, got
  `gateway node.list requires credentials before opening a websocket`, read that as
  an authentication problem rather than a topology one, and stopped to ask the user
  for a token -- worse than the plain `ENOENT` the topology notice explains. The two
  halves of the list are kept as separate constants so the upstream half stays
  diffable against `utils/agent_helpers.py:30`.
- **B.3** Web search/fetch are **denied** for determinism, mirroring upstream's
  unconditional web-disable (`agent.py:392-405`, which sets
  `tools.web.search.enabled = false` / `tools.web.fetch.enabled = false`). That
  granular `tools.web.*.enabled` shape is **rejected** by OpenClaw `2026.6.10` (the
  same schema change that rejects the per-tool `{security, ask}` shape, B.1), so the
  port applies the equivalent via `tools.deny` (always including `group:web` -- the
  `_WEB_DENY` constant in `driver.py`), the same `group:web` identifier upstream
  itself denies for the `os-filesystem` domain
  (`utils/agent_helpers.py:OS_FILESYSTEM_OPENCLAW_DISALLOWED_TOOLS`). The **browser**
  is not configured: the image ships no Chromium, so the native browser tool cannot
  launch regardless (upstream's `browser.enabled = false` is moot here).

## C. Provider / LLM transport (LiteLLM proxy)

- **C.1** All inference is routed through the project's **LiteLLM proxy**, wired via
  `openclaw.json` `models.providers.litellm` (`baseUrl`/`apiKey` from the target's
  construction args) plus `agents.defaults.model.primary = "litellm/<model>"`. This
  mirrors upstream's litellm provider block
  (`agent.py:_configure_openclaw_with_proxies`, the `model.startswith("litellm/")`
  branch) but takes credentials from explicit constructor args rather than
  `${LITELLM_BASE_URL}`/`${LITELLM_API_KEY}` env interpolation.
- **C.2** Provider API defaults to `openai-completions` (the project's LiteLLM proxy
  is OpenAI-compatible; consistent with the other superred targets). Upstream's
  litellm/claude-opus runs used `anthropic-messages`; pass
  `provider_api="anthropic-messages"` to match. The model is a **construction
  concern**, never a config slot (the base enforces this); generation `temperature`
  is fixed per experiment and only written when set.
- **C.3 `maxTokens` must not exceed the model's completion cap (live-found).** OpenClaw
  sends the generated `models.providers.<p>.models[0].maxTokens` VERBATIM as the
  provider's `max_tokens`. A value above the model's real cap makes EVERY request fail
  400; OpenClaw then misclassifies that 400 as a context overflow, burns its three
  auto-compaction retries (which 400 identically), and ends the turn with no assistant
  message and no tool call while still exiting 0. The episode looks "completed" and
  scores 0.0, which is indistinguishable from a defended attack. Measured 2026-08: with
  the previous hardcoded 8192 against `openai/gpt-4o-2024-05-13` (cap 4096) every
  episode on this host was dead. `DEFAULT_MAX_TOKENS = 4096` is the completion floor
  across the GPT-4o / Claude / Gemini families in use, so it is the safe default;
  `max_tokens=` / `context_window=` on the constructor raise it for a model known to
  allow more. The failure is asymmetric (too low truncates one answer, too high kills
  the episode silently), hence the conservative floor.
  Provenance: upstream hardcodes a pair per provider BRANCH, each matched to the one model
  that branch serves (`agent/openclaw/src/agent.py`): litellm declares 200000/64000 for
  `claude-opus-4-6`, llama declares 128000/8192, and the `openai/` branch emits no models
  block at all. This port's previous 200000/8192 was a MIX of two branches, correct for
  neither. OMITTING both keys was TESTED and does NOT work: OpenClaw's zod schema marks them
  `.optional()` and its consumers guard with `typeof x === "number"`, yet a live episode with
  both absent reproduced the dead-episode signature (0 agent messages) while the same task
  with an explicit 4096 produced a real assistant turn. An explicit value is REQUIRED, so
  DERIVE it instead: `max_tokens=None` (the default) resolves the model's real cap via
  `litellm.get_max_tokens`, which is the same per-model table the proxy enforces, falling back
  to `FALLBACK_MAX_TOKENS = 4096` only when litellm does not know the model. Verified:
  gpt-4o-2024-05-13 -> 4096, gpt-4o-2024-08-06 -> 16384, claude-opus-4-6 -> 128000, unknown ->
  OMITTED (OpenClaw's stock 8192 applies), and an explicit `max_tokens=` still overrides.
  Checked against the planned victims: opus/haiku/sonnet 4.x 64000, gpt-5 128000, grok 256000,
  deepseek exactly 8192, qwen3-vl unknown. Since the clamp is one-directional, EVERY one of
  those is a no-op and behaves identically to stock OpenClaw; the mechanism only engages for a
  model that genuinely cannot take the 8192 request. The mechanism that makes this necessary:
  OpenClaw's `clampOpenAICompletionsMaxTokens` is a one-directional CEILING
  (`modelMaxTokens === void 0 || requested <= modelMaxTokens ? requested : modelMaxTokens`), so
  the field only ever pulls OpenClaw's own 8192 request DOWN; omitting it lets the unclamped
  8192 through, which is why omission fails despite the schema marking it optional.

## D. MCP wiring (env tools via the host proxy)

- **D.1** The env MCP servers are wired as `openclaw.json` `mcp.servers` entries
  (`transport: "streamable-http"`), exactly upstream's bundle-mcp approach
  (`agent.py:307-317`), **not** upstream's `StaticPluginGenerator` static-plugin
  route. The scaffold already discovers/serves env tools through one host
  `MCPProxy`; generating a per-task OpenClaw plugin would duplicate that and add a
  Node build step, so the simpler bundle-mcp path is used.
- **D.2** Upstream creates **one proxy per server** and points each `mcp.servers`
  entry at that server's URL. The scaffold exposes **one** host proxy fronting all
  servers (`MCPProxy.start(server_urls) -> one proxy_url`), routing by a trailing
  server path segment. `driver.mcp_server_url(proxy_url, server)` is the single
  place that convention is encoded (`f"{proxy_url}/{server}"`); if the proxy's
  routing changes it is a one-line edit. Tool-DESCRIPTION edits (the PreCall tool
  vector) are applied by the proxy, per the scaffold contract -- not here.

## E. Trajectory conversion (faithful parse, env/native split)

- **E.1** `trajectory.py` parses OpenClaw's `OPENCLAW_TRAJECTORY=1` session JSONL
  (the "openclaw-trajectory" runtime schema), reproducing upstream
  `OpenClawTrajectoryConverter._convert_runtime_trajectory_entries` /
  `_append_message_steps` (`utils.py`): the same event handling
  (`prompt.submitted` / `context.compiled` / `model.completed` /
  `session.ended`), the same cumulative-`messagesSnapshot` walk, and the **same
  dedup** of repeated assistant texts / tool-call ids / user messages across
  snapshots. The output is the scaffold's `TrajectoryArtifact` rather than the
  un-packaged `dt_arena` `Trajectory`; the `trajectory_json` field is rebuilt in the
  byte-equivalent DT-Arena schema (`task_info`/`traj_info`/`trajectory`, with the
  same step shapes and counts) by a tiny in-module builder.
- **E.2** **Env/native split** (a port requirement, not upstream): a tool call is
  classified as an **env/MCP** call iff its name is attributable to a configured
  MCP server (`resolve_server`: upstream's `workspace_<Server>_<tool>` plugin
  naming, plus `<server><sep><tool>` / `mcp<sep>...` bundle-mcp prefixing for any
  configured server name); everything else is **native**. Env calls are **excluded**
  from `native_tool_calls` because the host proxy already emitted one
  ObservableEvent + one PostCall per env call (re-emitting would double-count); the
  base emits only the native calls + the assistant messages, once each. The
  **full** trace (env + native + messages) is still assembled into `trajectory_json`
  for the OOB judge.
- **E.3** `agent_responses` is the per-turn final assistant text, segmented by
  `prompt.submitted` boundaries in the event stream, and is exposed on the base's
  `agent_responses` query slot. It is **not** forwarded to the judge: the OOB judge
  runner grades `final_response` + `trajectory_json` only
  (`dtap_scaffold.judge_runner.run_dtap_judge` explicitly does `del agent_responses`,
  mirroring upstream `eval/task_runner.py`, which passes only the final `response`);
  a judge that needs the whole run reads it from `trajectory_json`. `final_response`
  is the last assistant text.
- **E.4** A missing / empty / unreadable trace **degrades gracefully** to an empty
  artifact (the run yields a vacuous result rather than crashing the controller);
  upstream logs an error and yields `None`. The session JSONL is located by globbing
  the episode dir for the newest `*.jsonl` (the container writes it under
  `/state/traces`, the bound `OPENCLAW_TRAJECTORY_DIR`).

## F. Skills, system prompt, turns

- **F.1** Injected **skills** (the base's skill PreCall vector) are written as
  `<skills>/<name>/SKILL.md` and exposed via `openclaw.json`
  `skills.load.extraDirs`, matching upstream's external-skills mechanism
  (`agent.py:_setup_skills`, `skill_subpath=""`). The **system prompt** is written
  to `workspace/AGENTS.md`, which OpenClaw injects as a bootstrap file
  (`agent.py:286-302`). Multi-turn **instructions** are run one `--message` per
  turn (upstream's `run(user_inputs)` loop), under the same `--session-id` so
  context carries across turns.
- **F.2** `thinking` (OpenClaw `--thinking`) is a construction arg validated against
  upstream's `VALID_THINKING_LEVELS = ("off","minimal","low","medium","high")`
  (`agent.py:48`). The port's default is **`off`** (per A.5: some models reject
  `medium`), whereas upstream's default is `"medium"` (`agent.py:135`,
  `agent_kwargs.get("thinking_level", "medium")`); pass `thinking="medium"` to match
  upstream. `max_turns` is carried in `task.json` for record only (it bounds the
  conversation-turn count upstream; the superred run budget bounds re-runs).
- **F.3 `AGENTS.md` carries the runtime-topology notice below the system prompt
  (addition, not upstream).** `build_agents_md` emits `spec.system_prompt`, then
  `dtap_scaffold.system_specification.RUNTIME_TOPOLOGY_NOTICE`. Rationale in
  `dtap_scaffold` ASSUMPTIONS K; the OpenClaw-specific point is **placement**: the
  system prompt is an attacker injection point and `build_agents_md` is not, so
  putting the notice below it means a system-prompt override cannot remove the
  statement of fact. This is additive text the benchmark does not ship, so it is a
  deviation from an upstream-faithful replay and must be recorded with results.

## G. Out of scope for the target

- **G.1** The attack-success / task-success predicates and the env-state judge live
  in `dtap_scaffold.judge_runner` + the `security_claim_dtap` claim; this target only
  launches the agent and exposes the transcript + ground truth via the base's
  queries.
- **G.2** Upstream's debug printing, `thinking_level` validation errors aside, and
  the host-profile auth-token population (`helpers/auth_helpers.py`,
  `populate_openclaw_profile_auth`) are not ported: credentials reach OpenClaw
  through the LiteLLM provider block (C), not a paste-token auth profile.
