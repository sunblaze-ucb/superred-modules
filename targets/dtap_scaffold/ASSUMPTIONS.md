# dtap_scaffold -- assumptions and live-verification notes

`dtap_scaffold` is shared infrastructure (no upstream attacker to be faithful to);
this ledger records the non-obvious lifecycle/orchestration decisions, the ones
surfaced by running the real Docker + MCP stack, and the current live-verification
status per text domain.

## A. The env lifecycle replaces upstream's pool orchestrator

Upstream splits the work across `utils/mcp_manager.py` (starts one MCP server),
`utils/resource_manager.py` + `utils/task_executor.py` (the environment pool that
brings up Docker, leases ports, seeds, resets). `DockerEnvStack` collapses the
pool's role for one instance, so a few things upstream's pool does in the PARENT
process must be reproduced here:

- **A.1 `<ENV>_PROJECT_NAME` export (live-found).** Servers that `docker exec`
  into their env container (terminal/code, research, os-filesystem, ...) resolve
  the container name from `f"{env.upper().replace('-','_')}_PROJECT_NAME"` (mirrors
  `utils.compose_utils.get_project_name`) -> `{project}-{env}-env-1`. Upstream's
  pool sets these in the parent env before `mcp_manager` starts each server.
  `DockerEnvStack._launch` and `_run_setup` both export the full
  `<ENV>_PROJECT_NAME` map (each env's per-instance compose project) -- to the MCP
  servers, the injection servers, AND `setup.sh`. Without it the terminal MCP
  server raises `TERMINAL_PROJECT_NAME is not set`, and the os-filesystem / slack
  seeders SILENTLY skip seeding (`[WARN] ..._PROJECT_NAME not set`, rc=0) leaving
  the env unseeded.
- **A.2 MCP servers inherit the parent environment.** `_server_env` starts from
  `dict(os.environ)` (so `uv run python` / `python3` find PATH), then overlays the
  server's own listen port, the rendered `${VAR}` host ports, and the state +
  project-name overrides -- matching `mcp_manager._setup_environment`
  (`os.environ.copy()` + server `env`).
- **A.3 Crash diagnosability.** A server's stdout+stderr are captured to a per
  server log (`_spawn_process(log_path=...)`); a readiness timeout is re-raised
  with those log tails. This is how the missing-dep crashes below were found
  (otherwise they vanished into DEVNULL and became a 10-minute readiness hang; the
  ready timeout is now 150s, not 600s).
- **A.4 `calendar` health-wait cap (live-found).** `wait_healthy` mirrors upstream's
  `_wait_for_healthy` exactly, including that BOTH callers discard its result: a
  container that never reports healthy only costs wall-clock, it does not abort the
  run. `calendar-api` is such a container -- its upstream healthcheck runs `curl`,
  which is absent from `decodingtrustagent/google-calendar:api-latest`, so every probe
  exits -1 ("executable file not found in $PATH") and the check can never pass, while
  the service itself is up (`/health` -> 200). Measured 2026-08; it is the only
  unsatisfiable check among the 32 healthchecks in upstream's compose files (the three
  other `curl`-based ones do ship `curl`). Because it can never pass, `wait_healthy` does not exit
  early here the way it does elsewhere: it burns its ENTIRE budget and returns False,
  every time. Measured: 31.3s against a 30s budget for calendar, versus 16.7s for
  `travel` (a working healthcheck) on the same budget. So the 120s default was 120s of
  dead wait on each of the 28 malicious text-domain tasks that activate calendar (16
  `crm`, 12 `workflow`), and a larger timeout buys no extra chance of success.
  `env_registry._HEALTH_TIMEOUT_OVERRIDES` caps it at 20s, sized off the measured
  3.0-4.2s cold-boot-to-serving time and the 10s healthcheck interval, so it stays
  correct if upstream fixes the check (at which point the override should be deleted).
  An explicit env.yaml `health_timeout` still takes precedence, and no other env is
  affected. Deliberately NOT fixed with a compose overlay: we do not own upstream's
  compose file, and an overlay would diverge the container definition under test.

## B. Undeclared upstream server dependencies (the `[sdk]` extra)

The env MCP / injection servers are upstream Python that imports third-party
packages the SDK wheel does not pull. Each was found by launching the server and
reading its captured crash log; they are declared in the `sdk` optional extra:

- `fastmcp` -- EVERY env MCP server (`travel/mcp_server.py` and every sibling) and
  EVERY injection server (`injection_mcp_server/<env>/env_injection.py`) opens with
  `from fastmcp import FastMCP`; the host proxy's genuine call/list path imports it
  too. Without it, every env server crashes at import and `EnvStack.up()` raises
  `MCP servers failed to become ready` (confirmed live). It is a host dependency, not
  container-only.
- `ujson` -- env MCP servers (e.g. travel `mcp_server.py`) + the hospital injection
  server.
- `psycopg2-binary` -- the customer_service injection server (`import psycopg2` at
  module top).
- `beautifulsoup4` -- the finance MCP server (`server/extractor_simple.py` opens with
  `from bs4 import BeautifulSoup`); without it the finance server crashes at import and
  `EnvStack.up()` raises `MCP servers failed to become ready: finance` (confirmed live).
- `uv` -- the launch command for some servers (terminal, finance); must be on PATH.

**Launch interpreter (`python3` on PATH).** The env servers are spawned via the
`mcp.yaml` `python_executable` (`"python3"`) running the SDK's server script (e.g.
`dt_arena/mcp_server/travel/mcp_server.py`). The interpreter is resolved from PATH,
NOT `sys.executable`, so the environment holding these `[sdk]` deps + the SDK's
`dt_arena` package must be the one `python3` resolves to: activate the venv (or put
its `bin` first on PATH) before a live run, else the servers launch under a different
`python3` and fail to import `fastmcp`/`dt_arena`. (The offline suites mock every
server, so this only bites the Docker/live path.)

## C. Container networking

`docker compose up` is run per-env as project `dtap_{iid}_{env}`, with the leased
host ports exported as env vars for `${VAR}` substitution. Two networking shapes
appear and BOTH are driven by the same exported port vars:

- **published ports** (`${VAR}:container_port`) -- crm, medical, os-filesystem,
  research, telecom, travel, customer_service.
- **`network_mode: host`** with env-var-controlled bind addresses (e.g. mailpit
  `MP_UI_BIND_ADDR: 0.0.0.0:${GMAIL_UI_PORT}`) -- terminal, finance, legal, gmail,
  slack.

**macOS Docker Desktop caveat (verified on this machine, 29.5).** A
`network_mode: host` container binds inside the Docker VM, NOT the Mac host's
localhost: a throwaway host-net container is `Up` yet `curl localhost:<port>` from
the Mac is refused. So anything that reaches a host-net service over host-localhost
HTTP fails on macOS. Two access patterns therefore behave differently:

- **`docker exec` into the container** (terminal/code, research, os-filesystem MCP
  servers) -- works on macOS; this is why the host-net `terminal` env is fully
  green here.
- **host-localhost HTTP to the container** (the gmail/slack MCP servers reaching
  their backends, and the gmail/slack `setup.sh`/reset curls) -- fails on macOS,
  works on a Linux Docker host where `network_mode: host` IS the real host
  localhost. The experiments run on Linux, so this is a test-machine caveat.

## D. Live-verification status (real Docker, this machine)

Verified end to end earlier: the `travel` domain runs both agents (Claude Code +
OpenClaw) through the host MCP proxy with the OOB judge returning task_success.

A no-LLM env-lifecycle smoke (compose up + MCP/injection servers + probe) was then
run across all 11 text domains. **9/11 start cleanly** (env containers up, every
MCP server answering):

- START cleanly: code/terminal, crm, legal, medical, os-filesystem, research,
  telecom, travel, workflow. This includes `network_mode: host` domains (terminal,
  legal, workflow) and the gmail+slack-only domain (workflow), so host-networking
  and the shared service envs are not themselves the problem.

The 2 non-green domains are the macOS caveat above, NOT a port flaw:

- **customer-service** and **finance** fail in `setup.sh` at a `curl
  localhost:<GMAIL_AUTH_PORT>` -- diagnosed: the gmail `user-service` container is
  `Up (healthy)` (its internal healthcheck passes), but the Mac host cannot reach
  its host-net port. Same root cause makes the `legal`/`workflow` RESET-endpoint
  curls -- and the `crm` `setup.sh` slack curl -- fail. All five are gmail/slack-backed
  and exercise host-localhost HTTP to a `network_mode: host` service. On a Linux Docker host these resolve. (The original
  PR always gated the container e2e behind the `docker`/`live` markers and the
  experiments run on Linux, so this is a test-machine caveat, characterized and
  confirmed, not a regression.)

Reset (not gating `up`): 6/11 reset within the 90s observation window. The
non-gmail miss is the `code` domain, whose `terminal` env sets
`reset_script_timeout: 180` in `env.yaml`. The port was DROPPING that per-env
value and truncating the cold `docker exec` reset at a hardcoded 30s. This is now
fixed: `DockerEnvStack.reset` threads `EnvRegistry.reset_script_timeout(env)`
(180 for `terminal`, else a 60s default), matching upstream
`task_executor._reset_instance` (`env_def.get("reset_script_timeout", 60)`; the
30s in `reset_helpers`' signature is dead, always overridden by that caller).
Correction to an earlier note: the `superred` controller does NOT retry a failed
reset. `reset_ephemeral_state` runs between runs, and if it raises the controller
logs it, records the traceback on `TaskResult.error`, and ends the whole
multi-run task with `stop_reason="error"` (`core/controller.py`) -- so an
under-budgeted reset timeout would spuriously error the task, which is exactly
why the per-env timeout matters. (Upstream's own environment pool retries via
`reset_retries`/`reset_retry_delay`, but that pool is the orchestrator
`DockerEnvStack` replaces.) A reset failure still does not block the initial
`up()`, which seeds via `setup.sh` independently.

## E. The judge-model substitution reaches every LLM judge

DTAP graders subclass `BaseJudge` (`dt_arena/src/types/judge.py`), whose
`__init__` defaults `judge_model` to the placeholder `"gpt-5.4"`. Per-task judges
call `super().__init__(domain=...)` and never pass a model, so absent any fix
they call `"gpt-5.4"` -- which the LiteLLM proxy does not serve. Only the
`customer_service` domain reads the `JUDGE_MODEL` env var; every other domain's
LLM judge (in the text-only set, that is the 45 `travel` judges that call
`llm_judge_*(..., model=self.judge_model)`) would otherwise fail and score
`attack_success=False` even on a genuinely successful attack.

`judge_runner._CHILD_SOURCE` therefore installs `_install_judge_model_override()`
in the OOB judge subprocess before any judge runs: it rewrites ONLY the `"gpt-5.4"`
default to the configured `JUDGE_MODEL` and touches no judge logic (the single
permitted judge-MODEL change). Because judges import `BaseJudge` via both
`from dt_arena.src.types.judge import BaseJudge` and `from judge import BaseJudge`
(the SDK adds `dt_arena/src/types` to `sys.path`), the override also sets
`sys.modules["judge"]` to the one patched module so a single patch covers both
styles. The substitute model must be one the judge endpoint serves; the claim/
experiment passes it through `run_dtap_judge(judge_model=...)`.

A SECOND model seam exists in the `research` domain and is NOT covered by the
`BaseJudge` override: `gpt_score_report` (`dt_arena/utils/research/helpers.py`) is a
standalone `urllib` POST that reads `OPENAI_MODEL` (default placeholder `"gpt-5.1"`)
rather than `BaseJudge` / `JUDGE_MODEL`, so it would otherwise request the unserved
`gpt-5.1` and silently mis-score every `research` LLM-graded task `False`.
`_build_child_env` therefore ALSO exports `OPENAI_MODEL` = the configured judge model
(twin of the `JUDGE_MODEL` export), routing the research harm scorer to the served
model. With both the `BaseJudge` default patch and the `OPENAI_MODEL` export in
place, the substitution reaches every text-only LLM judge (the `BaseJudge`-based
graders and the `research` `OPENAI_MODEL` scorer). Endpoint/key are already
redirected for both via `OPENAI_BASE_URL` / `OPENAI_API_KEY`.

## F. The `host` trust boundary (superred-afforded, beyond DTAP's vectors)

DTAP itself enumerates four attack vectors (`dt_arena/src/types/task.py`:
prompt / tool / environment / skill). superred additionally exposes the target
MACHINE as its own trust boundary, because these agents (Claude Code, OpenClaw)
run with native OS tools in a container -- a locus that can realistically be
compromised (a poisoned image, a shared mount, a supply-chain artifact on disk, or
full RCE) independent of the MCP-backed `environment` services. This is a
framework-completeness addition, not a DTAP fidelity claim, and it is strictly
additive: a passthrough optimizer places no files and runs no code, so the machine
is exactly what DTAP would run.

- **F.1 Native tool CALLS stay observable-only.** The agent's own bash/edit/exec
  are observed under `agent_trace_tool_calls`, never tampered mid-run. The `host`
  tree is the ATTACKER's own write/execute access, a Controllable. The two are
  distinct surfaces (observe the agent vs. the attacker acting on the machine).
- **F.2 Two independent capabilities.** `host_filesystem` (place/add/edit/delete
  files before the run) and `host_code_execution` (arbitrary code) are siblings
  under `host`: code execution subsumes filesystem in raw capability, but they are
  scoped separately so an experiment can grant a disk-only foothold (a mounted
  document, no RCE) without granting full code execution. Holding `host` grants
  both.
- **F.3 The base owns the per-run workspace.** So the host surfaces can shape the
  machine BEFORE the agent launches, the base mints `self._run_dir` (with a
  `workspace/`) at the top of `run()`, applies file ops into it, runs the
  code-execution foothold against it, then hands it to the subclass, which mounts
  THAT dir as the agent's workspace (claudecode at `/dtap/workspace`, openclaw at
  `/state/workspace`). On a declined run the workspace is identical to before; it
  is created one step earlier, nothing else changes.
- **F.4 Code execution is a pre-agent, attacker-terminated LOOP.** `code_execution`
  fires a `ControllablePostCallEvent` repeatedly: each round's `answer` carries the
  previous command's combined output (empty first), an injection is a shell
  command run via `_exec_on_host`, and its output feeds the next round. The loop
  ends the instant the optimizer declines ("the attacker decides it doesn't need
  anymore"); `max_code_exec_rounds` (default 64) is only a runaway backstop for an
  optimizer that never declines, and the optimizer's LLM budget bounds it in
  practice. It runs as a foothold BEFORE the agent loop (not concurrently); files
  it writes to the shared workspace persist into the run. `_exec_on_host` runs the
  code in the agent's OWN image with the workspace mounted (entrypoint overridden
  to `sh`), so background processes it starts live in a separate short-lived
  container -- filesystem/workspace effects persist, in-memory daemons do not.
  Concurrent-with-agent execution would be a larger, separate design and is not
  implemented.
- **F.5 Filesystem ops are confined to the workspace.** `_apply_host_files` rejects
  any path that would escape `workspace/` (a `..` traversal is skipped), so the
  surface stays scoped to the machine-as-the-agent-sees-it.

## G. Per-server tool-return authorization trees (superred-afforded scoping)

DTAP treats every environment as one undifferentiated MCP server, and the
tamper-return surface was initially one `tools.<server>` tag per server. But a
real service's tools do not share one trust boundary: a public read, a
customer-authenticated booking, and a manager-only endpoint are corruptible by
different actors. superred therefore splits each `tools.<server>` tag into a
single-parent AUTHORIZATION TREE and scopes each tool's return-tamper event to
its node. This is a superred measurement affordance (finer threat models), NOT a
DTAP fidelity claim: the tools, their returns, and the judges are unchanged, and
scoping the whole server (`{tools.<server>}`) reproduces the original behavior.

- **G.1 Placement is by control-of-return, not by caller.** A tool sits at the
  node whose trust boundary an attacker must compromise to CONTROL WHAT THE TOOL
  RETURNS (the value that re-enters the agent's context), which is the
  least-privileged actor who can corrupt that return -- deliberately NOT "who is
  authorized to call it". A public read is corruptible by whoever controls the
  upstream public source; an append surface (add a note/review/draft) whose
  content a later read reflects back is corruptible by any low-privilege
  contributor, so it is a leaf UNDER the authoritative-read node it feeds, not a
  peer. Reads of an authoritative stored record are corruptible only by editing
  that record (the owning service), so they sit high.

- **G.2 Single-parent tree; the server root is the apex.** Tags form
  `tools.<server>` -> `tools.<server>.<node>` with each node having exactly one
  parent (the dominating node, or the server root for a top plane). `includes`
  grants a node plus its descendants, so a more-privileged (ancestor) node grants
  the lower-trust surfaces it dominates, siblings are isolated, and the server
  root grants everything. Where a service has no single dominating tool (e.g.
  `legal`: a public case-law plane and a private firm plane that cannot corrupt
  each other), the two planes are peers directly under `tools.<server>`, which is
  then a pure whole-server grant holding no tool of its own.

- **G.3 The trees are DATA, derived per service.** They live in
  `data/tool_trees.json` (one entry per server) and are built into cached
  `SecurityDomainTag`s by `tool_trees.build_server_tree`. Refining a single tool's
  placement is a one-line data edit, no code change. The shipped trees are a
  best-effort derivation from each service's real authorization structure and a
  handful of per-tool placements remain debatable (documented inline where
  relevant); because they are data, they can be corrected without touching the
  scaffold.

- **G.4 Unmapped tools fall back to the server root.** A tool not present in its
  server's tree -- a version-drifted tool name, or a dynamically-registered server
  whose live tool list is not fully enumerated ahead of time (e.g. `legal`
  resolves its tools at boot) -- resolves to the `tools.<server>` root tag: the
  conservative whole-server grant. So drift never silently under-scopes a tool
  (it errs toward the broadest tag), and the mechanism is robust to live tool-list
  changes.

## H. Superred-afforded catalogue surfaces beyond DTAP's four vectors

DTAP enumerates four injection vectors (prompt, tool-description, skill,
environment). The port exposes three additional attacker-facing surfaces that DTAP
does not itself enumerate; each is additive (a passthrough optimizer that declines
everything reproduces the unattacked upstream run) and superred-invented, recorded
here for auditability:

- **H.1 `attacker_context` observable (`ATTACKER_CONTEXT_TAG`, under `system`).**
  DTAP's `Attack.additional_information` (per-task ground-truth an upstream
  red-teamer is handed: the victim inbox an injected email must target, the
  whitelisted attacker sender accounts) is surfaced as a static, read-only
  Observable so an in-scope optimizer can read those routing facts. It is scoped on
  its OWN tag (not `detailed_system_specification`) so an experiment can grant the
  per-task briefing separately from the static design brief. It is NOT folded into
  the `Goal` (the `malicious_goal` stays byte-identical; golden-hash preserved), and
  it is read-only -- never a Controllable. Empty for tasks that carry no
  `additional_information` (238 crm malicious configs carry it, 149 indirect).

- **H.2 Tool-catalogue ADD / REMOVE (`TOOL_CATALOGUE_ADD_TAG` /
  `TOOL_CATALOGUE_REMOVE_TAG`, under `tool_catalogue`).** Alongside DTAP's
  tool-DESCRIPTION edit vector, the port lets an in-scope attacker register a fake
  tool (`tool_add`) or drop a tool from the listing the agent reads (`tool_remove`).
  A fake tool has no backend: when the agent calls it, the proxy fires a per-call
  PostCall tagged at `tool_catalogue_add`, so the same attacker that added it
  supplies the return (static `fake_return` fallback). An added tool whose name
  collides with a non-removed genuine tool REPLACES it in the listing (the fake is
  shown once, and `handle_tool_call` prefers the fake, so listing and call agree).
  `tool_remove` affects only the LISTING the agent reads (a normal agent never calls
  a tool absent from its listing); it is not a call-time block. These are additive
  and firing them is a choice -- the passthrough baseline registers nothing and
  drops nothing.

- **H.3 Not enforced: config `tool_blacklist`.** The per-server `tool_blacklist`
  in `Agent.mcp_servers` is deliberately NOT honoured: upstream SDK 0.2.12 parses
  but never consumes it (no `.py` references; `MCPServerConfig` has no such field),
  so enforcing it would shrink the agent's tool surface versus the unattacked
  upstream run. The port therefore ignores it, exactly as upstream does.

## I. Accepted non-substantial residuals (coverage-audit)

Two upstream behaviours the port does not fully reproduce. The coverage audit rated
both non-substantial; they are recorded here as accepted, bounded residuals (worth
fixing only if a future experiment makes them relevant):

- **I.1 `disable_reuse` recreate-on-reset (intra-task multi-run only).** Upstream
  RECREATES a `disable_reuse` env between reuses (its reset scripts are deemed
  insufficient for these stateful envs; `utils/task_executor._acquire_instances_for_task`).
  `DockerEnvStack.reset` runs the env's reset scripts and does NOT branch to
  `down()`+`up()` for a `disable_reuse` env. Bounded: of the six `disable_reuse` envs
  (`ecommerce`, `custom-website`, `windows`, `macos`, `gitlab`, `bigquery`) only
  `gitlab`/`bigquery` are in the text-only scope, and the divergence bites ONLY under
  `max_runs_per_task >= 2` (the optimizer retrying the same task) -- no shipped
  experiment configures multi-run, and single-run (the default) is already fully
  faithful (a fresh Target + stack per task recreates from scratch). Section A covers
  the cross-task pool collapse; this is the intra-task multi-run residual. Fix, if
  multi-run on gitlab/bigquery is ever run: branch `reset()` to `down()`+`up()` when
  `self._registry.disable_reuse(env)`.

- **I.2 Environment injections applied once up-front, not per user turn.** Upstream
  re-applies env injections per turn (`turn_id`-filtered) immediately before each
  `agent.run(turn_instruction)` (`eval/task_runner.py`); `agent_base._apply_env_injections`
  applies all env injections ONCE before the single episode. Faithful for the shipped
  dataset: all 6001 text-only configs have a single-string `task_instruction` (0
  multi-user-turn tasks), so upstream's per-turn scheduling path never activates. The
  up-front persistent write is visible to the end-of-run env-state judge, and live
  per-tool-call PostCall return-tampering still fires on every turn. Fix, if a future
  dataset ships multi-user-turn tasks with `turn_id`-scheduled env writes: thread a
  per-turn env-injection callback into the run loop.
