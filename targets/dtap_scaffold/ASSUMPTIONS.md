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

## B. Undeclared upstream server dependencies (the `[sdk]` extra)

The env MCP / injection servers are upstream Python that imports third-party
packages the SDK wheel does not pull. Each was found by launching the server and
reading its captured crash log; they are declared in the `sdk` optional extra:

- `ujson` -- env MCP servers (e.g. travel `mcp_server.py`) + the hospital injection
  server.
- `psycopg2-binary` -- the customer_service injection server (`import psycopg2` at
  module top).
- `uv` -- the launch command for some servers (terminal, finance); must be on PATH.

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
  curls fail. All four are gmail/slack-backed and exercise host-localhost HTTP to a
  `network_mode: host` service. On a Linux Docker host these resolve. (The original
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
