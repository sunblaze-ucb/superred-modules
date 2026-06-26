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

Reset (best-effort, not gating `up`): 6/11 reset within 90s. The non-gmail reset
miss is `code`, hitting the faithful upstream 30s `script_timeout` on a cold
`docker exec` reset; resets are retried by the controller between runs and are not
required for the env to come up.
