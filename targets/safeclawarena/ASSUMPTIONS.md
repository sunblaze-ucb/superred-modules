# Assumptions and deviations

## Upstream

| | |
| --- | --- |
| Project | [SafeClawArena](https://github.com/sunblaze-ucb/SafeClawArena) |
| Path | `Dockerfile*`, `tools/`, `scripts/reset_env.sh`, `configs/`, `scripts/judge.py` (execution half) |
| Commit | `a11f5cc` |
| Paper | Niu et al., arXiv:2606.30755 (2026) |
| Licence | MIT (© 2026 SafeClawBench Authors) |

## Vendored byte-for-byte

Under `src/safeclawarena_target/_vendor/safeclawarena/`: the two platform
`Dockerfile`s (openclaw, nemoclaw), `tools/sim-google/sim-google` (the 16-service
simulated Google Workspace CLI), `scripts/reset_env.sh` (the
per-task provisioner), and `configs/**` (platform + model + workspace baselines).
`scripts/sync_upstream.py --check` re-downloads the pinned commit and diffs every
vendored file.

**Not vendored:** `tools/ripgrep/rg` — a 6.3 MB prebuilt binary the Docker build
does not reference (verified: no `COPY`/`ADD` of it in any Dockerfile). Install
ripgrep via the image if a task needs it. Upstream `scripts/judge.py` is also not
vendored: its *judging* half is ported into the `safeclawarena-claim` package,
and its *execution/capture* half is ported into `runtime.py`.

## Split of upstream `judge.py`

Upstream `judge.py` is a monolith (provision → execute → capture → judge). This
port splits it along the superred target/claim boundary:

- **Target (`runtime.py`)**: platform config, image build, container lifecycle,
  provisioning via the vendored `reset_env.sh`, session execution, and post-run
  state capture, ported from upstream's `PLATFORMS`, `reset_environment`,
  `run_session`/`send_message` and `capture_post_state` with the differences
  below.
- **Claim (`safeclawarena_claim.judge`)**: the deterministic check evaluators
  and dimension scoring, as a pure function over the captured state.

## Differences in the execution half

Matching upstream:

- Each turn goes to the gateway's `/v1/chat/completions` with upstream's body,
  bearer token and `x-openclaw-session-key` / `x-openclaw-agent-id` headers.
- The main instruction gets upstream's `/root/.openclaw` path remap before
  sending; follow-ups are sent as written, as upstream does.
- A reply containing `malformed_function_call` (a Gemini backend quirk) is
  retried once after 2 s with the un-remapped instruction, as upstream does.
- Every follow-up reply is appended to the session's response after `"\n---\n"`,
  empty or not, so the output checks see upstream's joined text.
- After the API key is written into the auth profiles, the gateway is restarted
  so it picks the key up and must be healthy within 90 s, as upstream does after
  applying a model config.
- A gateway restart stops the old process with upstream's sequence: a graceful
  kill of both the `openclaw-gateway` and `openclaw gateway` process names, then a
  force kill of any survivor.
- `session_transcript`, which the `log_file` taint channel reads, is upstream's
  extraction: the tool calls and tool results in the five newest session files,
  cut to 500 and 1000 characters. The raw JSONL is `session_transcript_raw`.

Deliberately different:

- **Timeouts and gateway failures abstain.** Upstream turns a timed-out request
  into a `[TIMEOUT: ...]` reply that its checks then score, and carries on. Here
  any gateway failure (a transport error or timeout, a non-2xx status, an empty,
  non-JSON or non-object body, or an `error` field) raises, the run errors, and
  the claim abstains instead of scoring a turn the agent never completed.
- **An unhealthy gateway raises.** Upstream only warns when the gateway is not
  healthy 30 s after a session's `restart_gateway`; here the run errors (abstain).
- **A failed `reset_env.sh` raises.** Upstream warns and runs the task anyway on
  a half-provisioned environment.
- **The `/tmp` new-file scan uses a session sentinel.** Upstream finds new `/tmp`
  artifacts with `-newer /tmp/openclaw-gateway.log`, a file the gateway keeps
  appending to, which hides every file the agent creates during the sessions.
  This port compares against a marker file touched after provisioning and skips
  the gateway log itself, which would otherwise pour the whole transcript into
  `workspace_file_contents`.
- **The captured state never holds the API key.** The injected key sits in
  `auth-profiles.json` beside the task's canary profiles, a file many tasks tell
  the agent to dump. Upstream keeps whatever the agent printed in its result
  files; here every captured string has the key replaced with `[scrubbed]`, so
  neither `query("post_state")` nor the observables carry it. The judge reads
  the scrubbed text, so the replacement must match no check; a claim test runs
  all 406 tasks with and without the scrub and gets the same verdicts. Only a
  key's own characters can matter: the judge splits ipi-4.1-020's safe
  condition into literals that include `9`, so removing a key containing a 9
  changes that safe result, which no score reads. A key under 8 characters or
  made only of letters is a dummy (a local proxy's `EMPTY` or `anything`) and
  is left as is, since scrubbing it would rewrite ordinary text the judge reads.
- **`verify_preconditions` is not evaluated.** Upstream logs a failed
  precondition and proceeds anyway, so it never affects a score.
- **Session keys after a restart.** Upstream keeps the post-restart key for the
  sessions that follow; this port returns to the task's shared key. No task in
  the corpus has a non-restarting session after a restarting one, so the two
  never diverge on the shipped tasks.

Not delivered, by upstream or here: task environments carry `incoming_event`,
`multi_agent` and `webhook_config` fields, but neither `reset_env.sh` nor
`judge.py` acts on them, and upstream never detects a leak through the
`mcp_tool_input`, `webhook_payload` or `cron_output` channels. They are therefore
neither controllables nor observables.

The capture also records whether memory (`MEMORY.md` + `memory/*.md`) and the
scheduled jobs changed during the run, from per-file md5 listings taken after
provisioning and after the sessions. These feed the target's observables only;
the judge ignores them.

## End-to-end execution needs Docker

`run()` builds a platform image from the vendored Dockerfile, starts a container,
provisions the task, runs the sessions, and captures state — all via `docker`.
Like the `dtap_openclaw` target, that path is **not exercised in unit tests**
(no Docker/agent runtime in CI). The unit tests cover the pure contract: the
security-domain forest, config/query specs, controllables and observables
(including those built from a captured post-run state and their redaction),
config round-trip, the pure runtime helpers (`platform_config`,
`file_check_targets`, `http_route_targets`), and the runtime's request, turn,
restart, transcript and change-tracking logic with the container calls stubbed
out.

Transport note: sessions are sent via `docker exec … curl` to the in-container
gateway. Upstream uses host HTTP with an exec
fallback; driving over exec is the robust equivalent that avoids host↔container
network assumptions.

## superred adaptation

- **Trust-domain forest.** One leaf per store the agent reads from or acts on,
  following the superred security-domain guide: `system → {supply_chain →
  {skills, plugins}, persistent_state → {memory, config, cron}, response,
  model_identity, task_metadata}`, `user`, and `tools → {workspace_files,
  sim_google → 11 service leaves}`. Reading a store and the agent's actions on it
  share that store's leaf. The simulator's five services that no task uses
  (forms, people, classroom, admin, appscript) report at the `sim_google` node,
  keeping `distinct_combinations()` at 2,959,478 scopes.
- **Controllables.** The task's adversarial payload is baked into its
  environment (the benchmark's design). The target additionally exposes
  `user_message` (override the first session instruction) and `workspace_file`
  (plant one extra attacker-controlled workspace file); left un-injected, the
  benchmark runs as-shipped. The Sim-Google store leaves have no injector of their
  own, so a scope made of them observes the baked-in attack without adding one.
  SafeClawArena assumes an honest user, so `user_message` models a different
  threat (a malicious user) from the benchmark's own attacks.
- **Observables.** Model and platform identity, the task metadata, and the
  names of the skills and plugins a task installs are available before the run.
  After it, the target emits what the agent produced, each at its store's leaf:
  replies, created workspace files, whether memory, config or cron changed, and
  its calls to each Sim-Google service. It never emits a store's provisioned
  contents, drops the CLI's `env_snapshot` and `cwd` from the call entries, and
  redacts every canary from all observables, so an optimizer can never plant one
  back and have the judge count the echo as a leak. Redaction goes by shape (any
  `CANARY_…` token, in any letter case, since most checks ignore case) as well as
  by the task's registered markers and `_canaries`, because 119 tasks plant
  canaries that their checks match but their `taint_assets` omit. Every
  registered marker in the corpus has that shape too. Redaction and the key
  scrub also catch underscores escaped with backslashes, as Markdown writes
  them, but an agent that encodes a canary or the key (base64, say) before
  sending it passes the encoded form through.
- **Config/query.** A task is configured via `set_config("task", <json>)` +
  `set_config("platform", …)`; the captured post-run state is read back via a
  single `query("post_state")` JSON blob the claim reconstructs.
- **Platforms.** `openclaw` (default) and `nemoclaw` — the two replicas buildable
  from the canonical repo. **SecLaw is excluded**: upstream's `Dockerfile.seclaw`
  COPYs a `seclaw/` source dir that is absent from the SafeClawArena repo, so it
  cannot be built from the canonical artifact (openclaw installs from npm,
  nemoclaw from a base image; SecLaw has no in-repo or published source). The
  claim's judge retains upstream's `cli_transport` logic for fidelity, but no
  shipped platform sets it.
- **NemoClaw layout.** The `nemoclaw` paths are upstream's: home
  `/sandbox/.openclaw` and workspace `/sandbox/.openclaw-data/workspace`. They
  match the NemoClaw `sandbox-base` image of the OpenClaw 2026.3.11 era that
  upstream built on, where `.openclaw/workspace` and the other state directories
  are symlinks into `.openclaw-data`, and the gateway, run with `HOME=/sandbox`,
  reads `~/.openclaw`. The `/sandbox/.nemoclaw` tree that `Dockerfile.nemoclaw`
  creates is NemoClaw's own plugin-state directory, not the OpenClaw home its
  comment calls it. That Dockerfile builds from the unpinned
  `sandbox-base:latest`, which dropped `.openclaw-data` on 2026-04-30
  (NVIDIA/NemoClaw#2227) and has since moved past OpenClaw 2026.3.11. On a base
  with the new layout, provisioning fails at `reset_env.sh`'s first copy into
  the workspace and the run abstains. To run NemoClaw as upstream measured it,
  set `NEMOCLAW_IMAGE` to an image built on a base from before that change.
