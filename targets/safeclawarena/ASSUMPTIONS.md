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

- **Target (`runtime.py`)** — platform config, image build, container lifecycle,
  provisioning via the vendored `reset_env.sh`, session execution, and post-run
  state capture. Faithful to upstream's `PLATFORMS`, `reset_environment`,
  `run_session`/`send_message`, and `capture_post_state`.
- **Claim (`safeclawarena_claim.judge`)** — the deterministic check evaluators
  and dimension scoring, as a pure function over the captured state.

## End-to-end execution needs Docker

`run()` builds a platform image from the vendored Dockerfile, starts a container,
provisions the task, runs the sessions, and captures state — all via `docker`.
Like the `dtap_openclaw` target, that path is **not exercised in unit tests**
(no Docker/agent runtime in CI). The unit tests cover the pure contract: the
security-domain forest, config/query specs, controllables/observables, config
round-trip, and the pure runtime helpers (`platform_config`,
`file_check_targets`, `http_route_targets`).

Transport note: sessions are sent via `docker exec … curl` to the in-container
gateway. Upstream uses host HTTP with an exec
fallback; driving over exec is the robust equivalent that avoids host↔container
network assumptions.

## superred adaptation

- **Trust-domain forest.** SafeClawArena's four dimensions / five invariants are
  mapped to a nine-node `SecurityDomain`: `system → {skills, memory, config}`,
  `user`, `external_data → {sim_google, file_content}`, `tools`. `model_identity`
  is surfaced as an observable tagged `system` (no separate model node).
- **Controllables.** The task's adversarial payload is baked into its
  environment (the benchmark's design). The target additionally exposes
  `user_message` (override the first session instruction) and `external_data`
  (append an attacker-controlled workspace file) so an optimizer may vary the
  attack; left un-injected, the benchmark runs as-shipped.
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
