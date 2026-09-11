# superred-target-safeclawarena

A superred **Target** over [SafeClawArena](https://github.com/sunblaze-ucb/SafeClawArena)'s
container harness (Niu et al., arXiv:2606.30755). It runs one SafeClawArena
adversarial task against a containerized self-hosted-agent platform replica —
**OpenClaw**, **NemoClaw**, or **SecLaw** — with a simulated Google Workspace,
provisions the task's environment, executes its sessions, and captures the
post-run state that the paired [`superred-claim-safeclawarena`](../../security_claims/safeclawarena)
package judges (deterministic canary/taint).

## Requirements

End-to-end execution needs **Docker** (rootless or rootful) and the platform
image, built from the vendored Dockerfile on first use (Ubuntu 24.04 + the
pinned OpenClaw npm + the Sim-Google CLI). Like the `dtap_openclaw` target, the
Docker path is not exercised in unit tests; the target's contract is.

## Usage

```python
from safeclawarena_target import safeclawarena_target_factory

target_factory = safeclawarena_target_factory(platform="openclaw")
# pair with `safeclawarena_claim(...)` and an optimizer via the superred Controller.
```

The target exposes:

- **config** — `task` (the full SafeClawArena task JSON) and `platform`.
- **query** — `post_state` (a JSON blob of agent responses, tool calls, the
  Sim-Google call log, memory/config, new files + contents, and probed
  file/route state).
- **controllables** — `user_message` (override the first session turn) and
  `external_data` (append an attacker-controlled workspace file); the baked
  benchmark attack runs regardless.
- **security domain** — `system → {skills, memory, config}`, `user`,
  `external_data → {sim_google, file_content}`, `tools`.

## Provenance & faithfulness

SafeClawArena's container harness (Dockerfiles, Sim-Google CLI, SecLaw wrapper,
`reset_env.sh`, configs) is vendored **byte-for-byte** (verify with
`python scripts/sync_upstream.py --check`); the 6.3 MB `ripgrep` binary is
intentionally not vendored. The execution/capture logic is ported from upstream
`scripts/judge.py`. Deviations are in [`ASSUMPTIONS.md`](ASSUMPTIONS.md).
MIT-licensed; SafeClawArena's MIT licence is shipped in `LICENSES/` and the
vendor root.

Cite Niu et al., 2026 (arXiv:2606.30755).
