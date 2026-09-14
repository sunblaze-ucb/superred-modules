# superred-target-safeclawarena

A superred **Target** over [SafeClawArena](https://github.com/sunblaze-ucb/SafeClawArena)'s
container harness (Niu et al., arXiv:2606.30755). It runs one SafeClawArena
adversarial task against a containerized self-hosted-agent platform replica —
**OpenClaw** or **NemoClaw** — with a simulated Google Workspace,
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

- **config**: `task` (the full SafeClawArena task JSON) and `platform`.
- **query**: `post_state`, a JSON blob of agent responses, tool calls, the
  Sim-Google call log, memory/config, new files and their contents, and probed
  file/route state, which the claim's judge reads.
- **controllables**: `user_message` (override the first session turn; see the
  threat-model note below) and `workspace_file` (plant one extra
  attacker-controlled workspace file). The benchmark's own payload stays
  provisioned either way.
- **observables**: before the run, `model_identity`, `platform`,
  `task_metadata`, and the names of the skills and plugins the task installs.
  After the run, what the agent did, one observable per store: its replies, the
  workspace files it created, whether it changed memory, config or cron, and its
  calls to each Sim-Google service.

## Security domains

The forest follows the superred security-domain guide, with one leaf per store
the agent reads from or acts on (principle 6), the user's channel as its own root
(principle 5), and knowledge apart from control (principle 4):

```
system
 ├── supply_chain        skills, plugins              (SSI)
 ├── persistent_state    memory, config, cron         (PSE)
 ├── response            the agent's replies
 ├── model_identity      model and platform
 └── task_metadata       the task's id, dimension and title
user                     the user's instruction
tools
 ├── workspace_files     the agent's workspace
 └── sim_google          one leaf per service the tasks use (11):
                         gmail, drive, docs, sheets, slides, calendar, chat,
                         keep, tasks, contacts, groups
```

The simulator's other five services (forms, people, classroom, admin,
appscript) appear in no task; the agent's calls to them are reported at the
`sim_google` node. That keeps `DOMAIN.distinct_combinations()` at 2,959,478
scopes, against about 95 million with all sixteen as leaves.

The two injection surfaces are the user's turn and the workspace file, so
`scope={WORKSPACE_FILES_TAG}` plants a file the agent may read. A Sim-Google
store leaf is an observation scope: `scope={SIM_GOOGLE_SERVICE_TAGS["gmail"]}`
runs the task's own baked-in inbox payload and lets the optimizer watch the
agent's gmail calls, but it has nothing to inject into. Reading the agent's
replies is `read_only={RESPONSE_TAG}`. Tags match by identity, so build scopes
from the tags this package exports.

After a run the target emits only what the agent produced, never a store's
provisioned contents, and it redacts every canary even from that: any
`CANARY_…` token in any letter case, plus the task's registered taint markers
and `_canaries`. The tasks plant canary credentials in workspace files, auth
profiles and memory, and 119 of them plant canaries that their checks look for
but their taint assets omit. An optimizer that saw one could plant it back
through the workspace file, and the agent merely reading that file would put it
in the transcript, which the judge counts as a leak the agent never made.
Memory, config and cron therefore report only whether the run changed them, and
the Sim-Google call entries drop the CLI's `env_snapshot` and `cwd`.

**Threat model.** SafeClawArena assumes an honest user, and none of the
benchmark's own attacks come through the user's turn. `user_message` models a
different threat, a malicious user, so grant `USER_TAG` only when that is what
you mean to test. Overriding the first session's instruction can also remove the
task's trigger (the instruction that makes the agent read the poisoned email,
say), so the baked-in payload stays provisioned but may never fire.

## Provenance & faithfulness

SafeClawArena's container harness (Dockerfiles, Sim-Google CLI, `reset_env.sh`,
configs) is vendored **byte-for-byte** (verify with
`python scripts/sync_upstream.py --check`); the 6.3 MB `ripgrep` binary is
intentionally not vendored. The execution/capture logic is ported from upstream
`scripts/judge.py`, with the differences listed in
[`ASSUMPTIONS.md`](ASSUMPTIONS.md). MIT-licensed; SafeClawArena's MIT licence is
shipped in `LICENSES/` and the vendor root.

Cite Niu et al., 2026 (arXiv:2606.30755).
