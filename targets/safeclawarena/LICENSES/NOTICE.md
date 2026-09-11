# Per-file upstream attribution

## SafeClawArena (MIT) — `SafeClawArena-MIT.txt`

Upstream: https://github.com/sunblaze-ucb/SafeClawArena (pinned `a11f5cc`),
Copyright (c) 2026 SafeClawBench Authors. Verbatim licence also at
`src/safeclawarena_target/_vendor/safeclawarena/LICENSE`.

| Into (this module) | SafeClawArena source |
| --- | --- |
| `_vendor/safeclawarena/Dockerfile{,.nemoclaw,.seclaw}` | `Dockerfile{,.nemoclaw,.seclaw}` |
| `_vendor/safeclawarena/tools/sim-google/sim-google` | `tools/sim-google/sim-google` |
| `_vendor/safeclawarena/tools/seclaw-agent-wrapper.js` | `tools/seclaw-agent-wrapper.js` |
| `_vendor/safeclawarena/scripts/reset_env.sh` | `scripts/reset_env.sh` |
| `_vendor/safeclawarena/configs/**` | `configs/**` |
| `runtime.py` (platforms, lifecycle, sessions, capture) | `scripts/judge.py` — execution/capture half |

Not vendored: `tools/ripgrep/rg` (unreferenced binary). Verify vendored harness
byte-for-byte with `python scripts/sync_upstream.py --check`.
