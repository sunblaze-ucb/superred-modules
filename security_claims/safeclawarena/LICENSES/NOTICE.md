# Per-file upstream attribution

## SafeClawArena (MIT) — `SafeClawArena-MIT.txt`

Upstream: https://github.com/sunblaze-ucb/SafeClawArena (pinned `a11f5cc`),
Copyright (c) 2026 SafeClawBench Authors. Verbatim licence also at
`src/safeclawarena_claim/data/SAFECLAWARENA_LICENSE`.

| Into (this module) | SafeClawArena source |
| --- | --- |
| `data/tasks/{ipi,pse,ssi,cdf}/*.json` (406 tasks) | `tasks/{ipi,pse,ssi,cdf}/*.json` |
| `data/task_schema.json` | `schema/task_schema.json` |
| `judge.py` (21 checks, detection, taint-leakage, 4 scores) | `scripts/judge.py` — `run_check` / `evaluate_detection` / `evaluate_taint_leakage` / `compute_score` |

Verify vendored data byte-for-byte with `python scripts/sync_upstream.py --check`.
