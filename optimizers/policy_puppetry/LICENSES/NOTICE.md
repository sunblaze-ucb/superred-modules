# Per-file upstream attribution

This directory records the provenance of third-party material redistributed by
`superred-optimizer-policy-puppetry`. The full text of the upstream license
lives alongside this file (see `garak-Apache-2.0.txt`). The top-level
`NOTICE` carries the same attribution in prose.

## NVIDIA garak (Apache-2.0, Copyright NVIDIA Corporation)

Upstream: https://github.com/NVIDIA/garak — license: `garak-Apache-2.0.txt`

Pinned upstream commit: `2233d51bedf6bfb634872ef9efbcd94eae38cc1a`

Refresh the vendored template with `python scripts/sync_upstream.py`.

| Vendored into (this module) | garak source |
| --- | --- |
| `src/policy_puppetry_optimizer/data/upstream/bypass_template_0.txt` | `garak/probes/doctor.py` — `Bypass.templates[0]` |
| `src/policy_puppetry_optimizer/data/upstream/bypass_template_1.txt` | `garak/probes/doctor.py` — `Bypass.templates[1]` |
| `LEET_SLOT_SUFFIX` in `src/policy_puppetry_optimizer/templates.py` | `garak/probes/doctor.py` — `BypassLeet.__init__` |
| `render()` in `src/policy_puppetry_optimizer/templates.py` | `garak/probes/doctor.py` — `Bypass._build_prompts()` |
| `leetspeak()` in `src/policy_puppetry_optimizer/templates.py` | `garak/resources/encodings.py` — `leetspeak()` |

Not vendored: `Puppetry` (hard-coded request, no slot). See `ASSUMPTIONS.md`.
