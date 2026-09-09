# Per-file upstream attribution

## DeepTeam (Apache-2.0) — `DeepTeam-Apache-2.0.txt`

Upstream: https://github.com/confident-ai/deepteam (pinned `dc148aad62f71330cfec7121d6afb4c620dfa683`)

| Into (this module) | DeepTeam source |
| --- | --- |
| `_vendor/dt_mathprompt/template.py` | `.../single_turn/math_problem/template.py` (byte-identical) |
| `compliance.py` `build_compliance_check_prompt` | `.../single_turn/compliance.py` (prompt reproduced verbatim) |
| enhancement loop in `optimizer.py` | `.../single_turn/math_problem/math_problem.py` (`enhance`) |
| JSON fields in `parsing.py` | `.../single_turn/math_problem/schema.py`, `compliance.py` |

Verify with `python scripts/sync_upstream.py --check`.
