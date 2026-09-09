# Per-file upstream attribution

## DeepTeam (Apache-2.0) — `DeepTeam-Apache-2.0.txt`

Upstream: https://github.com/confident-ai/deepteam (pinned `dc148aad62f71330cfec7121d6afb4c620dfa683`)

| Vendored into (this module) | DeepTeam source |
| --- | --- |
| `_vendor/deepteam_blj/template.py` | `deepteam/attacks/multi_turn/bad_likert_judge/template.py` |
| `_vendor/deepteam_blj/base_template.py` | `deepteam/attacks/multi_turn/base_template.py` |
| turn sequence in `optimizer.py` | `bad_likert_judge/bad_likert_judge.py` — `_get_turns` |
| `SUPPORTED_CATEGORIES` | `BadLikertJudge.get_supported_categories()` |
| JSON field names in `parsing.py` | `bad_likert_judge/schema.py` |

Verify the vendored files with `python scripts/sync_upstream.py --check`.
