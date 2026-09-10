# Per-file upstream attribution

## NVIDIA garak (Apache-2.0) — `garak-Apache-2.0.txt`

Upstream: https://github.com/NVIDIA/garak (pinned `2233d51bedf6bfb634872ef9efbcd94eae38cc1a`)

| Into (this module) | garak source |
| --- | --- |
| `_vendor/garak_payloads/keyedprod_win1{0,1}.json` | `garak/data/payloads/keyedprod_win1{0,1}.json` |
| `data.json` (templates, parameter lists) | `garak/probes/grandma.py` — `Win10`, `Win11`, `Slurs`, `Substances`, `GrandmaIntent` |
| prompt construction in `optimizer.py` | the `__init__` / `_prompts_from_stub` of those probes |

Verify with `python scripts/sync_upstream.py --check`.
