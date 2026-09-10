# Per-file upstream attribution

## NVIDIA garak (Apache-2.0) — `garak-Apache-2.0.txt`

Upstream: https://github.com/NVIDIA/garak (pinned `2233d51bedf6bfb634872ef9efbcd94eae38cc1a`)

| Into (this module) | garak source |
| --- | --- |
| `data.json` (word lists, templates, repetition counts) | `garak/probes/divergence.py` — `Repeat`, `RepeatExtended` |
| prompt construction in `optimizer.py` | `divergence.Repeat.__init__` |

Verify with `python scripts/sync_upstream.py --check`.
