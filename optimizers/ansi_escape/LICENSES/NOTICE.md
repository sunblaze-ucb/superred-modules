# Per-file upstream attribution

## NVIDIA garak (Apache-2.0) — `garak-Apache-2.0.txt`

Upstream: https://github.com/NVIDIA/garak (pinned `2233d51bedf6bfb634872ef9efbcd94eae38cc1a`)

| Into (this module) | garak source |
| --- | --- |
| `_vendor/garak_ansi/ansi.py` (payload constants) | `garak/resources/ansi.py` |
| `data.json` (prompt-stub lists) | `garak/probes/ansiescape.py` — `AnsiEscaped`, `AnsiRaw` |
| prompt construction in `optimizer.py` | `ansiescape.AnsiEscaped.__init__` / `AnsiRaw.__init__` |

Verify with `python scripts/sync_upstream.py --check`.
