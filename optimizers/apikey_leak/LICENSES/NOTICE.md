# Per-file upstream attribution

## NVIDIA garak (Apache-2.0) — `garak-Apache-2.0.txt`

Upstream: https://github.com/NVIDIA/garak (pinned `2233d51bedf6bfb634872ef9efbcd94eae38cc1a`)

| Into (this module) | garak source |
| --- | --- |
| `_vendor/garak_apikey/{regexes,serviceutils}.py` | `garak/resources/apikey/{regexes,serviceutils}.py` |
| `data.json` (base prompts, partial keys) | `garak/probes/apikey.py` — `GetKey`, `CompleteKey` |
| prompt construction in `optimizer.py` | the `__init__` of those probes |

Verify with `python scripts/sync_upstream.py --check`.
