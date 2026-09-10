# Per-file upstream attribution

## NVIDIA garak (Apache-2.0) — `garak-Apache-2.0.txt`

Upstream: https://github.com/NVIDIA/garak (pinned `2233d51bedf6bfb634872ef9efbcd94eae38cc1a`)

| Into (this module) | garak source |
| --- | --- |
| `_vendor/garak_data/truefalse_falseclaims_50.txt` | `garak/data/truefalse_falseclaims_50.txt` |
| `data.json` (framing prefixes) | `garak/probes/misleading.py` — `FalseAssertion` |
| prompt construction in `optimizer.py` | `misleading.FalseAssertion.__init__` |

Verify with `python scripts/sync_upstream.py --check`.
