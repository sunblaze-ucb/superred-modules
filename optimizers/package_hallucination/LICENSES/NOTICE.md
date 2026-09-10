# Per-file upstream attribution

## NVIDIA garak (Apache-2.0) — `garak-Apache-2.0.txt`

Upstream: https://github.com/NVIDIA/garak (pinned `2233d51bedf6bfb634872ef9efbcd94eae38cc1a`)

| Into (this module) | garak source |
| --- | --- |
| `data.json` (`stub_prompts`, code tasks, language names) | `garak/probes/packagehallucination.py` — `Python` … `Dart` |
| prompt construction in `optimizer.py` | `PackageHallucinationProbe.__init__` |

Verify with `python scripts/sync_upstream.py --check`.
