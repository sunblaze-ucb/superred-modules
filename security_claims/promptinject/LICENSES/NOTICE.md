# Per-file upstream attribution

## NVIDIA garak (Apache-2.0) — `garak-Apache-2.0.txt`

Upstream: https://github.com/NVIDIA/garak (pinned `2233d51bedf6bfb634872ef9efbcd94eae38cc1a`)

| Vendored into (this module) | garak source |
| --- | --- |
| `loader.py` build config | `garak/probes/promptinject.py` — `load_data_all_attacks` |
| `judge.py` `rogue_emitted()` | `garak/detectors/promptinject.py` — `AttackRogueString` |
| `src/promptinject_claim/_vendor/promptinject/*` | `garak/resources/promptinject/*` |

## PromptInject (MIT, Copyright (c) 2020 Agency Enterprise, LLC) — `PromptInject-MIT.txt`

Upstream: https://github.com/agencyenterprise/PromptInject — the `_vendor`
files are the same subset garak redistributes.

Verify the vendored files with `python scripts/sync_upstream.py --check`.
