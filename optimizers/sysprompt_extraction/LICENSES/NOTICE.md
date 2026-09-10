# Per-file upstream attribution

## NVIDIA garak (Apache-2.0) — `garak-Apache-2.0.txt`

Upstream: https://github.com/NVIDIA/garak (pinned `2233d51bedf6bfb634872ef9efbcd94eae38cc1a`)

| Into (this module) | garak source |
| --- | --- |
| `_vendor/garak_sysprompt/attacks.json` | `garak/data/sysprompt_extraction/attacks.json` |
| `load_attack_templates()` in `optimizer.py` | `sysprompt_extraction.SystemPromptExtraction._load_garak_attacks` |

Verify with `python scripts/sync_upstream.py --check`.
