# Per-file upstream attribution

## CipherChat (MIT) — `CipherChat-MIT.txt`

Upstream: https://github.com/RobustNLP/CipherChat (pinned `6fce7891a9a072b933f13bba7f58651577499fb5`)

| Vendored into (this module) | CipherChat source |
| --- | --- |
| `_vendor/cipherchat/encode_experts.py` | `encode_experts.py` |
| `_vendor/cipherchat/prompts_and_demonstrations.py` | `prompts_and_demonstrations.py` |
| `assembly.py` prompt construction | `main.py` |

Verify with `python scripts/sync_upstream.py --check`.
