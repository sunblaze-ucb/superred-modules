# Per-file upstream attribution

**Based on Tencent Zhuque Lab AI-Infra-Guard
(https://github.com/Tencent/AI-Infra-Guard)** — required by that project's
NOTICE under Apache-2.0 Section 4(d).

## Tencent AI-Infra-Guard (Apache-2.0) — `AI-Infra-Guard-Apache-2.0.txt`, `AI-Infra-Guard-NOTICE.txt`

Upstream: https://github.com/Tencent/AI-Infra-Guard (pinned `dd6bd54655c9ff5fb7351f4299b56916f09ec6da`)

| Vendored into (this module) | AI-Infra-Guard source |
| --- | --- |
| `_vendor/aig_past_tense/template.py` | `.../single_turn/past_tense/template.py` |
| reformulation logic in `optimizer.py` | `.../single_turn/past_tense/past_tense.py` (`enhance`) |

## Technique

Andriushchenko & Flammarion (2024). The authors' repository
(https://github.com/tml-epfl/llm-past-tense) carries **no licence**, so no code
was taken from it; this module ports Tencent's Apache-2.0 implementation.

Verify with `python scripts/sync_upstream.py --check`.
