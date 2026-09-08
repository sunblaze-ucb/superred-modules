# Per-file upstream attribution

**Based on Tencent Zhuque Lab AI-Infra-Guard
(https://github.com/Tencent/AI-Infra-Guard)** — required by that project's
NOTICE under Apache-2.0 Section 4(d).

## Tencent AI-Infra-Guard (Apache-2.0) — `AI-Infra-Guard-Apache-2.0.txt`, `AI-Infra-Guard-NOTICE.txt`

Upstream: https://github.com/Tencent/AI-Infra-Guard (pinned `dd6bd54655c9ff5fb7351f4299b56916f09ec6da`)

| Vendored into (this module) | AI-Infra-Guard source |
| --- | --- |
| `_vendor/aig_actor_attack/template.py` | `AIG-PromptSecurity/deepteam/attacks/multi_turn/actor_attack/template.py` |
| turn loop in `optimizer.py` | same directory — `actor_attack.py` (`enhance`) |
| JSON field names in `parsing.py` | same directory — `schema.py` |

## Technique

Ren et al., ActorAttack. The authors' repository
(https://github.com/AI45Lab/ActorAttack) carries **no licence**, so no code was
taken from it; this module ports Tencent's independent Apache-2.0
implementation of the published method.

Verify with `python scripts/sync_upstream.py --check`.
