# Per-file upstream attribution

## NVIDIA garak (Apache-2.0) — `garak-Apache-2.0.txt`

Upstream: https://github.com/NVIDIA/garak (pinned `2233d51bedf6bfb634872ef9efbcd94eae38cc1a`)

| Into (this module) | garak source |
| --- | --- |
| `data.json` (EICAR/GTUBE/GTPHISH signatures, `REPEAT_PROMPTS`) | `garak/probes/av_spam_scanning.py` — `EICAR`, `GTUBE`, `GTphish` |
| prompt construction in `optimizer.py` | the `__init__` of those probes |

## Origin of the signatures (one layer up from garak)

The three signatures are public scanner-test standards, not garak's authorship;
garak carries them verbatim and this module reproduces them unchanged:

| Signature | Canonical origin |
| --- | --- |
| EICAR test file | EICAR — European Institute for Computer Antivirus Research (free public test string) |
| GTUBE | Apache SpamAssassin — https://spamassassin.apache.org/gtube/ |
| GTphish | Cisco — outbreak-filter URL-rewriting test string (per garak's `GTphish` `doc_uri`, a Cisco Email Security Appliance page; not SpamAssassin) |

They are fixed public test constants cited as their standards, not carried under
a software licence. Only the repetition prompt templates around them are garak's.

Verify with `python scripts/sync_upstream.py --check`.
