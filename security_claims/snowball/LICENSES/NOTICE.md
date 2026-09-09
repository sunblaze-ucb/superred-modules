# Per-file upstream attribution

Third-party material redistributed by `superred-claim-snowball`.

## NVIDIA garak (Apache-2.0, Copyright NVIDIA Corporation)

Upstream: https://github.com/NVIDIA/garak — license: `garak-Apache-2.0.txt`
Pinned commit: `2233d51bedf6bfb634872ef9efbcd94eae38cc1a`

| Vendored into (this module) | garak source |
| --- | --- |
| `src/snowball_claim/data/graph_connectivity.json` | `garak/data/graph_connectivity.json` |
| `src/snowball_claim/data/primality_testing.json` | `garak/data/primality_testing.json` |
| `src/snowball_claim/data/senator_search.json` | `garak/data/senator_search.json` |
| `_YES` / `_NO` regexes in `judge.py` | `garak/detectors/snowball.py` — `DontStartYes`, `DontStartNo` |

## Original source (MIT)

The question sets originate from Zhang et al., "How Language Model
Hallucinations Can Snowball" (2023) —
https://github.com/Nanami18/Snowballed_Hallucination (MIT).

Verify the vendored data and regexes with `python scripts/sync_upstream.py --check`.
