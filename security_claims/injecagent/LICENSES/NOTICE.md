# Per-file upstream attribution

## InjecAgent (MIT) — `InjecAgent-MIT.txt`

Upstream: https://github.com/uiuc-kang-lab/InjecAgent (pinned `f19c9f2`),
Copyright (c) 2023 Qiusi Zhan. Verbatim licence also at
`src/injecagent_claim/data/INJECAGENT_LICENSE`.

| Into (this module) | InjecAgent source |
| --- | --- |
| `data/test_cases_{dh,ds}_{base,enhanced}.json` (1,054 base + enhanced) | `data/test_cases_{dh,ds}_{base,enhanced}.json` |
| `judge.py` — `evaluate_output_prompted`, `output_parser`, detectors | `src/output_parsing.py` |
| `judge.py` — `evaluate_output_finetuned` | `src/output_parsing.py` |
| `judge.py` — `get_score` (ASR table) | `src/utils.py` |

Verify vendored data byte-for-byte with `python scripts/sync_upstream.py --check`.

The tool schemas (`tools.json`) InjecAgent adapted from ToolEmu (Apache-2.0) are
vendored in the paired **target** package, not here; see that package's NOTICE.
