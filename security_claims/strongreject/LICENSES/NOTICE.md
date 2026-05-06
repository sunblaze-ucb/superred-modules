# NOTICE

This module bundles material from the StrongREJECT project (Souly et al.,
NeurIPS 2024). Upstream license texts are preserved verbatim in
`dsbowen-MIT.txt` and `souly-MIT.txt`. This NOTICE describes per-file
attribution for the vendored data and prompts.

## Code (this module)

MIT, see top-level `superred-modules/LICENSE` (or the framework root).

## Judge prompt templates

Source: `github.com/dsbowen/strong_reject/strong_reject/eval_files/judge_templates.json`,
keys `strongreject_rubric` and `strongreject_rubric_system`.
Extracted verbatim into:

- `src/strongreject_claim/prompts/rubric_user.txt`
- `src/strongreject_claim/prompts/rubric_system.txt`

License: MIT, Copyright (c) 2024 Dillon Bowen. See `dsbowen-MIT.txt`.

## Forbidden-prompt dataset

Source: `github.com/alexandrasouly/strongreject/strongreject_dataset/`.
Both files vendored verbatim:

- `src/strongreject_claim/data/strongreject_dataset.csv` (313 rows)
- `src/strongreject_claim/data/strongreject_small_dataset.csv` (60 rows)

The dataset combines custom-authored prompts and prompts curated from
prior published datasets. License terms differ by source. Per the
paper supplementary section I and the upstream README:

| Source column value | Count (full) | License |
|---|---:|---|
| custom | 221 | MIT (Center for Human-Compatible AI 2024) |
| DAN | 35 | MIT (Shen et al. 2023) |
| AdvBench | 25 | MIT (Zou et al. 2023) |
| MaliciousInstruct | 12 | unspecified upstream |
| HarmfulQ | 11 | unspecified upstream |
| MasterKey | 3 | unspecified upstream |
| Jailbreaking via Prompt Engineering | 3 | unspecified upstream |
| OpenAI System Card | 3 | unspecified upstream |

Upstream README states the custom-authored questions are released under
MIT and that questions sourced from prior work retain their original
licenses.

Note on the `custom` count: paper section 2 (camera-ready, page 4)
reports 213 custom prompts; the upstream CSV that we vendor byte-equal
contains 221 rows tagged `source=custom`. The 8-row delta is an
upstream paper-vs-data discrepancy preserved here for byte-equality.

## Citation

When reporting numbers produced with this module, cite Souly et al.
2024 (BibTeX in the module README).
