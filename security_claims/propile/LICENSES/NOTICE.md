# Per-file upstream attribution

## NVIDIA garak (Apache-2.0) — `garak-Apache-2.0.txt`

Upstream: https://github.com/NVIDIA/garak, licensed Apache-2.0. Method: Kim et al.,
"ProPILE: Probing Privacy Leakage in Large Language Models", NeurIPS 2023
(arXiv:2307.01881).

| Into (this module) | garak source | Change |
| --- | --- | --- |
| `src/propile_claim/data/prompt_templates.tsv` | `garak/data/propile/prompt_templates.tsv` | none (byte-for-byte, sha256-pinned) |
| `judge.py` — `pii_leak_score`, per-type matchers | `garak/detectors/propile.py` — `PIILeak` | ported to pure functions |
| `judge.py` — `pii_leak_exact_score` | `garak/detectors/propile.py` — `PIILeakExact` | ported to a pure function |
| `probes.py` — the four `build_*_items` | `garak/probes/propile.py` — `PIILeak{Twin,Triplet,Quadruplet,Unstructured}` | prompt-construction logic ported |
| `jaro_winkler.py` | `nltk.metrics.distance.{jaro_similarity,jaro_winkler_similarity}` (garak's fuzzy dependency) | reimplemented dependency-free; verified identical to nltk |

The module's own additions (scoring policy, `PIILeakTask`, factories, loaders,
tests, the synthetic sample) are original and MIT-licensed (see top-level
`LICENSE`).

**Not vendored:** garak's `garak/data/propile/pii_data.jsonl` (derived from NVIDIA
Nemotron-CC-v2.1, whose "Data Agreement for Model Training" forbids redistribution
and identifying individuals). This module ships a synthetic sample instead and
takes real PII only via a user-supplied `pii_data_path`. See the top-level
`NOTICE`.
