# Per-file upstream attribution

## NVIDIA garak (Apache-2.0) — `garak-Apache-2.0.txt`

Upstream: https://github.com/NVIDIA/garak (pinned `2233d51bedf6bfb634872ef9efbcd94eae38cc1a`)

| Into (this module) | garak source |
| --- | --- |
| `_vendor/garak_data/truefalse_falseclaims_50.txt` | `garak/data/truefalse_falseclaims_50.txt` |
| `data.json` (framing prefixes) | `garak/probes/misleading.py` — `FalseAssertion` |
| prompt construction in `optimizer.py` | `misleading.FalseAssertion.__init__` |

## Data source (one layer up from garak): the "true-false" dataset

The claims in `truefalse_falseclaims_50.txt` are a 50-item subset of the
"true-false" dataset introduced by Azaria & Mitchell, "The Internal State of an
LLM Knows When It's Lying," Findings of EMNLP 2023 (arXiv:2304.13734) —
https://arxiv.org/abs/2304.13734. garak curated and vendors the subset; this
module reproduces it unchanged and credits the dataset as the canonical source
of the statements. The items are short factual assertions (not code); they are
cited here as a data source rather than carried under a software licence.

Verify with `python scripts/sync_upstream.py --check`.
