# NOTICE

This module ports the CoA (Chain-of-Attack) multi-turn jailbreak from
MT-JailBench (arXiv:2605.11002), pinned at commit `cb8184e0`. Upstream is
redistributed under the MIT License; the verbatim text is in
`mt-jailbench-MIT.txt`.

## Code (this module)

MIT, see the module `LICENSE`. The CoA control flow, chain generation, prompt
update, binary judge, and LLM-approximation similarity scorer are reimplemented
against superred's async event model in `src/mtjb_coa_optimizer/`.

## Vendored prompt payloads

Vendored verbatim (sha256-pinned in `_vendor/SHA256SUMS`), never reproduced in
authored code:

- `_vendor/mtjb/engine/attacks/coa/prompts.yaml`
- `_vendor/mtjb/engine/attacks/coa/similarity_scorer.py` (for `SIM_APPROX_PROMPT`)

License: MIT ("The MT-JailBench Authors", 2026). MT-JailBench's CoA is adapted
from YancyKahn/CoA; the technique is credited to its original authors.

## Citation

Cite MT-JailBench (arXiv:2605.11002) and the original Chain-of-Attack work when
reporting numbers produced with this module.
