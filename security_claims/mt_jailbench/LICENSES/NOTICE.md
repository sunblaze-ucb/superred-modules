# NOTICE

This module ports MT-JailBench's evaluation (the unanimous 3-judge vote) as a
superred SecurityClaim, pinned at commit `cb8184e0`. MT-JailBench and HarmBench
are redistributed under the MIT License; verbatim texts are in
`mt-jailbench-MIT.txt` and `harmbench-MIT.txt`.

## Code (this module)

MIT, see the module `LICENSE`. The judge parsers and the unanimous majority-vote
aggregation are reimplemented in async form in `src/mt_jailbench_claim/`.

## Vendored judge rubrics

Vendored verbatim (sha256-pinned in `_vendor/SHA256SUMS`); the rubric bodies are
read via `ast` and never reproduced in authored code:

- `_vendor/mtjb/engine/utils/independent_judge.py`

License: MIT ("The MT-JailBench Authors", 2026).

## Task set (HarmBench standard behaviors)

Vendored verbatim (sha256-pinned):

- `data/harmbench_behaviors.csv` (320 rows; 159 filtered to
  `FunctionalCategory == "standard"`)

Source: HarmBench (`github.com/centerforaisafety/HarmBench`), redistributed in
MT-JailBench. License: MIT. **HarmBench is a distinct upstream** and is
attributed separately from MT-JailBench.

## Citation

Cite MT-JailBench (arXiv:2605.11002) and HarmBench (Mazeika et al. 2024) when
reporting numbers produced with this module.
