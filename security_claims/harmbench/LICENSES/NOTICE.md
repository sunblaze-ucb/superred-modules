# NOTICE

superred-claim-harmbench

This product includes the superred HarmBench SecurityClaim module.
Original code is Copyright (c) 2026 the superred module authors, released under the MIT
License (see LICENSE).

It bundles data and code derived from the HarmBench project. The upstream
MIT license text is preserved in `LICENSES/harmbench-MIT.txt`.

## Upstream: HarmBench

Mazeika et al., "HarmBench: A Standardized Evaluation Framework for
Automated Red Teaming and Robust Refusal", ICML 2024
(https://arxiv.org/abs/2402.04249).

- Repository: https://github.com/centerforaisafety/HarmBench
- Vendored from commit 8e1604d1171fe8a48d8febecd22f600e462bdcdd
- License: MIT, Copyright (c) 2024 centerforaisafety

### Bundled data (verbatim)

- `src/harmbench_claim/data/harmbench_behaviors_text_test.csv` (320 rows)
- `src/harmbench_claim/data/harmbench_behaviors_text_val.csv` (80 rows)
  Source: upstream `data/behavior_datasets/harmbench_behaviors_text_{test,val}.csv`.
- `src/harmbench_claim/data/copyright_classifier_hashes/*.pkl` (100 files)
  Source: upstream `data/copyright_classifier_hashes/`. These `.pkl` files
  contain only `datasketch.MinHash` signatures (one-way hashes); no
  copyrighted source text is included in the hashes, by upstream design.

### Third-party excerpts in the contextual behaviors

The contextual-category behaviors carry a short `ContextString` alongside
each instruction. Some of these context strings embed brief third-party
excerpts that centerforaisafety does not itself own (for example, passages
of Wikipedia text under CC-BY-SA, an excerpt of a published paper, forum/HR
posts, and a code snippet). They are redistributed here byte-identically to
the upstream, widely-mirrored HarmBench ICML benchmark, so shipping them
mirrors the upstream published benchmark rather than asserting that
centerforaisafety's MIT license itself covers those excerpts. HarmBench
carries no non-commercial, research-only, gating, or no-redistribute clause;
the only obligation retained is the MIT copyright/permission notice.

### Vendored / ported code

- `src/harmbench_claim/judge.py` — the `LLAMA2_CLS_PROMPT_STANDARD` and
  `LLAMA2_CLS_PROMPT_CONTEXTUAL` classifier prompt templates are verbatim
  from upstream `eval_utils.py`.
- `src/harmbench_claim/advbench.py` — the `REFUSAL_PREFIXES` list is
  verbatim from upstream `eval_utils.py` (`advbench_metric_test_prefixes`).
  This list originates in Zou et al. 2023, AdvBench / llm-attacks
  (https://github.com/llm-attacks/llm-attacks, MIT); it is vendored here via
  HarmBench.
- `src/harmbench_claim/copyright.py` — faithful port of
  `compute_hashes_with_sliding_window`, `check_output_with_sliding_window`,
  and `compute_results_hashing` from upstream `eval_utils.py`.

## Citation

When reporting numbers produced with this module, cite Mazeika et al. 2024
(BibTeX in the module README).
