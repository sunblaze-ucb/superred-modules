# Bundled HarmBench data provenance

All files in this directory are vendored verbatim from the official
HarmBench repository:

- Repository: https://github.com/centerforaisafety/HarmBench
- Commit SHA: `8e1604d1171fe8a48d8febecd22f600e462bdcdd` (default branch
  `main`, commit dated 2024-08-05). This is the latest merged commit on
  `main` as of 2026-05-05.
- License: MIT (see upstream `LICENSE`).
- Citation: Mazeika et al., "HarmBench: A Standardized Evaluation Framework
  for Automated Red Teaming and Robust Refusal", ICML 2024
  (https://arxiv.org/abs/2402.04249).

## Files

| Local path | Upstream path | Verification |
|---|---|---|
| `harmbench_behaviors_text_test.csv` | `data/behavior_datasets/harmbench_behaviors_text_test.csv` | 320 rows (159 standard + 81 contextual + 80 copyright) |
| `harmbench_behaviors_text_val.csv` | `data/behavior_datasets/harmbench_behaviors_text_val.csv` | 80 rows (41 standard + 19 contextual + 20 copyright) |
| `copyright_classifier_hashes/lyrics_*.pkl` | `data/copyright_classifier_hashes/lyrics_*.pkl` | 50 files |
| `copyright_classifier_hashes/passage_*.pkl` | `data/copyright_classifier_hashes/passage_*.pkl` | 50 files |

## Schema (text CSVs)

Columns: `Behavior, FunctionalCategory, SemanticCategory, Tags, ContextString, BehaviorID`.

- `FunctionalCategory ∈ {standard, contextual, copyright}`
- `SemanticCategory ∈ {chemical_biological, illegal, misinformation_disinformation, harmful, harassment_bullying, cybercrime_intrusion, copyright}`
- `Tags`: comma-separated; relevant flags are `context` (use contextual judge
  prompt + ContextString), `hash_check` (use MinHash matcher instead of LLM judge),
  `book` / `lyrics` (sub-flags within copyright determining the sliding-window size).
- `BehaviorID`: stable string key used by upstream and us.

## Hash file format

Each `.pkl` file unpickles to a `list[datasketch.MinHash]` of reference
sliding-window MinHashes for the copyrighted source text. The pickled
objects depend on `datasketch` being importable at unpickle time
(see `pyproject.toml` for the version pin).

Files exist for behavior IDs in the copyright category. The book vs lyrics
distinction in the filename matches the `Tags` field.

## Why we vendor

HarmBench is not pip-installable (upstream issue #84 and PRs #58 / #85 are
unmerged). Copying the data into the wheel avoids a runtime download and
makes the module reproducible without network access. HarmBench is
MIT-licensed by centerforaisafety with no non-commercial, research-only,
gating, or no-redistribute clause. The copyright-classifier `.pkl` files
contain only one-way MinHash signatures, no copyrighted source text. Note,
however, that the contextual behaviors' `ContextString` fields embed short
third-party excerpts (e.g. Wikipedia text under CC-BY-SA, a published paper,
forum posts, a code snippet) that centerforaisafety does not itself own;
these are redistributed here byte-identically to the upstream, widely-mirrored
HarmBench benchmark, mirroring the upstream publication rather than asserting
fresh MIT coverage of those excerpts.
