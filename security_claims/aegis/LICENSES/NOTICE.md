# Third-party attribution — superred-claim-aegis

This package is an original superred `SecurityClaim` port. Its own code is MIT
(see `../LICENSE`). It incorporates the following third-party material.

## Aegis 1.0 dataset (vendored, CC-BY-4.0)

- **Source:** [nvidia/Aegis-AI-Content-Safety-Dataset-1.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0)
  (a.k.a. *Nemotron Content Safety Dataset V1*).
- **License:** CC-BY-4.0 — full text in `aegis-dataset-CC-BY-4.0.txt`.
- **Attribution:** curated by Shaona Ghosh, NVIDIA.
- **Pinned revision:** `bd96d862068e47630197de64eb91f8d1481ff3e0`, `test` split.
- **Vendored files:**
  - `src/aegis_claim/_vendor/Content Moderation Extracted Annotations 02.08.24_test_release_0418_v1.parquet`
    — byte-identical to upstream (the provenance anchor).
  - `src/aegis_claim/_vendor/aegis_test.csv` — a deterministic, lossless CSV
    rendering of that parquet, loaded by the stdlib `csv` module.
- Both are sha256-pinned in `src/aegis_claim/_vendor/SHA256SUMS` and verified
  against the pinned revision by `scripts/sync_upstream.py --check`.

Redistribution here is permitted under CC-BY-4.0 provided attribution is given,
which this file and `../NOTICE` supply. No modifications were made to the parquet.

## Paper / taxonomy

- Shaona Ghosh, Prasoon Varshney, Erick Galinkin, Christopher Parisien,
  "AEGIS: Online Adaptive AI Content Safety Moderation with Ensemble of LLM
  Experts", arXiv:2404.05993 (NVIDIA).

The 13 harm-category names in `categories.py` follow the dataset's own labels.

## The judge is ours, not NVIDIA's

The safety judge shipped here (`AegisSafetyJudge`) is a generic LLM-as-judge — our
scoring choice — flagged `generic-judge`. It is **not** NVIDIA's Aegis-Guard
(LlamaGuard-based) classifier and reproduces none of NVIDIA's model code. See
`../ASSUMPTIONS.md`.

## Runtime / dev dependencies

- `litellm` (MIT) — used at runtime for the judge's LLM calls.
- `pyarrow` (Apache-2.0) and `huggingface_hub` (Apache-2.0) — used only by
  `scripts/sync_upstream.py` to refresh/verify the vendored data; not imported by
  the installed package.
