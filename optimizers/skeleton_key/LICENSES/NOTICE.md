# Per-file upstream attribution

This directory records the provenance of third-party material redistributed by
`superred-optimizer-skeleton-key`. The full text of the upstream license lives
alongside this file (see `PyRIT-MIT.txt`). The top-level `NOTICE` carries the
same attribution in prose.

## Microsoft PyRIT (MIT, Copyright (c) Microsoft Corporation)

Upstream: https://github.com/microsoft/PyRIT — license: `PyRIT-MIT.txt`
Pinned commit: `2016c4a8566bd66253d431ff38400bade4c77fa3`

Vendored byte-for-byte (sha256-pinned in `_vendor/SHA256SUMS`):

| Vendored into (this module) | PyRIT source |
| --- | --- |
| `src/skeleton_key_optimizer/_vendor/pyrit/datasets/executors/skeleton_key/skeleton_key.prompt` | `pyrit/datasets/executors/skeleton_key/skeleton_key.prompt` |
| `src/skeleton_key_optimizer/_vendor/pyrit/datasets/executors/skeleton_key/skeleton_key_acceptance.prompt` | `pyrit/datasets/executors/skeleton_key/skeleton_key_acceptance.prompt` |

Everything else in this module (the optimizer harness, the SeedDataset seed
loader, surface selection, and the prepended-conversation transcript framing) is
original superred code, MIT licensed under the top-level `LICENSE`, Copyright (c)
2026 the superred module authors.
