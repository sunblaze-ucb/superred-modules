# NOTICE

This module ports the Mix composite attack from MT-JailBench (arXiv:2605.11002),
pinned at commit `cb8184e0`. Upstream is redistributed under the MIT License; the
verbatim text is in `mt-jailbench-MIT.txt`.

## Code (this module)

MIT, see the module `LICENSE`. The optimizer adapter, the stdlib
`client.unified_llm_client` shim, and the reconstructed engine loop are original
to this module (`src/mtjb_mix_optimizer/`).

## Vendored engine (driven verbatim)

The MT-JailBench `engine/` subtree required to run Mix is vendored byte-for-byte
(sha256-pinned in `_vendor/SHA256SUMS`): `core`, `interfaces`, `attack_type`,
`utils`, and the `coa` / `fitd` / `xteaming` / `mix` attack packages including
every prompt payload. License: MIT ("The MT-JailBench Authors", 2026).

Mix composes Crescendo, ActorAttack, Chain-of-Attack (adapted from
YancyKahn/CoA), Foot-in-the-Door and X-Teaming; each technique is credited to its
original authors.

## Dependencies

Required: `jinja2` (BSD-3-Clause), `tenacity` (Apache-2.0). Optional (extra
`refine`): `textgrad` (MIT), `tiktoken` (MIT) -- referenced by name, not bundled.

## Citation

Cite MT-JailBench (arXiv:2605.11002) and the original works for each technique.
