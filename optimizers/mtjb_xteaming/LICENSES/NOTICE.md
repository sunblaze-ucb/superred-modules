# NOTICE

This module ports X-Teaming from MT-JailBench (arXiv:2605.11002), pinned at
commit `cb8184e0`. Upstream is redistributed under the MIT License; the verbatim
text is in `mt-jailbench-MIT.txt`.

## Code (this module)

MIT, see the module `LICENSE`. The strategy generation, per-turn prompting,
judge, and flow control are reimplemented against superred's async event model
in `src/mtjb_xteaming_optimizer/`.

## Vendored prompt payloads

Vendored verbatim (sha256-pinned in `_vendor/SHA256SUMS`), never reproduced in
authored code:

- `_vendor/mtjb/engine/attacks/xteaming/prompts/*.yaml`
- `_vendor/mtjb/engine/attacks/xteaming/updater_utils.py` (XTEAMING_SCORING_POLICY)
- `_vendor/mtjb/engine/attacks/xteaming/xteaming_prompt_generator.py` (TEXTGRAD_LOSS_TEMPLATE)

License: MIT ("The MT-JailBench Authors", 2026).

## Optional dependencies

`textgrad` (MIT) and `tiktoken` (MIT) power the optional refine path (extra
`refine`); they are referenced by name, not bundled.

## Citation

Cite MT-JailBench (arXiv:2605.11002) and the original X-Teaming work.
