# Per-file upstream attribution

This directory records the provenance of third-party material redistributed by
`superred-optimizer-token-smuggling`. The full text of the upstream license
lives alongside this file (see `garak-Apache-2.0.txt`). The top-level
`NOTICE` carries the same attribution in prose.

## NVIDIA garak (Apache-2.0, Copyright NVIDIA Corporation)

Upstream: https://github.com/NVIDIA/garak — license: `garak-Apache-2.0.txt`

Pinned upstream commit: `2233d51bedf6bfb634872ef9efbcd94eae38cc1a`

| Vendored into (this module) | garak source |
| --- | --- |
| `DEFAULT_HOMOGLYPH_MAP` in `src/token_smuggling_optimizer/smuggling.py` | `garak/probes/smuggling.py` — `DEFAULT_HOMOGLYPH_MAP` |
| `homoglyph_replace()` in `src/token_smuggling_optimizer/smuggling.py` | `garak/probes/smuggling.py` — `_homoglyph_replace()` |
| `HYPOTHETICAL_TEMPLATE` in `src/token_smuggling_optimizer/smuggling.py` | `garak/probes/smuggling.py` — `HypotheticalResponse.prompts` |

Not vendored: `FunctionMasking` (see `ASSUMPTIONS.md`).
