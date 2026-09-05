# Per-file upstream attribution

Third-party material redistributed by `superred-optimizer-agentbreaker`. Full
license text is in `garak-Apache-2.0.txt`; the top-level `NOTICE` carries the
same attribution in prose.

## NVIDIA garak (Apache-2.0, Copyright NVIDIA Corporation)

Upstream: https://github.com/NVIDIA/garak — license: `garak-Apache-2.0.txt`

Pinned upstream commit: `2233d51bedf6bfb634872ef9efbcd94eae38cc1a`

Verify the vendored prompts with `python scripts/sync_upstream.py --check`.

| Vendored into (this module) | garak source |
| --- | --- |
| `src/agentbreaker_optimizer/data/upstream/prompts.yaml` | `garak/data/agent_breaker/prompts.yaml` |
| `format_attack_prompt()`, `analysis_prompt()`, `exploit_prompt()` in `prompts.py` | `garak/probes/agent_breaker.py` |
| `parse_analysis()`, `build_tool_plans()` in `analysis.py` | `garak/probes/agent_breaker.py` — `_analyze_attackable_tools`, `_build_tool_configs` |

Only the `ANALYSIS` and `EXPLOIT` keys of `prompts.yaml` are used; the
discovery and verify keys are carried unused (SuperRed provides those phases).
See `ASSUMPTIONS.md`.
