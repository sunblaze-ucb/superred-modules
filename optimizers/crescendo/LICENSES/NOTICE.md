# Per-file upstream attribution

This directory records the provenance of third-party material redistributed by
`superred-optimizer-crescendo`. The full text of each upstream license lives
alongside this file (see `PyRIT-MIT.txt`). The top-level `NOTICE` carries the
same attribution in prose.

## Microsoft PyRIT (MIT, Copyright (c) Microsoft Corporation)

Upstream: https://github.com/microsoft/PyRIT — license: `PyRIT-MIT.txt`

| Vendored into (this module) | PyRIT source |
| --- | --- |
| `src/crescendo_optimizer/prompts/variant_1.py` | `pyrit/datasets/executors/crescendo/crescendo_variant_1.yaml` |
| `src/crescendo_optimizer/prompts/variant_2.py` | `pyrit/datasets/executors/crescendo/crescendo_variant_2.yaml` |
| `src/crescendo_optimizer/prompts/variant_3.py` | `pyrit/datasets/executors/crescendo/crescendo_variant_3.yaml` |
| `src/crescendo_optimizer/prompts/variant_4.py` | `pyrit/datasets/executors/crescendo/crescendo_variant_4.yaml` |
| `src/crescendo_optimizer/prompts/variant_5.py` | `pyrit/datasets/executors/crescendo/crescendo_variant_5.yaml` |
| `SCORING_SYSTEM_PROMPT` in `src/crescendo_optimizer/evaluator.py` | `pyrit/datasets/score/scales/red_teamer_system_prompt.yaml` + `pyrit/datasets/score/scales/task_achieved_scale.yaml` |

The variant prompts credit their upstream authors Mark Russinovich, Ahmed
Salem, and Ronen Eldan (Microsoft). Everything else in this module (the
optimizer harness, evaluator logic, capability-aware extensions, and
deterministic-replay mechanism) is original superred code, MIT licensed under
the top-level `LICENSE`.
