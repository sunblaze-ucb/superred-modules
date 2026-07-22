# superred-optimizer-crescendo

Crescendo multi-turn jailbreak optimizer for the
[superred](https://superred.simonsure.com) red-teaming framework.

Crescendo escalates a conversation from benign to harmful over several turns,
backtracking and restoring history when the target refuses, until the
conversation objective is achieved.

## Install

```bash
pip install superred-optimizer-crescendo
```

## Credits / upstream

This optimizer reimplements the **Crescendo** multi-turn jailbreak
(Russinovich, Salem, Eldan; USENIX Security 2025 —
https://crescendo-the-multiturn-jailbreak.github.io/).

The bundled attacker prompt variants
(`src/crescendo_optimizer/prompts/variant_1..5.py`) and the task-achieved
scoring prompt (`evaluator.py`) are taken from **Microsoft PyRIT**
(https://github.com/microsoft/PyRIT), MIT licensed, Copyright (c) Microsoft
Corporation. See `LICENSES/NOTICE.md` for per-file attribution.

Our optimizer harness, evaluator logic, capability-aware extensions, and
deterministic-replay mechanism are original superred code, MIT licensed
(see `LICENSE`).
