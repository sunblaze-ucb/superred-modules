# Per-file upstream attribution

## HackAPrompt (MIT) — `HackAPrompt-MIT.txt`

Upstream: https://github.com/PromptLabs/hackaprompt (pinned `f0a035a`),
Copyright (c) 2023 hackaprompt team. Paper: Schulhoff et al., EMNLP 2023
(arXiv:2311.16119). Dataset: `hackaprompt/hackaprompt-dataset` (MIT).

| Into (this module) | HackAPrompt source |
| --- | --- |
| `judge.py` — `validate_completion`, `format_completion` | `hackaprompt/evaluator.py` (`LevelEvaluator`) |
| `judge.py` — `remove_whitespace` | `hackaprompt/utils.py` |
| `judge.py` — `compute_score` | `hackaprompt/evaluator.py` |

No dataset is vendored (the levels are code; the attack is optimizer-driven). The
11 level defenses (prompt templates + input filters) are ported in the paired
**target** package; see that package's NOTICE.
