# Per-file upstream attribution

## HackAPrompt (MIT) — `HackAPrompt-MIT.txt`

Upstream: https://github.com/PromptLabs/hackaprompt (pinned `f0a035a`),
Copyright (c) 2023 hackaprompt team. Paper: Schulhoff et al., EMNLP 2023
(arXiv:2311.16119).

| Into (this module) | HackAPrompt source |
| --- | --- |
| `_vendor/levels.py` — 11 `format_prompt` templates | `hackaprompt/evaluator.py` (`Level0Evaluator` … `Level10Evaluator`) |
| `_vendor/levels.py` — `contains_blacklisted_word`, `contains_pwned_letters`, `remove_non_emoji` | `hackaprompt/utils.py` |
| `target.py` — per-level orchestration (filter/escape, two-step, secret key) | `hackaprompt/evaluator.py` (`evaluate` bodies) |

The level templates are ported byte-for-byte and verified level-by-level against
upstream. The completer (`completers.py`), the MongoDB game DB, and the Gradio UI
are not ported — the target drives the model via litellm.
