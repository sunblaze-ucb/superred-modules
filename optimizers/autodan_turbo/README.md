# superred-optimizer-autodan-turbo

An **AutoDAN-Turbo** lifelong-strategy jailbreak optimizer for the
[superred](https://superred.simonsure.com) red-teaming framework.

AutoDAN-Turbo discovers jailbreak strategies from scratch, with no
human-designed seed strategies, and remembers what worked in a strategy
library that persists across attempts. One superred run is one paper "epoch"
against the goal: generate an attack (cold, or informed by retrieved
strategies), probe the target, score the response, and — if the score
improved — mine a new `{Strategy, Definition}` pair into the library for
future runs to retrieve.

Reference: Liu et al., "AutoDAN-Turbo: A Lifelong Agent for Strategy
Self-Exploration to Jailbreak LLMs,"
[arXiv:2410.05295](https://arxiv.org/abs/2410.05295) (ICLR 2025 spotlight).
Official implementation:
[`SaFo-Lab/AutoDAN-Turbo`](https://github.com/SaFo-Lab/AutoDAN-Turbo)
(MIT License; GitHub org renamed from `SaFoLab-WISC` since the paper's
release — old links redirect). See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the
full paper-to-code mapping and every deliberate departure.

## Install

```bash
pip install superred-optimizer-autodan-turbo
```

## Usage

```python
from autodan_turbo_optimizer import AutoDANTurboOptimizer

optimizer = AutoDANTurboOptimizer(max_attempts=10, break_score=8.5)
```

The strategy library lives on the optimizer instance and persists across
every run of that instance (not just within one task), matching the paper's
"lifelong" framing — construct a fresh optimizer for an unrelated experiment
if you don't want strategies to carry over.

Wire it into a `Controller` like any other superred optimizer — see the
framework's [architecture overview](https://superred.simonsure.com) and a
target's README (e.g. `superred-target-chatbot`) for a full `Controller.run()`
example.

## Behaviour notes

- **Score-based 3-tier retrieval** replaces upstream's FAISS embedding
  retrieval step (avg strategy score ≥5 or in [2,5) → retrieve and use;
  otherwise cold/`find_new_strategy`), matching upstream's post-FAISS score
  filter's decision boundary without the embedding-index dependency — a
  documented v1 simplification, not a silent behavior change.
- **Refusal filter.** If the attacker's own output looks like a refusal, the
  optimizer falls back to the raw goal text as the jailbreak prompt for that
  turn rather than sending a refusal to the target.
- **Runs in all four adversarial information-access settings**: a cold
  attacker every run when responses aren't visible (settings 1, 2); the full
  lifelong loop (internal scorer + library growth + retrieval) when responses
  are visible (settings 3, 4). External evaluator success early-stops
  settings 2 and 4.
- Single-line score/JSON parsers replace upstream's two-stage wrapper LLM
  calls, halving scorer/summarizer cost per turn.

## License

MIT for this port's code. Prompt templates (attacker ×3, scorer, summarizer)
are reproduced verbatim from the upstream reference implementation
(MIT License, Copyright (c) 2023 Xiaogeng Liu) — see
[ASSUMPTIONS.md](ASSUMPTIONS.md) for exact source files.
