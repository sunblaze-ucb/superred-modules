# superred-optimizer-gepa

A **GEPA (Genetic-Pareto) reflective prompt evolution** optimizer for the
[superred](https://superred.simonsure.com) red-teaming framework.

GEPA treats the attack prompt as a single evolving candidate. Each superred
run rolls out one candidate; at the end of a run, a separate reflection LM
call inspects that rollout (score, target reply, evaluator feedback) and
proposes a revised candidate for the next run to try. Across many runs this
grows a small pool of candidates by reflective mutation rather than
gradient-free search or manual prompt engineering.

Reference: Agrawal et al., "GEPA: Reflective Prompt Evolution Can Outperform
Reinforcement Learning," [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)
(ICLR 2026). The reflective meta-prompt is reproduced verbatim from the
official [`gepa-ai/gepa`](https://github.com/gepa-ai/gepa) reference
implementation (MIT-licensed) — see [ASSUMPTIONS.md](ASSUMPTIONS.md) for the
exact source files and every deliberate departure this superred port makes
(all forced by superred's single-`Goal`, single-instance session model, where
GEPA's paper assumes a full training set to batch and Pareto-select over).

For an agent-target variant of this same reflective loop (surface discovery,
tool-return injection), see `superred-optimizer-gepa-agentic`.

## Install

```bash
pip install superred-optimizer-gepa
```

## Usage

```python
from gepa_optimizer import GEPAOptimizer

optimizer = GEPAOptimizer(max_attempts=20)
```

Wire it into a `Controller` like any other superred optimizer — see the
framework's [architecture overview](https://superred.simonsure.com) and a
target's README (e.g. `superred-target-chatbot`) for a full `Controller.run()`
example.

## Behaviour notes

- **One run = one rollout of one candidate.** The seed candidate is the
  adversarial goal's own description; each later candidate is proposed by the
  reflection LM from the best-scoring candidate's most recent rollout.
- **Runs in all four adversarial information-access settings** without a
  setting flag. Settings without a score signal collapse candidate selection
  to latest-wins on ties; settings with feedback drive score-based parent
  selection and success early-stop. `max_no_signal_runs` (disabled by default)
  bounds cost in the score-blind settings.
- **No acceptance test / single-instance Pareto.** The paper's acceptance test
  replays a candidate on a fixed minibatch before keeping it; superred's
  single-`Goal`, single-attempt-per-run model makes that replay impossible, so
  every rolled-out candidate enters the pool and parent selection collapses to
  current-best. Documented as a deliberate departure, not an oversight.
- Reflection parse failures (malformed or missing fenced-code-block output) do
  not crash the run — the optimizer logs a warning and skips that mutation,
  keeping the previous best candidate as the parent.

## License

MIT for this port's code. The reflective meta-prompt text is reproduced
verbatim from `gepa-ai/gepa` (MIT License, Copyright (c) 2025 Lakshya A
Agrawal and the GEPA contributors) — see the attribution comment in
`src/gepa_optimizer/prompts.py` and [ASSUMPTIONS.md](ASSUMPTIONS.md).
