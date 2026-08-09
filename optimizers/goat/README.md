# superred-optimizer-goat

A **GOAT (Generative Offensive Agent Tester)** multi-turn jailbreak optimizer
for the [superred](https://superred.simonsure.com) red-teaming framework.

> **Community port — not an official implementation.** This module is an
> unofficial re-implementation of GOAT (Pavlova et al.) for superred. It is not
> affiliated with, endorsed by, or maintained by the original authors. See
> [ASSUMPTIONS.md](ASSUMPTIONS.md) for every deliberate deviation from the
> paper and reference code.

GOAT plays the attacker in an automated, multi-turn conversation: an attacker
LLM reasons about which adversarial technique to apply next, then sends a
message to the target, using the target's own replies (when visible) to adapt.
One superred run is one independent `K`-turn attack conversation; multiple
runs give the paper's ASR@k metric.

Reference: Pavlova et al., "Automated Red Teaming with GOAT: the Generative
Offensive Agent Tester," [arXiv:2410.01606](https://arxiv.org/abs/2410.01606)
(Meta, 2024). See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the full paper-to-code
mapping and every deliberate departure.

## Install

```bash
pip install superred-optimizer-goat
```

## Usage

```python
from goat_optimizer import GOATOptimizer, ATTACKS, REFUSAL_SUPPRESSION

# Default: all seven Table 1 attacks available to the attacker LLM at once
# (the paper's main-result configuration).
optimizer = GOATOptimizer(max_turns=5, max_attempts=10)

# Per-attack ablation: restrict the attacker to a single technique.
optimizer = GOATOptimizer(attacks=(REFUSAL_SUPPRESSION,))
```

Wire it into a `Controller` like any other superred optimizer — see the
framework's [architecture overview](https://superred.simonsure.com) and a
target's README (e.g. `superred-target-chatbot`) for a full `Controller.run()`
example.

## Behaviour notes

- **One run = one attempt.** Each run builds a fresh attacker conversation
  (`RunStartEvent`); attempts are independent, not a continuation of the
  previous run's conversation.
- **Structured attacker output.** The attacker LLM emits
  `{observation, thought, strategy, response}` as one JSON object per turn (so
  the `response` slot can be lifted off its chain-of-thought reasoning
  cleanly); markdown fences around the JSON are tolerated. On a parse failure
  the optimizer does not retry — it substitutes a benign fallback message for
  that single turn and continues, rather than crashing the attempt.
- **Runs in all four adversarial information-access settings** (response
  visible/hidden × feedback available/unavailable) without a setting flag,
  driven purely by which controllables/observables the `Controller` scope puts
  in view. `max_no_response_runs` bounds cost in the blind (no-response)
  settings.
- **No in-loop judge.** Unlike some jailbreak optimizers, GOAT's own loop does
  not score its own success; it delegates to `RunEndEvent.evaluation.success`
  (the paper has no in-loop judge either — success is measured out of band).
- Attack examples (paper Table 1's `aex` column) are intentionally omitted:
  the paper withholds them "to effectively balance disclosure of reproducible
  results and enabling of malicious actors," so this port does not fabricate
  substitutes.

## License

MIT. See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the upstream reference; the
paper's own authors (Meta) never released official code, so only their
prompts/definitions (Table 1, Figs A.1-A.3) are reproduced (as text, not code)
and reimplemented here as new, original Python. Two third-party, unofficial
GOAT probes exist independently — NVIDIA's [`garak`](https://github.com/NVIDIA/garak)
scanner (`garak/probes/goat.py`, Apache-2.0) and Praetorian's
[`augustus`](https://github.com/praetorian-inc/augustus) scanner
(`goat.Goat`, Apache-2.0) — this port does not use, vendor, or derive from
either implementation.
