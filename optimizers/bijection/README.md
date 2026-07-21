# superred-optimizer-bijection

A **Bijection Learning** jailbreak optimizer for the
[superred](https://superred.simonsure.com) red-teaming framework.

Bijection learning is a black-box, model-agnostic jailbreak: it randomizes a
bijective character map ("Language Alpha"), teaches the target model that map
in-context with a few encode/decode examples, then sends the harmful query
encoded under the same map. Capable models follow the encoding into the
encoded answer space, sidestepping refusal training that was only ever
trained on plain English. One superred run samples and tries one fresh random
bijection; multiple runs give best-of-N, as in the paper.

Reference: Huang, Li, Tang, "Endless Jailbreaks with Bijection Learning,"
[arXiv:2410.01294](https://arxiv.org/abs/2410.01294) (ICLR 2025, Haize Labs).
See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the full paper-to-code mapping and
every deliberate departure.

## Install

```bash
pip install superred-optimizer-bijection
```

## Usage

```python
from bijection_optimizer import BijectionOptimizer

# Auto-tunes codomain + fixed_size from the in-scope `model` observable
# (paper Table 1: digit codomain for stronger models, letter for weaker ones).
optimizer = BijectionOptimizer(max_attempts=6)

# Explicit configuration always wins over auto-tune.
optimizer = BijectionOptimizer(bijection_type="digit", fixed_size=8, num_digits=2)
```

Wire it into a `Controller` like any other superred optimizer — see the
framework's [architecture overview](https://superred.simonsure.com) and a
target's README (e.g. `superred-target-chatbot`) for a full `Controller.run()`
example.

## Behaviour notes

- **Single-turn, packed prompt.** Upstream can use multi-turn teaching;
  superred has no assistant-prefill primitive, so a multi-turn version would
  cost ~11x more model calls for no accuracy upside here, and this port stays
  single-turn by design (see ASSUMPTIONS.md).
- **Two-channel split when `system_prompt` is in scope**: the alphabet-table
  teaching content goes in the system prompt, the encoded query in the user
  message, matching the paper's stronger configuration.
- **Response-blind construction.** Bijection construction never depends on
  seeing the target's previous response (matches the paper); in
  response-visible settings, only the early-stop decision uses
  `RunEndEvent.evaluation.success`.
- `target_controllable_name` lets you point the optimizer at a non-`chatbot`
  target's user-message controllable by name.

## License

MIT for this port's code. The upstream reference implementation,
[`haizelabs/bijection-learning`](https://github.com/haizelabs/bijection-learning),
carries **no license file** (default all-rights-reserved). This port does not
vendor or copy any of its files — `src/bijection_optimizer/bijection.py` is an
independent reimplementation of the bijection construction/encode/decode
*algorithm* described in the paper and observable in the upstream repo (new
classes, new structure, new docstrings), not a copy of upstream's source text.
If upstream publishes a license in the future, or if any literal upstream code
is ever vendored here, this notice should be revisited.
