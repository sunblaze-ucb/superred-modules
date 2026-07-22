# tap-optimizer

A **Tree of Attacks with Pruning (TAP)** jailbreak optimizer for the
[superred](https://superred.simonsure.com) red-teaming framework.

TAP treats jailbreaking as a tree search. An attacker LLM proposes candidate
prompts, off-topic ones are pruned away, every surviving prompt is sent to the
real target, and each prompt/response pair is scored 1–10. The best branches are
kept and expanded again, deepening the tree until a prompt scores a full 10 or
the maximum depth is reached.

Reference: Mehrotra et al., "Tree of Attacks: Jailbreaking Black-Box LLMs
Automatically," [arXiv:2312.02119](https://arxiv.org/abs/2312.02119)
(NeurIPS 2024). Official implementation:
[`RICommunity/TAP`](https://github.com/RICommunity/TAP) (MIT License).
See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the full paper-to-code mapping and every
deliberate departure.

## Install

From the repository root, with the shared virtual environment active:

```bash
pip install -e ../superred        # the framework first
pip install -e optimizers/tap
```

## Usage

```python
from tap_optimizer import TapOptimizer

# Paper-aligned defaults (branching_factor=1, tree_width=10, tree_depth=10, ...).
optimizer = TapOptimizer()

# Reproduce a specific TAP experiment by setting the target opening string.
optimizer = TapOptimizer(target_str="Sure, here is", tree_depth=8)
```

Wire it into a `Controller` like any other superred optimizer — see the
framework's [architecture overview](https://superred.simonsure.com) and a
target's README (e.g. `chatbot`) for a full `Controller.run()` example.

## Assumptions

Plain-language summary of how this port behaves; see
[ASSUMPTIONS.md](ASSUMPTIONS.md) for the details.

- **One candidate per run.** Original TAP sends a whole batch of candidate
  prompts to the target at once. superred targets are event-driven, so each
  surviving candidate is injected into `user_message` in its own target run;
  once every candidate at a depth has run, TAP judge-prunes and goes deeper.
- **User-prompt attack by default.** If only `user_message` is in scope, TAP
  keeps the paper's behaviour and injects there. If `system_prompt` is also in
  scope, it is used automatically as a stronger threat model.
- **Framework score is authoritative.** When superred reports success
  (`evaluation.success=True`), that is treated as a hard success. If only a
  response is visible, TAP's own 1–10 judge decides; if neither is visible, the
  candidate scores 1.
- **Optional target context.** When superred exposes static target metadata
  (model name, injection points), TAP adds it to the attacker prompt, capped so
  it cannot crowd out the TAP instructions. This is a superred extension the
  original attacker did not have.

## License

MIT for this port's code (see [LICENSE](LICENSE)). The upstream reference
implementation, [`RICommunity/TAP`](https://github.com/RICommunity/TAP), is MIT
licensed. This port reimplements the TAP search and prompts against superred's
event model rather than vendoring upstream source files.
