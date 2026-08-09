# superred-optimizer-tap

A **Tree of Attacks with Pruning (TAP)** jailbreak optimizer for the
[superred](https://superred.simonsure.com) red-teaming framework.

> **Community port — not an official implementation.** This module is an
> unofficial re-implementation of TAP (Mehrotra et al.) for superred. It is not
> affiliated with, endorsed by, or maintained by the original authors. See
> [ASSUMPTIONS.md](ASSUMPTIONS.md) for every deliberate deviation from the
> paper and reference code.

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

```bash
pip install superred-optimizer-tap
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
target's README (e.g. `superred-target-chatbot`) for a full `Controller.run()` example.

## License

MIT for this port's code (see [LICENSE](LICENSE)). The upstream reference
implementation, [`RICommunity/TAP`](https://github.com/RICommunity/TAP), is MIT
licensed. This port reimplements the TAP search and prompts against superred's
event model rather than vendoring upstream source files.
