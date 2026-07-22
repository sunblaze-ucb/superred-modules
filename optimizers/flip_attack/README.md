# flip-attack-optimizer

A **FlipAttack** single-turn jailbreak optimizer for the
[superred](https://superred.simonsure.com) red-teaming framework.

FlipAttack scrambles the harmful task by flipping it — reversing characters or
words — and then teaches the target, in the same prompt, how to unflip and
follow it. The scrambled task slips past refusal filters that only recognize
plain text, while a capable model still restores and executes it. It is a pure
text transform: no attacker LLM is needed to build the prompt.

Reference: Liu et al., "FlipAttack: Jailbreak LLMs via Flipping,"
[arXiv:2410.02832](https://arxiv.org/abs/2410.02832) (ICML 2025). Official
implementation: [`yueliu1999/FlipAttack`](https://github.com/yueliu1999/FlipAttack).
See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the full paper-to-code mapping and every
deliberate departure.

## Install

```bash
pip install flip-attack-optimizer
```

## Usage

```python
from flip_attack_optimizer import FlipAttackOptimizer

# Default rotation starts with FCS (the paper's default mode).
optimizer = FlipAttackOptimizer()

# Paper-faithful single-mode behaviour: configure exactly one flip mode.
optimizer = FlipAttackOptimizer(flip_modes=["FCS"])
```

Wire it into a `Controller` like any other superred optimizer — see the
framework's [architecture overview](https://superred.simonsure.com) and a
target's README (e.g. `superred-target-chatbot`) for a full `Controller.run()` example.

## License

MIT for this port's code (see [LICENSE](LICENSE)). The upstream reference
implementation, [`yueliu1999/FlipAttack`](https://github.com/yueliu1999/FlipAttack),
carries **no license file** (default all-rights-reserved). This port does not
vendor upstream source; the flip transforms and prompt structure are
reconstructed from the paper and the publicly observable repository. If upstream
publishes a license in the future, this notice should be revisited.
