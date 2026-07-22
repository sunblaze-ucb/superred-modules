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

From the repository root, with the shared virtual environment active:

```bash
pip install -e ../superred        # the framework first
pip install -e optimizers/flip_attack
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
target's README (e.g. `chatbot`) for a full `Controller.run()` example.

## Assumptions

Plain-language summary; see [ASSUMPTIONS.md](ASSUMPTIONS.md) for the details.

- **Flip modes.** Four official modes are supported: `FCS`, `FWO`, `FCW`, and
  `FMM`. Prompt text and structure follow the official code (the triple `NEVER`,
  the Step 1/2 layout, and the hardcoded few-shot examples).
- **Mode rotation is a framework extension.** By default the optimizer rotates
  through the four modes, one per run. The original attack uses a single mode
  per instance — set `flip_modes=["FCS"]` for that paper-faithful behaviour.
- **Channel mapping.** For a single user-like channel, decode instructions and
  the transformed task are folded into one payload. When a `system_prompt`
  channel is visible, the instructions go there and the user message carries only
  the transformed task.
- **Scoring.** `asr_gpt` (default) uses the official judge prompt and counts only
  a rating of `10` as success; `asr_dict` uses the official refusal-string list.
  The optimizer only calls the LLM to *score* — never to build the attack.

## License

MIT for this port's code (see [LICENSE](LICENSE)). The upstream reference
implementation, [`yueliu1999/FlipAttack`](https://github.com/yueliu1999/FlipAttack),
carries **no license file** (default all-rights-reserved). This port does not
vendor upstream source; the flip transforms and prompt structure are
reconstructed from the paper and the publicly observable repository. If upstream
publishes a license in the future, this notice should be revisited.
