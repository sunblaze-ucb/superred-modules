# superred-optimizer-many-shot

A **Many-Shot Jailbreaking** single-turn optimizer for the
[superred](https://superred.simonsure.com) red-teaming framework.

Many-shot jailbreaking abuses long context windows. It builds one prompt packed
with many faux user/assistant exchanges in which the assistant always complies
with a harmful request, then appends the real objective as the final user turn.
Faced with a long run of "compliant" examples, the model tends to continue the
pattern and comply too.

Reference: Anil et al., "Many-shot Jailbreaking," Anthropic, April 2024
([research page](https://www.anthropic.com/research/many-shot-jailbreaking),
[PDF](https://www-cdn.anthropic.com/af5633c94ed2beb282f6a53c595eb437e8e7b630/Many_Shot_Jailbreaking__2024_04_02_0936.pdf);
later NeurIPS 2024). There is no official code release; this is an independent
implementation from the paper. See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the full
paper-to-code mapping and every deliberate departure.

## Install

```bash
pip install superred-optimizer-many-shot
```

## Usage

```python
from many_shot_optimizer import ManyShotOptimizer

# Default: one 256-shot attempt (matches the paper's main many-shot setup).
optimizer = ManyShotOptimizer()

# Sweep the number of shots across staged runs.
optimizer = ManyShotOptimizer(example_counts=[32, 64, 128, 256, 400, 512])
```

Wire it into a `Controller` like any other superred optimizer — see the
framework's [architecture overview](https://superred.simonsure.com) and a
target's README (e.g. `superred-target-chatbot`) for a full `Controller.run()` example.

## License

MIT for this port's code (see [LICENSE](LICENSE)). Many-shot jailbreaking was
published by Anthropic without an official code release, so there is no upstream
repository to vendor from; this module is an independent implementation of the
technique described in the paper.
