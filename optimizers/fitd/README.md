# superred-optimizer-fitd

A **FITD (Foot-in-the-Door)** multi-turn jailbreak optimizer for the
[superred](https://superred.simonsure.com) red-teaming framework.

> **Community port — not an official implementation.** This module is an
> unofficial re-implementation of Foot-In-The-Door (Ding et al.) for superred.
> It is not affiliated with, endorsed by, or maintained by the original
> authors. See [ASSUMPTIONS.md](ASSUMPTIONS.md) for every deliberate deviation
> from the paper and reference code.

FITD exploits commitment and consistency. It opens with a benign question, then
walks the model up a ladder of gradually more harmful "bridge" prompts, at each
step asking it to revise its earlier answers a little further. Because the model
has already committed to helping, each small step is easy to accept — the foot
is in the door.

Reference: Ding et al., "Foot-In-The-Door: A Multi-turn Jailbreak for LLMs,"
[arXiv:2502.19820](https://arxiv.org/abs/2502.19820) (EMNLP 2025). Official
implementation:
[`Jinxiaolong1129/Foot-in-the-door-Jailbreak`](https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak).
See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the full paper-to-code mapping and every
deliberate departure.

## Install

```bash
pip install superred-optimizer-fitd
```

## Usage

```python
from fitd_optimizer import FITDOptimizer

# Defaults follow the official code (level=10, max_attempts=5, ...).
optimizer = FITDOptimizer()

# Replay a pre-generated official prompt ladder exactly.
optimizer = FITDOptimizer(prompt_sequence=[...])
```

Wire it into a `Controller` like any other superred optimizer — see the
framework's [architecture overview](https://superred.simonsure.com) and a
target's README (e.g. `superred-target-chatbot`) for a full `Controller.run()` example.

## License

MIT for this port's code (see [LICENSE](LICENSE)). The upstream reference
implementation,
[`Jinxiaolong1129/Foot-in-the-door-Jailbreak`](https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak),
carries no LICENSE file; its authors granted the superred maintainers permission,
in private correspondence, to redistribute the ported prompts and judge text
under MIT terms. That material
is credited in [`NOTICE`](NOTICE), and the MIT text applied to it is reproduced
in
[`LICENSES/Foot-in-the-door-Jailbreak-MIT.txt`](LICENSES/Foot-in-the-door-Jailbreak-MIT.txt).
