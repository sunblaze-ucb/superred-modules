# fitd-optimizer

A **FITD (Foot-in-the-Door)** multi-turn jailbreak optimizer for the
[superred](https://superred.simonsure.com) red-teaming framework.

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
pip install fitd-optimizer
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

**No license is granted yet** (see [LICENSE](LICENSE)). The upstream reference
implementation,
[`Jinxiaolong1129/Foot-in-the-door-Jailbreak`](https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak),
carries **no license file** (default all-rights-reserved), and this module
bundles prompts and judge text reproduced from it. Because there is no upstream
license to redistribute that material under, this module is withheld pending the
upstream author's permission and is not released under MIT (unlike other superred
optimizers). Once permission is obtained or the material is fully replaced with
original text, a proper license will be applied here.
