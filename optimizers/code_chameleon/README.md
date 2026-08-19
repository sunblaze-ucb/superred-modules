# superred-optimizer-code-chameleon

A **CodeChameleon** jailbreak optimizer for the
[superred](https://superred.simonsure.com) red-teaming framework.

> **Community port — not an official implementation.** This module is an
> unofficial re-implementation of CodeChameleon (Lv et al.) for superred. It is
> not affiliated with, endorsed by, or maintained by the original authors. See
> [ASSUMPTIONS.md](ASSUMPTIONS.md) for every deliberate deviation from the
> paper and reference code.

CodeChameleon hides the unsafe request by encrypting it, then hands the target a
matching decryption function and asks it to complete a `ProblemSolver` class.
Because the harmful text never appears in plain form, the target's refusal
training — trained on plain English — tends to miss it, and a capable model
decrypts and answers inside the code-completion frame.

Reference: Lv et al., "CodeChameleon: Personalized Encryption Framework for
Jailbreaking Large Language Models,"
[arXiv:2402.16717](https://arxiv.org/abs/2402.16717) (Feb 2024). Official
implementation:
[`huizhang-L/CodeChameleon`](https://github.com/huizhang-L/CodeChameleon).
See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the full paper-to-code mapping and every
deliberate departure.

## Install

```bash
pip install superred-optimizer-code-chameleon
```

## Usage

```python
from code_chameleon_optimizer import CodeChameleonOptimizer

# Default: one attempt with the `binary_tree` rule and the code-style template
# (the official README example, and the strongest average rule in the paper).
optimizer = CodeChameleonOptimizer()

# Stage several official encryption rules, one per run.
optimizer = CodeChameleonOptimizer(encrypt_rules=["reverse", "odd_even", "length"])
```

Wire it into a `Controller` like any other superred optimizer — see the
framework's [architecture overview](https://superred.simonsure.com) and a
target's README (e.g. `superred-target-chatbot`) for a full `Controller.run()` example.

## License

MIT for this port's code (see [LICENSE](LICENSE)). The upstream reference
implementation,
[`huizhang-L/CodeChameleon`](https://github.com/huizhang-L/CodeChameleon), is MIT
licensed. The prompt templates, encryption rules, and judge prompt ported from it
are credited in [`NOTICE`](NOTICE), and its MIT copyright notice is preserved in
[`LICENSES/CodeChameleon-MIT.txt`](LICENSES/CodeChameleon-MIT.txt).
