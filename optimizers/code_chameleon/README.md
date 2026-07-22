# code-chameleon-optimizer

A **CodeChameleon** jailbreak optimizer for the
[superred](https://superred.simonsure.com) red-teaming framework.

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
pip install code-chameleon-optimizer
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

**No license is granted yet** (see [LICENSE](LICENSE)). The upstream reference
implementation,
[`huizhang-L/CodeChameleon`](https://github.com/huizhang-L/CodeChameleon),
carries **no license file** (default all-rights-reserved), and this module
bundles prompt templates reproduced from it. Because there is no upstream
license to redistribute that material under, this module is withheld pending the
upstream author's permission and is not released under MIT (unlike other superred
optimizers). Once permission is obtained or the material is fully replaced with
original text, a proper license will be applied here.
