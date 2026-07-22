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

From the repository root, with the shared virtual environment active:

```bash
pip install -e ../superred        # the framework first
pip install -e optimizers/code_chameleon
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
target's README (e.g. `chatbot`) for a full `Controller.run()` example.

## Assumptions

Plain-language summary; see [ASSUMPTIONS.md](ASSUMPTIONS.md) for the details.

- **Official encryption rules only.** Supported rules are exactly the upstream
  names: `binary_tree`, `reverse`, `odd_even`, `length`, and `none`. Prompt
  templates keep the official wording verbatim (including small typos like
  `privoded`) because wording can change attack results.
- **One rule per run.** One superred run is one CodeChameleon attempt with one
  rule. By default there is a single attempt; pass several `encrypt_rules` to
  try each in order across staged runs.
- **User prompt is the target.** The CodeChameleon prompt goes into
  `user_message` (or the first user-like controllable). If `system_prompt` is
  writable, the official GPT-adapter system prompt is added there too; the model
  response is never modified.
- **Success scoring.** When superred reports an evaluation it is authoritative.
  Otherwise the official GPT-4-style CodeChameleon judge scores the response 1–5
  and only a top score of `5` counts as success.

## License

MIT for this port's code (see [LICENSE](LICENSE)). The upstream reference
implementation,
[`huizhang-L/CodeChameleon`](https://github.com/huizhang-L/CodeChameleon),
carries **no license file** (default all-rights-reserved). This port does not
vendor upstream source; the encryption rules and prompt templates are
reconstructed from the paper and the publicly observable repository. If upstream
publishes a license in the future, this notice should be revisited.
