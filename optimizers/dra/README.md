# superred-optimizer-dra

A **DRA (Disguise and Reconstruction Attack)** jailbreak optimizer for the
[superred](https://superred.simonsure.com) red-teaming framework.

> **Community port — not an official implementation.** This module is an
> unofficial re-implementation of DRA (Liu et al.) for superred. It is not
> affiliated with, endorsed by, or maintained by the original authors. See
> [ASSUMPTIONS.md](ASSUMPTIONS.md) for every deliberate deviation from the
> paper and reference code.

DRA has two halves. **Disguise** hides the harmful request one character at a
time inside ordinary cover words, where the real character is the one wrapped in
parentheses. **Reconstruction** then asks the target to pull those parenthesized
characters out, rebuild the hidden request, and answer it directly — so the
model assembles and answers the request itself rather than being asked it
openly.

Reference: Liu et al., "Making Them Ask and Answer: Jailbreaking Large Language
Models in Few Queries via Disguise and Reconstruction,"
[arXiv:2402.18104](https://arxiv.org/abs/2402.18104) (USENIX Security 2024).
Official implementation: [`LLM-DRA/DRA`](https://github.com/LLM-DRA/DRA)
(MIT License). See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the full paper-to-code
mapping and every deliberate departure.

## Install

DRA uses [Detoxify](https://github.com/unitaryai/detoxify) to decide which guide
words are sensitive. It is an extra rather than a base dependency, because it
pulls torch and a ~440MB checkpoint for what the code uses as one boolean.

```bash
pip install "superred-optimizer-dra[detoxify]"   # the paper's behaviour
pip install superred-optimizer-dra               # static word list only
```

Without it the optimizer falls back to the official static sensitive-word list,
which is one word (`illegally`). The attack still runs, but nearly every token
then takes the benign branch of the word guide, so the disguise step is much
weaker than the paper's. Prefer the extra, or pass your own token classifier.

## Usage

```python
from dra_optimizer import DraOptimizer

# Defaults follow the official script (T=20 retries, em_t=0.7 exact match,
# toxic_trunc=benign_trunc=0.5).
optimizer = DraOptimizer()

# A visible target model name auto-selects the GPT or Llama suffix family;
# override explicitly if you prefer.
optimizer = DraOptimizer(model="gpt", max_attempts=10)
```

Wire it into a `Controller` like any other superred optimizer — see the
framework's [architecture overview](https://superred.simonsure.com) and a
target's README (e.g. `superred-target-chatbot`) for a full `Controller.run()` example.

## License

MIT for this port's code (see [LICENSE](LICENSE)). The upstream reference
implementation, [`LLM-DRA/DRA`](https://github.com/LLM-DRA/DRA), is MIT licensed.
The bundled `behaviors.json` originates from that repository; its MIT copyright
notice is preserved in [`NOTICE`](NOTICE) and
[`LICENSES/DRA-MIT.txt`](LICENSES/DRA-MIT.txt). This port reimplements the DRA
prompt-generation structure against superred's event model rather than vendoring
upstream source files. Detoxify is a third-party dependency under its own
license.
