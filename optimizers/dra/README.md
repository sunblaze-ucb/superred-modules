# dra-optimizer

A **DRA (Disguise and Reconstruction Attack)** jailbreak optimizer for the
[superred](https://superred.simonsure.com) red-teaming framework.

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

DRA depends on [Detoxify](https://github.com/unitaryai/detoxify) to decide which
guide words are sensitive; it is a normal dependency of this package and is
installed automatically.

```bash
pip install dra-optimizer
```

If Detoxify cannot load in a constrained environment, the optimizer falls back
to the official static sensitive-word list so the attack can still run.

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
This port reimplements the DRA prompt-generation structure against superred's
event model rather than vendoring upstream source files. Detoxify is a
third-party dependency under its own license.
