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
installed automatically. From the repository root, with the shared virtual
environment active:

```bash
pip install -e ../superred        # the framework first
pip install -e optimizers/dra
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
target's README (e.g. `chatbot`) for a full `Controller.run()` example.

## Assumptions

Plain-language summary; see [ASSUMPTIONS.md](ASSUMPTIONS.md) for the details.

- **One randomized prompt per run.** Each superred run builds one randomized DRA
  prompt. After a visible response, a refusal lowers `toxic_trunc` by 0.1 and
  missing target-word coverage raises `benign_trunc` by 0.1, matching the
  official retry loop.
- **User prompt is the target.** The DRA prompt goes into `user_message`. If
  `system_prompt` is writable, the official `You are a helpful assistant.` system
  prompt is added; the model response is never edited.
- **Success scoring.** Framework evaluation is authoritative when present.
  Otherwise the official lightweight checks apply (refusal-prefix + exact-match
  coverage), with an optional GPT-style judge fallback that only counts a rating
  of `10` as success (disable with `use_llm_judge_when_no_feedback=False`).
- **Code-faithful, not paper-broad.** The paper defines exact match more broadly
  (word overlap or embedding similarity); this port follows the released
  script's target-word coverage for comparability with the reference code.

## License

MIT for this port's code (see [LICENSE](LICENSE)). The upstream reference
implementation, [`LLM-DRA/DRA`](https://github.com/LLM-DRA/DRA), is MIT licensed.
This port reimplements the DRA prompt-generation structure against superred's
event model rather than vendoring upstream source files. Detoxify is a
third-party dependency under its own license.
