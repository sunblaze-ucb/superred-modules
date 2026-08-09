# superred-optimizer-gptfuzzer

A **GPTFuzzer** jailbreak optimizer for the
[superred](https://superred.simonsure.com) red-teaming framework.

> **Community port — not an official implementation.** This module is an
> unofficial re-implementation of GPTFuzzer (Yu et al.) for superred. It is not
> affiliated with, endorsed by, or maintained by the original authors. See
> [ASSUMPTIONS.md](ASSUMPTIONS.md) for every deliberate deviation from the
> paper and reference code.

GPTFuzzer treats jailbreaking as fuzzing. It starts from a pool of human-written
jailbreak templates, mutates them (crossover, expand, generate-similar,
rephrase, shorten), inserts the harmful goal in place of the template's
`[INSERT PROMPT HERE]` slot, and uses an MCTS-Explore selector plus a RoBERTa
success classifier to keep and grow the templates that work.

Reference: Yu et al., "GPTFUZZER: Red Teaming Large Language Models with
Auto-Generated Jailbreak Prompts,"
[arXiv:2309.10253](https://arxiv.org/abs/2309.10253) (2023). Official
implementation: [`sherdencooper/GPTFuzz`](https://github.com/sherdencooper/GPTFuzz)
(MIT License). See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the full paper-to-code
mapping and every deliberate departure.

## Install

GPTFuzzer's default success classifier is the official `hubert233/GPTFuzz`
RoBERTa model, so this package depends on `torch` and `transformers`; they are
installed automatically.

```bash
pip install superred-optimizer-gptfuzzer
```

The classifier weights are downloaded lazily on the first response-visible run.
If the weights or dependencies are unavailable, the optimizer falls back to a
lightweight refusal-string classifier and logs a warning; set
`allow_predictor_fallback=False` to require the official model and fail the task
instead.

The fallback is scoped to one task, so a host that can never reach the weights
would quietly degrade every task rather than fail. Check the scorer once before
a sweep:

```python
from gptfuzzer_optimizer import RoBERTaPredictor

RoBERTaPredictor().load()  # raises PredictorUnavailableError if unusable
```

## Usage

```python
from gptfuzzer_optimizer import GPTFuzzerOptimizer

# Default budgets follow the official runner (max_query=1000, max_jailbreak=1,
# energy=1) — one mutation, one query per run.
optimizer = GPTFuzzerOptimizer()
```

Wire it into a `Controller` like any other superred optimizer — see the
framework's [architecture overview](https://superred.simonsure.com) and a
target's README (e.g. `superred-target-chatbot`) for a full `Controller.run()` example.

## License

MIT for this port's code (see [LICENSE](LICENSE)). The upstream reference
implementation, [`sherdencooper/GPTFuzz`](https://github.com/sherdencooper/GPTFuzz),
is MIT licensed. The bundled seed templates originate from that repository's
`GPTFuzzer.csv`; its MIT copyright notice is preserved in [`NOTICE`](NOTICE) and
[`LICENSES/GPTFuzz-MIT.txt`](LICENSES/GPTFuzz-MIT.txt). `torch`, `transformers`,
and the `hubert233/GPTFuzz` model are third-party components under their own
licenses.
