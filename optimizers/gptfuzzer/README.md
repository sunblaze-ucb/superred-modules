# superred-optimizer-gptfuzzer

A **GPTFuzzer** jailbreak optimizer for the
[superred](https://superred.simonsure.com) red-teaming framework.

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
If the weights or dependencies are unavailable, the optimizer can fall back to a
lightweight refusal-string classifier; set `allow_predictor_fallback=False` to
require the official model and fail fast instead.

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
