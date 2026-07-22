# many-shot-optimizer

A **Many-Shot Jailbreaking** single-turn optimizer for the
[superred](https://superred.simonsure.com) red-teaming framework.

Many-shot jailbreaking abuses long context windows. It builds one prompt packed
with many faux user/assistant exchanges in which the assistant always complies
with a harmful request, then appends the real objective as the final user turn.
Faced with a long run of "compliant" examples, the model tends to continue the
pattern and comply too.

Reference: Anil et al., "Many-shot Jailbreaking," Anthropic, April 2024
([research page](https://www.anthropic.com/research/many-shot-jailbreaking),
[PDF](https://www-cdn.anthropic.com/af5633c94ed2beb282f6a53c595eb437e8e7b630/Many_Shot_Jailbreaking__2024_04_02_0936.pdf);
later NeurIPS 2024). There is no official code release; this is an independent
implementation from the paper. See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the full
paper-to-code mapping and every deliberate departure.

## Install

From the repository root, with the shared virtual environment active:

```bash
pip install -e ../superred        # the framework first
pip install -e optimizers/many_shot
```

## Usage

```python
from many_shot_optimizer import ManyShotOptimizer

# Default: one 256-shot attempt (matches the paper's main many-shot setup).
optimizer = ManyShotOptimizer()

# Sweep the number of shots across staged runs.
optimizer = ManyShotOptimizer(example_counts=[32, 64, 128, 256, 400, 512])
```

Wire it into a `Controller` like any other superred optimizer — see the
framework's [architecture overview](https://superred.simonsure.com) and a
target's README (e.g. `chatbot`) for a full `Controller.run()` example.

## Assumptions

Plain-language summary; see [ASSUMPTIONS.md](ASSUMPTIONS.md) for the details.

- **Prompt shape.** A preamble, many repeated `User:` / `Assistant:` examples,
  then the real objective. Bundled examples live in `data/examples.json`; example
  order is shuffled by default (set `random_seed` for a repeatable shuffle).
- **Shot-count sweep is a framework feature.** The default is a single
  256-example attempt. Passing an `example_counts` schedule maps the paper's
  "success vs. number of examples" study onto staged superred runs.
- **Channel mapping.** For a single channel, the whole prompt goes into the first
  user-like controllable. When `system_prompt` is visible, the preamble and
  examples go there and the objective goes into `user_message` (force the
  single-prompt path with `use_system_prompt_when_available=False`).
- **Scoring.** The default evaluator is a refusal classifier (compliance `1.0`,
  refusal `0.0`), matching the paper's attack-success-rate framing; framework
  evaluation, when present, can also mark success. The paper's negative-log-
  likelihood metric needs target token logprobs, which superred does not expose,
  so it is not implemented.

## License

MIT for this port's code (see [LICENSE](LICENSE)). Many-shot jailbreaking was
published by Anthropic without an official code release, so there is no upstream
repository to vendor from; this module is an independent implementation of the
technique described in the paper.
