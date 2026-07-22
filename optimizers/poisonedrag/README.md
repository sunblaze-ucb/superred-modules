# poisonedrag-optimizer

A **PoisonedRAG** knowledge-corruption optimizer for the
[superred](https://superred.simonsure.com) red-teaming framework.

PoisonedRAG attacks retrieval-augmented generation (RAG) systems by poisoning
their knowledge base. It crafts a small number of malicious documents designed to
be retrieved for a target question and to steer the model toward an
attacker-chosen answer. The model is never asked to misbehave directly — the
corrupted context does the work.

Reference: Zou et al., "PoisonedRAG: Knowledge Corruption Attacks to
Retrieval-Augmented Generation of Large Language Models,"
[arXiv:2402.07867](https://arxiv.org/abs/2402.07867) (USENIX Security 2025).
Official implementation:
[`sleeepeer/PoisonedRAG`](https://github.com/sleeepeer/PoisonedRAG) (MIT License).
See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the full paper-to-code mapping and every
deliberate departure.

## Install

From the repository root, with the shared virtual environment active:

```bash
pip install -e ../superred        # the framework first
pip install -e optimizers/poisonedrag
```

## Usage

```python
from poisonedrag_optimizer import PoisonedRAGOptimizer

# Defaults match the released code (adv_per_query=5, top_k=5, LM_targeted).
# Set max_attempts=1 for the paper's single-shot poison batch (paper-parity ASR).
optimizer = PoisonedRAGOptimizer(max_attempts=1)

# Load bundled official attack results for a benchmark.
optimizer = PoisonedRAGOptimizer(official_adv_results_dataset="nq")
```

Wire it into a `Controller` like any other superred optimizer — see the
framework's [architecture overview](https://superred.simonsure.com) and a
target's README (e.g. `chatbot`) for a full `Controller.run()` example.

## Assumptions

Plain-language summary; see [ASSUMPTIONS.md](ASSUMPTIONS.md) for the details.

- **Official poison shape.** Poison docs use the black-box `question + "." +
  corpus` form, and the official multi-context RAG wrapper and JSON
  joint-generation prompt are preserved. Bundled official attack results for
  `nq`, `hotpotqa`, and `msmarco` are available.
- **Surface preference.** The optimizer prefers a writable corpus/context surface
  (true database poisoning), then a writable `system_prompt`, then `user_message`,
  then runtime context. User- and system-prompt injection are capability
  fallbacks, not real database poisoning.
- **Budget behaviour.** With no explicit `max_attempts` the optimizer keeps
  attempting until success or the attack becomes undeliverable, letting the
  controller's run budget bound the loop. Set `max_attempts=1` for paper-parity.
- **Not implemented on purpose.** HotFlip is out of scope because superred
  optimizers do not receive retriever weights, tokenizers, gradients, or BEIR
  scores. Retrieval precision/recall stays target- or benchmark-owned.

## License

MIT for this port's code (see [LICENSE](LICENSE)). The upstream reference
implementation, [`sleeepeer/PoisonedRAG`](https://github.com/sleeepeer/PoisonedRAG),
is MIT licensed. The bundled official attack-result datasets originate from that
repository. This port reimplements the attack against superred's event model
rather than vendoring upstream runtime source.
