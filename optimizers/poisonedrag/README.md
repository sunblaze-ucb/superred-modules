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

```bash
pip install poisonedrag-optimizer
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
target's README (e.g. `superred-target-chatbot`) for a full `Controller.run()` example.

## License

MIT for this port's code (see [LICENSE](LICENSE)). The upstream reference
implementation, [`sleeepeer/PoisonedRAG`](https://github.com/sleeepeer/PoisonedRAG),
is MIT licensed. The bundled official attack-result datasets originate from that
repository. This port reimplements the attack against superred's event model
rather than vendoring upstream runtime source.
