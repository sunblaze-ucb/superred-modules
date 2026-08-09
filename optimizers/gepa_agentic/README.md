# superred-optimizer-gepa-agentic

An **agent-target variant of GEPA reflective prompt evolution** for the
[superred](https://superred.simonsure.com) red-teaming framework.

> **Community port — not an official implementation.** This module is an
> unofficial re-implementation of GEPA (Agrawal et al.) for superred. It is not
> affiliated with, endorsed by, or maintained by the original authors. See
> [ASSUMPTIONS.md](ASSUMPTIONS.md) for every deliberate deviation from the
> paper and reference code.

This keeps GEPA's core reflective-mutation loop — one evolving textual
candidate, refined between runs by a reflection LM call on the best-scoring
rollout so far — but adds an agentic *delivery* policy on top: at
`initialize()` it inspects the in-scope controllables, classifies which ones
look like agent content surfaces (tool returns, retrieved context, memory
records, web/document content — AgentDojo-style `read__...`, inspect-agent
`tool:<name>`, skill aliases, and generic memory/RAG/web hints all count as
weak signals), builds a deterministic per-run injection plan across whichever
surfaces actually fire (capped per run), and falls back to the plain prompt
channel only when no content surface is available. Reflection also gets
richer scoped context than the base optimizer: which surfaces were selected
and observed, preserved tool returns, static observables, and a bounded slice
of the agent trace.

This is a **separate package** from `superred-optimizer-gepa` — the chatbot
GEPA implementation there is intentionally left unchanged; see its
[ASSUMPTIONS.md](../gepa/ASSUMPTIONS.md) for the base algorithm and
[this package's ASSUMPTIONS.md](ASSUMPTIONS.md) for the agentic delivery
layer and what is deliberately deferred (full Pareto-frontier maintenance,
full tool catalog register/replace).

Reference: Agrawal et al., "GEPA: Reflective Prompt Evolution Can Outperform
Reinforcement Learning," [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)
(ICLR 2026); official reference implementation
[`gepa-ai/gepa`](https://github.com/gepa-ai/gepa) (MIT License).

## Install

```bash
pip install superred-optimizer-gepa-agentic
```

## Usage

```python
from gepa_agentic_optimizer import GEPAAgenticOptimizer

optimizer = GEPAAgenticOptimizer(
    max_attempts=20,
    max_content_injections_per_run=3,
)
```

Wire it into a `Controller` against an agent target (e.g. `agentdojo_target`,
`inspect_agent_target`) — see the framework's
[architecture overview](https://superred.simonsure.com) for a full
`Controller.run()` example.

## Behaviour notes

- **Surface naming is a weak signal, not a contract.** Detection also uses
  PostCall metadata, value type, and runtime content shape, so it degrades
  gracefully against agent targets that don't follow the AgentDojo/inspect-agent
  naming conventions.
- **GEPA-style strict-improvement acceptance**: a reflected candidate only
  stays in the pool if it succeeded or scored strictly better than its
  parent, closer to the paper's acceptance test than the base `gepa`
  package's single-instance collapse (still not a full replay-based
  acceptance test, since superred has no fixed minibatch to replay).
- `tool_catalog_rewrite_doc` gives a minimal text-rewrite capability when a
  visible tool catalog identifies a content/read/search-like tool; full
  register/replace of tool definitions is a documented follow-up, not
  implemented, because it needs a stricter JSON contract than this optimizer
  currently assumes.

## License

MIT for this port's code. The reflective meta-prompt text is its own copy of
the same template `superred-optimizer-gepa` reproduces (both packages copy it
independently rather than one importing the other's), verbatim from
`gepa-ai/gepa` (MIT License, Copyright (c) 2025 Lakshya A Agrawal and the GEPA
contributors) — see the attribution comment in
`src/gepa_agentic_optimizer/prompts.py`.
