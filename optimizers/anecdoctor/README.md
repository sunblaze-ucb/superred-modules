# anecdoctor

Anecdoctor misinformation-elicitation optimizer, ported from
[PyRIT](https://github.com/microsoft/PyRIT)'s `AnecdoctorGenerator`
(`pyrit/executor/promptgen/anecdoctor.py`, pinned at commit `2016c4a`).

Technique: Cuevas, Dash, Nayak, Vann, and Daepp, "Anecdoctoring: Automated
Red-Teaming Across Language and Place," EMNLP 2025
([arXiv:2509.19143](https://arxiv.org/abs/2509.19143)), Microsoft Research.

## What it does

Anecdoctor coaxes a target into producing misinformation-style content by
framing a request around a set of example "claims". It runs in one of two modes:

- **few-shot** (default): the claims are formatted into a few-shot block and
  wrapped in a framing template that asks for content of a given type in a given
  language.
- **knowledge-graph**: the attacker LLM first builds a text "knowledge graph"
  from the claims, and that graph — rather than the raw claims — is wrapped in
  the framing template. Upstream reports the KG makes the generated content more
  coherent and targeted, especially across languages and cultures.

Either way the optimizer emits **one framed request** on a single free-text user
surface, once per run. The knowledge-graph build is the only model call and is
routed through the constrained `self.llm`. The framing / KG-build prompt bodies
are loaded from the byte-identically vendored PyRIT YAML templates and are never
reproduced in authored code.

This is a **misinformation-elicitation** technique (an "A1" content-generation
attacker), not a classic refusal-bypass jailbreak — it asks the target to
generate false/misleading content rather than to reveal a withheld harmful
how-to. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

## Usage

```python
from anecdoctor_optimizer import AnecdoctorOptimizer

# few-shot mode (no attacker LLM call)
opt = AnecdoctorOptimizer(language="english", content_type="news article")

# knowledge-graph mode (one attacker-LLM call at setup builds the KG)
opt_kg = AnecdoctorOptimizer(use_knowledge_graph=True)
```

Constructor knobs: `use_knowledge_graph`, `language`, `content_type`,
`example_claims` (defaults to an authored synthetic, benign set — never copied
harmful payloads), `include_goal_as_claim`, `max_kg_retries`. The SuperRed Goal
is mapped onto upstream's `evaluation_data` as the lead claim so the specific
claim/topic drives generation (see ASSUMPTIONS.md).

## Dependencies

`superred`, `pyyaml`. No heavy dependencies: Anecdoctor is LLM-driven — the
"knowledge graph" is a text structure built via prompts, so there is no
`torch`/`transformers`/`nltk`/`networkx`.

## Faithfulness

See [ASSUMPTIONS.md](ASSUMPTIONS.md) for every deliberate deviation. The main
ones: upstream sets the framing as the target's *system* prompt and sends the
examples/KG as a separate user message — this port folds both into one injected
user turn and never writes a system prompt; and the Goal→claim mapping, which
has no direct upstream analogue.
