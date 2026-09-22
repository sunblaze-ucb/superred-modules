# Assumptions and deviations — anecdoctor

Ported from PyRIT's `AnecdoctorGenerator` (`pyrit/executor/promptgen/anecdoctor.py`)
at commit `2016c4a`. Deviations from upstream, and why each is necessary or safe.

## Threat model: misinformation elicitation, not a refusal-bypass jailbreak

Anecdoctor is a **misinformation-generation** technique — an "A1"
content-generation attacker in the project's taxonomy — rather than a classic
jailbreak. It does not try to bypass a refusal to obtain a withheld harmful
how-to; it frames a request so the target produces false or misleading content
(of a chosen `content_type`, in a chosen `language`) that imitates a set of
example claims. Success is "the target generated the framed misinformation
content", which the security claim's judge grades — not an attacker self-score
(this port runs no internal judge; upstream has none either).

## Goal → claim mapping (no direct upstream analogue)

Upstream takes `evaluation_data` — a list of claims in ClaimsReview format —
directly from the caller, and its framing prompts are parameterised only by
`language` and `type`. The specific claim/topic can therefore influence the
output **only** through `evaluation_data`. SuperRed drives the optimizer with a
`Goal`, so:

- `Goal.description` is folded into `evaluation_data` as the **lead claim**
  (`include_goal_as_claim=True`, the default). The synthetic `example_claims`
  follow as style/format demonstrations.
- With `include_goal_as_claim=False`, only `example_claims` are used — upstream's
  dataset-driven behaviour, independent of the specific Goal (useful for
  reproducing a fixed example set).

## Example claims: authored synthetic default, never copied payloads

Upstream ships no example-claims dataset (the vendored files are prompt
templates only); the caller supplies `evaluation_data`. This port defaults
`example_claims` to `claims.DEFAULT_EXAMPLE_CLAIMS`, a small set of **benign,
entirely synthetic** civic-trivia claims authored for this package. They only
demonstrate the ClaimsReview-style few-shot shape; they are not harmful
misinformation and are not copied from any upstream file. Supply your own
`example_claims` to match a specific study.

## Framing folded into one user turn (no system-prompt write)

Upstream `_setup_async` sets the framing template (`anecdoctor_use_fewshot` /
`anecdoctor_use_knowledge_graph`, formatted with `language`/`type`) as the
target's **system prompt**, then sends the few-shot examples or the knowledge
graph as a **separate user message**. A superred optimizer injects at
controllables and, by default, must not assume a writable `system_prompt`
surface exists (and should not clobber a Task-configured system prompt). This
port therefore **folds the framing and the body into a single user-turn
payload** and never injects a system prompt. This preserves the technique's
content (framing + examples/KG) while fitting the single free-text injection
surface, consistent with the other single-turn optimizers.

## Knowledge-graph build timing and fallback

- The KG is built **once at `initialize()`** (the only model call), not per run,
  because the framed request is deterministic given the built graph. `RunStart`
  re-arms only the per-run `_injected` flag, so a re-run re-emits the same
  request.
- The KG build is routed through `self.llm` using the vendored
  `anecdoctor_build_knowledge_graph` prompt (parameterised by `language`) as the
  system prompt and the few-shot claim block as the user message — exactly
  upstream's `_extract_knowledge_graph_async` inputs.
- **Fallback:** upstream raises `RuntimeError` if the processing model returns
  no KG. This port instead degrades to **few-shot framing** (which needs no LLM)
  when no non-empty graph can be obtained within `max_kg_retries`, so the
  technique still emits a faithful request. A genuine budget exhaustion (cost
  already spent) is re-raised so a spent run is never reported as a quietly
  finished attack; the zero-cost noop client the controller hands non-LLM
  optimizers degrades quietly (falls back) instead.

## Single-turn, single surface

Anecdoctor produces one generation, so the optimizer injects once per run and
reports `done=True` at `RunEnd`. It injects into the first eligible free-text
surface and never the reserved `system_prompt` (name/value-type backstop in
`surfaces.py`); no LLM surface classifier is run.

## Converters, normalizer, memory labels not ported

Upstream threads PyRIT `StrategyConverterConfig` converters, a
`PromptNormalizer`, and `memory_labels` through the send path. Those are PyRIT
infrastructure with no superred analogue on the injection surface; the optimizer
emits the framed request directly on the controllable and lets superred handle
delivery, logging, and any target-side transformation.
