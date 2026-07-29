# superred-optimizer-libertas

A byte-faithful, model-aware replay optimizer for Pliny's
[L1B3RT4S](https://github.com/elder-plinius/L1B3RT4S) jailbreak prompt corpus.

L1B3RT4S is a living collection of model-specific jailbreak prompts and
auxiliary material rather than an executable attack algorithm.
`LibertasOptimizer` turns the compatible portion of that corpus into a superred
attack: it selects the target's vendor family from explicit model metadata,
asks `self.llm` to rank trusted template metadata for the target model and
goal, replaces the chosen prompt's explicit upstream goal slot with
`Goal.description`, and tries one template per run. The SecurityClaim alone
decides success.

## Upstream parity

Parity is pinned to L1B3RT4S commit
`64960b783249d36f76a48a33103cc4b168332b9b`.

- Bundled upstream files are copied as bytes and checked against committed
  SHA-256 hashes and byte sizes.
- Prompt bodies are slices of those exact files. The parser never strips
  whitespace, converts newlines, uses replacement decoding, or applies Unicode
  normalization.
- Rendering changes only an explicit generic upstream goal marker, such as
  `{Z}` or `<user_query>`. Untemplated sections are excluded by default.
- Prompt headings remain provenance metadata and are not sent to the target,
  matching the upstream copy/paste convention.
- Upstream filenames—including names beginning with `!`, `#`, `*`, and `-`—are
  preserved exactly inside the wheel. Corpus paths are handled only through
  Python resource APIs, never shell interpolation.

Run `verify_bundled_corpus()` at any time to check the installed snapshot.
See `ASSUMPTIONS.md` for the complete file-by-file boundary and deviations.

## Install

```bash
pip install superred-optimizer-libertas
```

This package is AGPL-3.0-only because it redistributes and adapts the upstream
AGPL-3.0 prompt corpus.

## Usage

```python
from libertas_optimizer import LibertasOptimizer

# Detect the provider from the target's model/model_identity observable,
# then use self.llm to rank compatible prompts.
optimizer = LibertasOptimizer()

# Supply exact target metadata when the target does not expose it.
optimizer = LibertasOptimizer(
    model_identity="openai/gpt-5.2",
    max_attempts=4,
)

# Reproduce a source-order baseline without a helper-LLM call.
optimizer = LibertasOptimizer(selection_strategy="deterministic")
```

By default the optimizer:

- uses only prompt bodies with an explicit upstream goal marker;
- uses only user-message delivery;
- makes at most one helper-LLM call to rank candidate metadata;
- never sends upstream prompt bodies to the helper LLM;
- validates every returned ID and falls back atomically to source order on
  malformed output, unknown IDs, duplicates, budget exhaustion, or any other
  helper failure;
- stops early only when the SecurityClaim reports success.

The helper receives the target model identity, provider, task goal, and a
catalog containing only template IDs, filenames, headings, delivery surfaces,
and goal-marker spellings. A partial valid ranking is accepted; unmentioned
candidates retain source order after the preferred entries. `max_attempts` is
applied after ranking, so the helper can select a later upstream prompt for a
one-attempt run. Inspect `selection_method` to distinguish `llm`,
`deterministic`, and `deterministic-fallback` schedules.

Privileged custom-instruction/system-prompt entries are opt-in:

```python
optimizer = LibertasOptimizer(
    provider="openai",
    include_system_templates=True,
)
```

They are scheduled only when both a `system_prompt` and a user-facing
controllable are in scope. The exact upstream template is installed on the
system surface and the task goal is sent as the subsequent user query, matching
the upstream custom-instruction workflow.

Untemplated sections require an explicit adaptation opt-in:

```python
optimizer = LibertasOptimizer(include_untemplated=True)
```

In that mode the goal is appended after the exact upstream text. Results from
this mode should be reported separately from the strict parity mode.

## Corpus API

```python
from libertas_optimizer import (
    load_prompt_templates,
    render_prompt,
    verify_bundled_corpus,
)

templates = load_prompt_templates(provider="anthropic")
prompt = render_prompt(templates[0], "the task goal")
assert verify_bundled_corpus() == []
```

`PromptTemplate` exposes its upstream file, heading, section index, delivery
surface, exact raw body, raw-body SHA-256, and recognized markers.

## Updating upstream

Upstream changes are never consumed dynamically. To update:

1. inspect and check out the intended L1B3RT4S commit;
2. deliberately change the pinned commit in `corpus.py` and
   `scripts/sync_upstream.py`;
3. run `python scripts/sync_upstream.py /path/to/L1B3RT4S`;
4. review every corpus and manifest change;
5. update `ASSUMPTIONS.md` and run the full test suite.

Dynamic downloads would make experiment results drift and could silently alter
special Unicode sequences, so they are intentionally unsupported.
