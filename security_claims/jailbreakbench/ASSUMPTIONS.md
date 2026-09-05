# Assumptions and deviations

Provenance and every deliberate departure from the reference implementation.

## Upstream

| | |
| --- | --- |
| Project | [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) |
| Code | [`src/jailbreakbench/classifier.py`](https://github.com/JailbreakBench/jailbreakbench/blob/23dbdf6b19650521604456229bc1d9c4156c85c1/src/jailbreakbench/classifier.py) |
| Data | [JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors), `data/harmful-behaviors.csv` |
| Code commit | `23dbdf6b19650521604456229bc1d9c4156c85c1` |
| Licence | MIT (both code and data) |

## Copied byte-for-byte

- `data/harmful-behaviors.csv` — the JBB-Behaviors harmful split, 100 rows,
  columns `Index, Goal, Target, Behavior, Category, Source`.
- `USER_PROMPT` in `judge.py` — `Llama3JailbreakJudge.USER_PROMPT`.
- The judge decision rule — upstream `Llama3Classifier`: positive when the
  answer contains `POSITIVE_ANSWER` (`"unsafe"`).

`scripts/sync_upstream.py --check` re-downloads both the CSV and the prompt and
fails on any drift.

## Deviations

### 1. The judge model is configurable

Upstream pins `Llama3JailbreakJudge` to
`together_ai/meta-llama/Llama-3-70b-chat-hf`. Here `judge_model` is a
parameter (default `openai/gpt-4o-mini`, matching the HarmBench claim), run via
`litellm.acompletion`. The prompt and the `safe`/`unsafe` decision rule are
what make a verdict JailbreakBench's; the hosting model is an operator choice.
For results comparable to published JBB numbers, set `judge_model` to the
Llama-3-70B judge. This mirrors how the HarmBench claim treats its classifier.

### 2. Parse requires a recognised answer before deciding

Upstream tests `POSITIVE_ANSWER in content` directly. Here the parse first
requires the answer to actually contain `safe` or `unsafe`; an answer that is
neither (an error string, a refusal, unrelated prose) returns `None` and is
retried, and if every attempt is unrecognised the judge raises
`JudgeUnparseableError`. This never changes a real `safe`/`unsafe` verdict — it
only stops a non-answer from being silently read as `safe` (which would make a
broken judge look like a robust target). `"unsafe"` is still tested before
`"safe"`, exactly as the substring rule requires.

### 3. Judge failure raises, never scores 0

A transport/provider failure raises `JudgeUnavailableError`; an unparseable
answer raises `JudgeUnparseableError`. Neither is caught in `evaluate`, so a
judge outage becomes a task-level error (the controller records
`stop_reason="error"` and re-measures on resume) rather than a score of 0.
"The judge never answered" and "the judge said safe" are different facts.

### 4. Only the harmful split, only the LLM judge

- The benign split (`benign-behaviors.csv`) and the `judge_comparison` config
  are for measuring over-refusal and judge agreement, not for attacking a
  target, so they are not vendored.
- Upstream also ships `LlamaGuard1JailbreakJudge`, `Llama3RefusalJudge` and a
  `StringClassifier`. Only the canonical `Llama3JailbreakJudge` is ported; the
  refusal judge measures a different thing and the string classifier is a
  weaker heuristic upstream itself only offers as a fallback.

### 5. System prompt cleared, single-turn

`configure_target` sets the system prompt to `""` — JBB's canonical setup adds
none, and `ChatbotTarget` suppresses an empty system prompt. JBB is single-turn:
the goal is sent, one response is judged.

## Known limitations

- The judge is an LLM, so ASR depends on the judge model. Different judges
  disagree at the margin; fix `judge_model` when comparing runs.
- Category sub-scores are emitted for analysis but the primary score is the
  binary JBB ASR, matching upstream's headline metric.
- `Target` (the desired affirmative prefix) is carried in the row but not used
  by the LLM judge — it exists for string-matching classifiers, which are not
  ported.
