# Assumptions and deviations — aegis

Ported from the NVIDIA Aegis AI Content Safety Dataset 1.0
(`nvidia/Aegis-AI-Content-Safety-Dataset-1.0`, CC-BY-4.0) at pinned revision
`bd96d862068e47630197de64eb91f8d1481ff3e0`, paper arXiv:2404.05993. The
decisions below are the ones a reviewer should check.

## (a) A content-moderation dataset, repurposed as a refusal/compliance claim

Aegis 1.0 is a **content-moderation** dataset: human-labelled human/LLM
interactions annotated for safety, built to train and evaluate *classifiers*
(guardrails). This claim repurposes it as a **red-team / refusal claim**: it
takes the dataset's unsafe **user prompts** as attack inputs, sends each to the
target, and grades whether the target's *response* is unsafe. This is exactly the
framing promptfoo's `AegisGrader` and deepteam's Aegis framework use — the
dataset's unsafe user messages become adversarial prompts, and success is a
safety violation in the response. It is a faithful *use* of the data, but note
the original annotations label the dataset's own dialogs, not the target's
replies; the target's replies are graded by our judge (below), not by the
dataset.

## (b) The judge is generic (`generic-judge`), not NVIDIA's Aegis-Guard

The shipped `AegisSafetyJudge` is a generic LLM-as-judge (default
`openai/gpt-4o`, matching deepteam's evaluation model) applying an Aegis-taxonomy
safety rubric and returning `{"pass": true|false}`. **It is not NVIDIA's native
scorer.** The Aegis paper's native scorer is **Aegis-Guard** — LlamaGuard-based
defensive models (`nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0` /
`...-Permissive-1.0`) fine-tuned on this taxonomy. We do **not** ship it because
it needs heavy, separately-gated model weights (a ~7B LlamaGuard checkpoint,
gated by Meta's Llama license) that a lightweight, import-on-3.13 claim must not
pull in.

**Plugging in Aegis-Guard.** The claim takes any object satisfying the `Judge`
protocol (`async score(prompt, response) -> (unsafe_score, rationale)`, where
`unsafe_score` is 1.0 for unsafe / 0.0 for safe). To use the native scorer, wrap
Aegis-Guard behind that protocol — run the LlamaGuard-Defensive checkpoint (via
`transformers`, vLLM, or an inference endpoint) over the target's response with
NVIDIA's prompt template, map its `unsafe`/`safe` head to `1.0`/`0.0`, and pass
it as `aegis_claim(judge=YourAegisGuardJudge(...))`. Nothing else changes.

## The rubric wording is ours

promptfoo and deepteam were read as references only; none of their code is
vendored. The safety rubric in `judge.py` (`AEGIS_SAFETY_RUBRIC`) is our own
wording that enumerates the 13 Aegis harm categories and fixes the `{"pass": …}`
output contract. The `pass` semantics (pass = safe) match promptfoo's grader so
the verdict is interpreted the same way.

## Test split, not train

deepteam and promptfoo both draw from the Aegis `train` split. This claim uses
the **`test` split** (per the build spec): 1199 rows, of which 359 are
`user_message`. Numbers here are therefore not comparable row-for-row to
deepteam/promptfoo runs, which sample the train split.

## "Needs Caution" is included by default

Aegis has 13 harm categories **plus** a gray-zone disposition `"Needs Caution"`
and the negative label `"Safe"`. The default filter is deepteam's
(`labels_0 != "Safe"`), which counts a bare `"Needs Caution"` prompt as unsafe →
**236** user-message tasks. `include_needs_caution=False` restricts to the 13
harm categories → **180**. "Needs Caution" is genuinely ambiguous (Aegis maps it
to safe or unsafe by deployment risk tolerance); the default follows the named
reference (deepteam) and the stricter reading is one kwarg away. Both counts are
pinned in `tests/`.

## Only `user_message` prompts

The test split has four `text_type`s: `user_message` (359), `combined` (332),
`llm_response` (330), `multi_turn` (178). A red-team claim sends a *user* prompt
to the target, so only `user_message` rows are used. `combined` (system+user),
`llm_response` (a Mistral reply, not an attack input) and `multi_turn` dialogs
are out of scope for this single-turn claim.

## No temperature

No temperature is ever sent to the judge (superred house rule; reasoning models
reject the parameter). `tests/test_no_temperature.py` enforces this structurally
(AST).

## Abstain = raise

A judge that cannot produce a parseable verdict raises `JudgeUnavailableError` /
`JudgeUnparseableError` (shared `judge_failure.py`, byte-identical across the
chatbot claims). The exception propagates out of `evaluate`; the controller
records `stop_reason="error"` rather than a fabricated "safe" verdict. An empty
target response short-circuits to safe (0.0) *without* consulting the judge, and
`judge_ran` records that it was not asked — an empty answer cannot contain unsafe
content.

## Output discipline

`evaluate`, the judge, and the loader never emit the prompt or the response text
— not in the returned rationale, not in logs. Only structural fields (row id,
risk category, judge cost, `safe`/`unsafe` verdict) are surfaced. The response is
embedded only in the message *sent* to the judge model, which is unavoidable for
grading it.

## Vendoring

The parquet is vendored byte-identically (the provenance anchor); the CSV is a
deterministic, lossless column-for-column rendering loaded with the stdlib `csv`
module (`text` cells with embedded newlines/commas/quotes round-trip via
`newline=""` + `QUOTE_MINIMAL`). Both are sha256-pinned;
`scripts/sync_upstream.py --check` byte-compares the parquet against HuggingFace
and re-derives the CSV to confirm it is exactly the parquet's rendering.

## Not reproduced

The dataset's per-annotator label columns (`labels_1..4`, `num_annotations`) are
vendored in the CSV but not used by the claim, which keys off the aggregated
`labels_0`. Aegis-Guard training/inference, the ensemble-of-experts routing, and
the "no-dialog-level vs prompt-level" annotation caveat from the dataset card are
out of scope for a prompt-level red-team claim.
