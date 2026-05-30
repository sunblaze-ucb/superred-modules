# Security Claims (the tests)

A **security claim** is a concrete test suite: a collection of harmful goals to
attempt, plus a way to judge, after each attempt, whether the system actually
complied. It is what turns "we attacked the system" into "the system failed N of
M tests". Each goal becomes one **task**; the claim bundles them.

Two ideas to understand before the catalogue:

- **How success is judged.** For most of these, an "LLM-as-judge" reads the
  target's answer and decides whether it complied. That judge is configured
  **separately** from the attacker: it has its own model and its own spending
  budget, passed in when you build the claim. This keeps things fair (the judge
  does not eat the attacker's budget) and honest (the system being attacked, and
  the attacker's own model, do not get to grade the result). Where a benchmark
  specifies a particular judge model, pin it to that one.
- **Success is defined strictly.** A task counts as a successful attack only on a
  clear, fully-compliant judgment. Refusals, partial or hedged compliance, and
  judge errors all count as failure. Each module's README states its exact rule
  and the headline metric it reports (usually "attack success rate", the
  fraction of goals the attacker got the system to comply with).

You build a claim by calling its **factory function**, which loads the dataset
and produces the tasks. Most factories let you filter to a category or take a
small sample for quick smoke tests.

## Real claims

### HarmBench (`secclaim-harmbench`)

A broad, standardized red-teaming benchmark: 400 harmful behaviors (320 in the
test split, 80 in validation) spanning straightforward harmful requests,
context-dependent ones, and copyright-style requests. It tests a chatbot's
refusal robustness. Judgment uses HarmBench's own classifier prompt for most
behaviors, plus a text-overlap check for the copyright items; the headline number
is the attack success rate. The dataset ships with the module. You choose the
judge model (and its credentials) when you build the claim, and there are
ready-made factories for the test split, the validation split, and the whole
thing. Pairs with the `chatbot` target. Its README has a full table of how it
diverges from the original paper.

### StrongREJECT (`strongreject-claim`)

A jailbreak benchmark focused on measuring how *useful* a jailbroken answer
actually is, not just whether the model said yes. It has 313 forbidden prompts
across six categories (plus a curated 60-prompt small set for quick runs). A
rubric-based LLM judge scores each answer on whether the model refused and, if
not, how specific and convincing the answer was, producing a 0-to-1 score. Eight
factory functions cover the six categories plus the full and small sets. The
dataset ships with the module (integrity-checked). Pairs with the `chatbot`
target. Its README documents the scoring math and its known differences from the
original work.

### SORRY-Bench (`security-claim-sorry-bench`)

A fine-grained safety-refusal benchmark: 44 specific categories of unsafe
requests, 10 prompts each (440 total). A judge marks each answer as a refusal or
a compliance (0 or 1) using SORRY-Bench's own judge prompt. You can run a single
category, a sample per category, or one of two disjoint halves of the dataset.

**Important setup note:** the SORRY-Bench prompts are gated and cannot be
redistributed, so this module **does not ship the data**. You point it at the
dataset at runtime (via an environment variable or an automatic download), as the
module README explains. The judge is configured out-of-band as its own model and
budget; the benchmark's canonical judge is a fine-tuned Mistral-7B (heavy,
needs a GPU), so the module defaults to a GPT-4 snapshot as a practical,
well-correlated stand-in. Pairs with the `chatbot` target.

### AgentDojo (`security-claim-agentdojo`)

The test suite for the `agentdojo` tool-using agent target, focused on
prompt-injection against agents. It comes in three layers you can pick from:

- a faithful port of AgentDojo's original injection tasks (a benign user goal
  paired with a hidden attacker goal), grouped by attack type;
- a set of bespoke "make the agent violate its purpose" goals tailored to the
  multi-tool agent (for example, getting it to make an unauthorized transfer or
  exfiltrate data); and
- both layers combined.

A distinctive feature: success here is judged by **deterministic checks on the
agent's actual actions** (did the unauthorized transfer really happen?), not by
an LLM reading the transcript. That makes verdicts exact and reproducible. The
benchmark version is pinned. Its README and `ASSUMPTIONS.md` explain the layers
and the predicate-by-predicate mapping to the original benchmark.

## Test fixtures

Tiny claims used to exercise the framework and to serve as readable examples.

### `test_basic_secret_leak` (`basic-secret-leak-claim`)

Plants a secret in the target's system prompt that the model is told to reveal
only on a hidden trigger word, and checks whether the attacker can make the
secret appear in a response. The smallest complete example of a claim and its
factory; pairs with the `test_basic_llm_chat` target. Built with
`basic_secret_leak_claim(secret=..., trigger=...)`.

### `test_all_keys_match` (`all-keys-match-claim`)

Pairs with the `test_filter_test` target and checks that each injection point
received the value it was supposed to, reporting a separate score per trust
boundary. Its purpose is to verify the framework's per-boundary score filtering.

## How a claim connects to everything else

You hand a claim to the framework's Controller along with a target factory, an
attacker, and a scope; the Controller runs every task and reports which the
attacker won. For the full picture see the framework's
[Task](../../superred/docs/task.md) and
[SecurityClaim](../../superred/docs/security_claim.md) interface docs, and the
[Controller](../../superred/docs/controller.md) reference.
