# security-claim-agentharm

A faithful superred port of the [AgentHarm](https://arxiv.org/abs/2410.09024)
benchmark (Andriushchenko et al.), harmful `test_public` split: **176 behaviors**
(44 base x 4 augmentations) across **8 harm categories**. It runs against the
general `inspect-agent-target` and reuses upstream `inspect_evals.agentharm`
tools, grading functions, and judges **verbatim**, so a run against a
(target, attacker) pair reproduces AgentHarm's own scores.

Threat model: the **user is the adversary** (refusal robustness + harm
completion), which is orthogonal to injection benchmarks like AgentDojo. See
`../../../benchmark-overlap-analysis/` for why this is the additive benchmark.

## Mapping to superred

- **Target** = the agent under test (general `InspectAgentTarget` wired with
  AgentHarm's tool resolver).
- **Optimizer = the attacker.** A passthrough optimizer is AgentHarm's
  *direct-request* baseline and reproduces the headline numbers; a jailbreak
  optimizer injects into the `user_prompt` controllable.
- **Score**: `primary_score = harm_score` (higher = attacker won, domain `user`);
  sub-scores carry `refusal`, `full_harm`, each grading sub-check, the category,
  and grading/judge error flags. `success = (not refusal) and harm_score == 1.0`
  (a strict per-task gate, not an AgentHarm headline metric).

## Usage

```python
from superred.core.controller import Controller
from inspect_agent_target import USER_TAG, SYSTEM_TAG, TOOL_OUTPUT_TAG
from security_claim_agentharm import agentharm_total_claim, agentharm_target_factory, aggregate

MODEL = "openai/gpt-4o-2024-08-06"
claim = agentharm_total_claim(target_model=MODEL, api_base=PROXY, api_key=KEY)   # 176
controller = Controller(
    optimizer_factory=MyAttackerOrPassthrough,
    target_factory=agentharm_target_factory(target_model=MODEL, api_base=PROXY, api_key=KEY),
    security_claim=claim,
    # the three root tags = the full attacker surface; narrow per experiment.
    scope=frozenset({USER_TAG, SYSTEM_TAG, TOOL_OUTPUT_TAG}),
)
result = await controller.run()
metrics = aggregate([tr.best_evaluation for tr in result.task_results])
# -> avg_score, avg_full_score, avg_refusals, avg_score_non_refusals, per-category
```

The `target_model` passed to the claim and to the target factory MUST match.

## Subclaims (8 categories + total)

`agentharm_total_claim` (176), `agentharm_category_claim(label)`, and 8 named
factories: `agentharm_copyright_claim`, `agentharm_cybercrime_claim`,
`agentharm_disinformation_claim`, `agentharm_drugs_claim`, `agentharm_fraud_claim`,
`agentharm_harassment_claim`, `agentharm_hate_claim`, `agentharm_sexual_claim`.
(The paper has 11 categories; the released public split covers 8.)

## Judges

Refusal + semantic judges are upstream's, pinned to `openai/gpt-4o-2024-08-06`
(verified available on the project proxy, so no substitution). Judge calls are
out-of-band (not counted in Controller `llm_usage`).

## Install / test

```bash
pip install -e targets/inspect_agent
pip install -e security_claims/agentharm
pytest -m "not (faithfulness or smoke)" security_claims/agentharm/tests   # offline
LITELLM_API_KEY=... LITELLM_API_BASE=... pytest -m faithfulness security_claims/agentharm/tests  # OUR==upstream parity
LITELLM_API_KEY=... LITELLM_API_BASE=... pytest -m smoke security_claims/agentharm/tests          # end-to-end vs the target
LITELLM_API_KEY=... LITELLM_API_BASE=... python security_claims/agentharm/tests/smoke/run.py       # printed demo
```

The dataset is downloaded from the gated HuggingFace repo
`ai-safety-institute/AgentHarm` (pinned revision) on first use. See `ASSUMPTIONS.md`.
