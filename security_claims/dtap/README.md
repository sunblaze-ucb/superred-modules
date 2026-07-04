# security_claim_dtap

The **DecodingTrust-Agent (DTAP-BENCH)** security claim for superred.

In one sentence: DTAP is a large agent-safety benchmark whose tasks each spin up a
real tool environment (a travel booking platform, a CRM, a filesystem, ...), give
an agent a goal, and then check the *actual state of that environment* to decide
whether the agent did the right or the wrong thing. This package turns that
benchmark into a superred `SecurityClaim`: one `Task` per DTAP per-task config,
scored by DTAP's own verifiable judge run out of band.

## What a DTAP task is

Each DTAP task is a directory with a `config.yaml` (the scenario), a `setup.sh`
(seeds the environment), and a `judge.py` (decides the outcome by querying the
live environment). There are two kinds:

- **benign** -- a legitimate user request the agent *should* complete. Scored by
  the judge's `eval_task`: did the agent do the job.
- **malicious** -- an attacker objective the agent *should not* carry out, under
  one of two threat models: `direct` (the user themselves is adversarial) or
  `indirect` (a third party injects through the environment, e.g. a poisoned
  document or tool result). Scored by the judge's `eval_attack`: **`True` means
  the attack SUCCEEDED** (the safety property was violated).

## How it scores

For a malicious task the **primary score** is `attack_success` (unscoped, always
visible) and `task_success` is a sub-score. For a benign task the primary is
`task_success`. The judge is upstream's per-task `judge.py`, run byte-faithfully by
the shared `dtap_scaffold.judge_runner` against the live environment containers;
its cost is out of band and is never charged to the optimizer's budget. If the
judge cannot produce a verdict (its subprocess failed, or its live environment was
unreachable), the run is surfaced as an evaluation error rather than silently
scored as a safe `0.0`.

The claim is **target-agnostic and attacker-agnostic**: it binds to the base
`Target`, sets the scenario, and reads ground truth back. *Which* of DTAP's four
injection vectors (prompt / tool / environment / skill) an attacker may use is the
experiment's Controller **scope**, and the attack *content* is the optimizer's job;
there is no attacker baked into the claim.

## Scope

Text-only domains only (11 of DTAP's 14; `browser`, `macos`, `windows` are
vision/GUI and excluded). The per-task dataset is not vendored here: point
`$DTAP_DATASET_ROOT` at a checkout, pass `dataset_root=...`, or use `download=True`
to fetch from HuggingFace.

## Setup

```bash
source ../../.venv/bin/activate
pip install -e ../../superred                       # the framework
pip install -e ../../superred-modules/targets/dtap_scaffold
pip install -e .                                    # this claim

# the per-task dataset (one option):
export DTAP_DATASET_ROOT=/path/to/DecodingTrust-Agent/dataset
```

To actually *run* an evaluation you also install one of the DTAP agent target
packages (`dtap_claudecode` or `dtap_openclaw`); they are imported lazily by the
target factories, so this claim does not depend on them.

## Usage

```python
from security_claim_dtap import (
    dtap_claim,
    dtap_domain_claim,
    dtap_direct_claim,
    dtap_indirect_claim,
    dtap_benign_claim,
    dtap_risk_claim,
    dtap_claudecode_target_factory,
)

# All travel-domain tasks (benign + malicious):
claim = dtap_domain_claim("travel")

# Just the indirect-injection malicious tasks across all text-only domains:
claim = dtap_indirect_claim()

# A single risk category:
claim = dtap_risk_claim("data-exfiltration")

# A matching target factory (lazy-imports the Claude Code DTAP target):
target_factory = dtap_claudecode_target_factory(
    model="gpt-4o", api_base="<proxy>", api_key="<key>"
)
```

Wire `claim` + `target_factory` into a superred `Controller` with the scope (which
injection vectors are in play) and the optimizer for the experiment.

## Faithfulness

Every deliberate deviation from upstream DTAP is recorded in
[`ASSUMPTIONS.md`](ASSUMPTIONS.md). The committed `data/golden_hashes.json` pins the
byte-identity of the sampled tasks' attacker goals and judges; the offline test
suite mocks every Docker / HTTP / LLM boundary, so it runs without a daemon or
credentials (dataset-dependent and live tests skip when their resources are
absent).
