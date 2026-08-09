# superred-modules

This repository holds the **plug-in components** for the
[superred](../superred/) red-teaming framework. The framework itself is just
machinery; the interesting pieces, the attacks, the systems being attacked, and
the definitions of "what counts as a successful attack", live here.

If you have never seen superred before, the one-paragraph version is: you take a
**target** (an AI system you want to stress-test), point an **optimizer** (an
automated attacker) at it, and check the outcome against a **security claim** (a
checklist of things the system should not be tricked into doing). This repo
provides ready-made versions of all three.

## The three kinds of module

```
optimizers/        the attackers      (how to try to break a system)
targets/           the systems        (what is being attacked)
security_claims/   the tests          (what counts as broken, and how it's judged)
```

- An **optimizer** is the attacker. It decides what text to feed into the
  system's input points and reads back what the system produced, trying to make
  it misbehave. Most of these are faithful re-implementations of published
  jailbreak techniques. See [optimizers/README.md](optimizers/README.md).
- A **target** wraps an AI system so the framework can drive it: a plain
  chatbot, a tool-using agent, and so on. A target also declares its **trust
  boundaries** (which parts an attacker might control), its **injection points**
  (where attacker text can go), and the **facts** an attacker is allowed to
  read. See [targets/README.md](targets/README.md).
- A **security claim** is a concrete test suite: a set of harmful goals plus a
  way to judge, after each attempt, whether the system actually complied. Most
  wrap a well-known safety benchmark. See
  [security_claims/README.md](security_claims/README.md).

You mix and match: any optimizer can attack any compatible target, measured by
any compatible claim. The combinations are wired together by short scripts in
the separate `superred-experiments/` repository.

## What is here

Each module is its own independently installable Python package. Names that
begin with `test_` are tiny built-in fixtures used to exercise the framework;
everything else is a real attack, system, or benchmark.

**Optimizers (attackers).** Real attacks: AutoDAN-Turbo, Bijection Learning,
CodeChameleon, Crescendo, DRA, FITD (Foot-in-the-Door), FlipAttack, GEPA, GOAT,
GPTFuzzer, Libertas, Many-Shot, PAIR, TAP. Fixtures: `demo_prompt_list`,
`test_hint_following`, `test_llm_prompt_generator`.

**Targets (systems under test).** Real: `chatbot` (any LLM as a chatbot),
`agentdojo` (a tool-using agent), `minimal_llm_chat` (the minimal
single-turn chat, and the reference example to read first). Fixtures:
`test_filter_test`.

**Security claims (tests).** Real: HarmBench, StrongREJECT, SORRY-Bench,
AgentDojo. Fixtures: `demo_secret_leak`, `test_all_keys_match`.

The per-category READMEs describe every module in plain language, including what
it controls and observes and how to set it up.

## Setup

Requires Python 3.11 to 3.13 and the shared virtual environment at the
repository root.

```bash
source ../.venv/bin/activate          # activate the shared venv
pip install -e ../superred            # the framework first

# then install whichever modules you need (each is editable, -e)
pip install -e optimizers/crescendo
pip install -e targets/chatbot
pip install -e security_claims/harmbench
```

Modules with extra dependencies (an LLM client, a benchmark dataset loader)
declare them in their own `pyproject.toml`, so pip pulls them in automatically.
A few benchmark claims need a dataset that cannot be redistributed (SORRY-Bench
is the main one); those READMEs explain how to point the module at the data.

## Usage

After installing, import each module by its package name (dashes in the pip
name become underscores in the import name):

```python
from crescendo_optimizer import CrescendoOptimizer
from chatbot_target import ChatbotTarget, USER_TAG
from secclaim_harmbench import harmbench_claim
```

Targets export the **security-domain tag constants** you need to describe an
attacker's level of access (for example `USER_TAG`); the targets README lists
them per target.

## Adding a new module

1. Create a directory under the right category. Use a descriptive name for a
   real module (e.g. `optimizers/my_attack/`); the `test_` prefix is reserved
   for framework test fixtures.
2. Add a `pyproject.toml` that lists `superred` as a dependency.
3. Add `src/<package_name>/` with an `__init__.py` that exports your public
   classes/functions, plus the implementation.
4. Install it editable: `pip install -e optimizers/my_attack`.
5. For a faithful port of a published attack or benchmark, also add an
   `ASSUMPTIONS.md` recording the source paper, the reference implementation,
   and every deliberate deviation. This is the established convention here and
   makes your faithfulness claims auditable.
6. If the module ports or integrates third-party work, open its `README.md` with
   a community-port notice directly under the first paragraph, so nobody mistakes
   it for the upstream authors' own release. The module README is also the PyPI
   page, so this is the first thing a reader sees. Copy the wording from any
   existing port, e.g. `optimizers/tap/README.md`.

For how these modules plug into the framework (the Target/Optimizer/Task/
SecurityClaim interfaces and the Controller), see the framework's
[architecture overview](../superred/docs/architecture.md).
