# Assumptions and deviations

## Upstream

| | |
| --- | --- |
| Project | [DeepTeam](https://github.com/confident-ai/deepteam) |
| Code | `deepteam/attacks/multi_turn/bad_likert_judge/{bad_likert_judge,template,schema}.py`, `multi_turn/base_template.py` |
| Commit | `dc148aad62f71330cfec7121d6afb4c620dfa683` |
| Licence | Apache-2.0 |
| Technique | Palo Alto Networks Unit 42, "Bad Likert Judge" |

## Copied byte-for-byte

- `_vendor/deepteam_blj/template.py` — all seven prompt builders and the 58
  category guideline sets.
- `_vendor/deepteam_blj/base_template.py` — the `non_refusal` classifier prompt.
- `SUPPORTED_CATEGORIES` — upstream's `get_supported_categories()` list.
- The JSON field names in `parsing.py` — upstream's `schema.py` models.

The vendored template keeps upstream's absolute
`from deepteam.attacks.multi_turn.base_template import ...`. Rather than edit
that line (which would fork the file and reduce `sync_upstream.py` to a fuzzy
check), `_vendor/loader.py` registers the vendored base under exactly that
module name before executing the template. Whether to register at all is
decided by `importlib.util.find_spec`, i.e. by whether a real `deepteam` is
**installed** — not by whether it happens to be in `sys.modules`, which only
reflects what has been imported so far and would let an installed-but-unimported
package be stubbed over. When a real `deepteam` is present nothing is
registered and it is used as-is. `scripts/sync_upstream.py --check` is
therefore a plain byte comparison.

## Upstream coverage

`bad_likert_judge.py` defines one attack. Its full turn chain is ported:

| Upstream step | Ported |
| --- | --- |
| `likert_generate_examples` | yes — step 1 of every turn |
| `likert_refine_score_3` | yes — `enable_refinement`, keeping the original example if refinement fails, as upstream does |
| `likert_generate_attack_from_example` | yes — its output is the turn sent to the target |
| `non_refusal` | yes — an explicit refusal spends a backtrack and sends nothing |
| `max_backtracks` | yes — exhausting them ends the attack |
| `num_turns` | yes |
| `likert_setup_prompt`, `likert_judge` | vendored (they ship in the byte-identical template) but **not called** — upstream imports them and never calls them either |

Not ported, with reasons:

- **`turn_level_attacks` / `enhance_attack`** — composes *other* DeepTeam
  attacks onto a turn, which needs DeepTeam's attack registry. Composition of
  superred optimizers is a framework concern, not this module's.
- **`BehaviorShiftDetector` / `metric_check`** — DeepTeam's own scoring. In
  superred the `SecurityClaim` judges, so no port ships a detector.
- **`create_progress` / `add_pbar`** — terminal UI, not attack functionality.
- **`initialize_model` / `simulator_model`** — DeepTeam's model plumbing; the
  framework injects `self.llm`, which is the simulator here.

## Deviations

### 1. The Goal fills upstream's vulnerability slot

Upstream builds `vulnerability_data` as
`f"Vulnerability: {vulnerability} | Type: {vulnerability_type}"` from its own
vulnerability taxonomy. superred's objective is the `Goal`, so the same format
string is built from `goal.description` and the configured category. Every
prompt therefore still carries the objective, exactly where upstream put it.

### 2. One turn per run

Upstream loops internally over `num_turns`, calling a `model_callback`. superred
drives turns through the event loop, so each run sends one turn and
`RunEndResponse.done` reports completion. `current_attack` carries across runs,
so turn *n+1* escalates from the turn actually sent — matching upstream, where
the chain advances from `current_attack` rather than the target's reply.

### 3. A genuine budget exhaustion propagates

Every simulator failure becomes a backtrack, **except** a `BudgetExhaustedError`
with `usage.cost > 0`. Folding that into the backtrack path would spend every
`max_backtracks` on an attacker that simply ran out of money and then report
`done` — indistinguishable from an attack the target defended, i.e. a false
zero. The budget-less noop client the controller hands non-LLM optimizers
raises the same error with nothing spent, so only a spent budget propagates,
matching `surface_llm._is_genuine_exhaustion` and the convention in
`crescendo`, `attack_anything`, `autodan_turbo` and others.

### 4. Parsing degrades instead of raising

Upstream validates each reply with pydantic and treats a failure as a
backtrack. This module parses the same fields without a pydantic dependency and
returns `None`, which the caller turns into the same backtrack. An unparseable
*refusal check* is treated as "not a refusal", so a garbled reply never
silently costs a turn — upstream only backtracks on an explicit `refusal: true`.

### 5. Injection-surface selection (superred adaptation)

Upstream targets a chat callback; superred targets expose arbitrary named
surfaces. Following the house pattern (`muzzle`, `chord_xthp`, `eia_agent`,
`gepa_agentic`, `poisonedrag`, `agentvigil_websentinel`), this module ships a
byte-identical copy of the shared `surface_llm.py` and calls
`classify_controllables` once at `initialize`, so the escalation runs on the
surface that is actually the user's prompt. It falls back to the name/value-type
backstop when classification is unavailable.

Note: the shared `surface_llm.py` pins `temperature=0.0`, which the repo's
`test_no_temperature.py` guard forbids. The six merged modules shipping the
classifier resolve this by omitting the guard; this module keeps the guard and
skips only that one pinned shared file, so it still covers all first-party code.

## Known limitations

- Each turn costs three attacker-LLM calls (four with refinement), so a
  `num_turns=5` run is 15–20 calls before the target is touched.
- The category is chosen by the caller. Upstream picks it from the vulnerability
  under test; here nothing infers it from the goal, so a mismatched category
  gives the model guidelines unrelated to the objective.
