# Assumptions and deviations

**Based on Tencent Zhuque Lab AI-Infra-Guard
(https://github.com/Tencent/AI-Infra-Guard)** — required by that project's
NOTICE under Apache-2.0 Section 4(d).

## Upstream

| | |
| --- | --- |
| Project | [Tencent AI-Infra-Guard](https://github.com/Tencent/AI-Infra-Guard) |
| Code | `AIG-PromptSecurity/deepteam/attacks/multi_turn/actor_attack/{actor_attack,template,schema}.py` |
| Commit | `dd6bd54655c9ff5fb7351f4299b56916f09ec6da` |
| Licence | Apache-2.0 (with a mandatory attribution requirement) |
| Technique | Ren et al., ActorAttack |

### Why this source and not the authors' repository

The paper's own repository, [AI45Lab/ActorAttack](https://github.com/AI45Lab/ActorAttack),
ships **no licence file**, so it is all-rights-reserved and nothing may be
copied from it. Tencent's AI-Infra-Guard contains an independent Apache-2.0
implementation of the published method against DeepTeam's attack interface
(`class ActorAttack(BaseAttack)`), and that is what this module ports.
DeepTeam upstream itself has no `actor_attack`.

## Copied byte-for-byte

- `_vendor/aig_actor_attack/template.py` — all four prompt builders
  (`generate_actor_network`, `next_probe_prompt`, `non_refusal`, `judge`).
  It has no imports, so it is loaded directly.
- The JSON field names in `parsing.py` — upstream's `schema.py` models.

`scripts/sync_upstream.py --check` re-downloads the template for a byte
comparison and asserts the schema still defines every field parsed here.

## Upstream coverage

`actor_attack.py` defines one attack; its whole `enhance` loop is ported:

| Upstream step | Ported |
| --- | --- |
| `generate_actor_network` → `actors[0]` | yes — an empty or unparseable network ends the attack, as upstream returns early |
| `opening_question` as turn 1 | yes |
| `non_refusal` classification | yes — a refusal scores 1 and skips the judge, as upstream does |
| `judge` rating | yes — **as the attacker-side search signal**, see deviation 2 |
| `best_message` / `best_score` tracking | yes, exposed as `best_message` |
| early stop at rating ≥ 10 | yes |
| `next_probe_prompt` from the last response | yes — the defining multi-turn behaviour |
| `is_final_probe` | yes — the probe is sent, then the attack stops |
| `num_actors`, `max_turns_per_actor` clamping | yes, `max(...)` exactly as upstream |

Not ported: `tqdm` progress bars (UI), `_generate_schema`/`simulator_model`
plumbing (the framework injects `self.llm`), and `enhance`'s return of
`best_message` — upstream is an *enhancer* returning one prompt, whereas here
the turns are the attack (see deviation 1).

## Deviations

### 1. The loop is the attack, not a prompt generator

Upstream's `enhance` runs the whole conversation internally against a
`model_callback` and then returns `best_message` for something else to send.
superred drives turns through the event loop, so each run sends one probe and
the target's reply arrives as a `ControllablePostCallEvent` — which is where
the next probe is written from. `best_message` is still tracked and exposed.

### 2. `judge` is kept, as an attacker-side signal

Elsewhere in these ports, upstream judges are dropped because scoring belongs
to the `SecurityClaim`. Here `judge` is not a verdict: it steers the search
(picking `best_message` and triggering the ≥10 early stop), exactly as PAIR's
and TAP's internal judges do. Dropping it would remove attack functionality, so
it is ported. The claim still decides whether the run actually succeeded.

### 3. A genuine budget exhaustion propagates

Every attacker-LLM failure is swallowed and the step degrades, matching
upstream's `except Exception: continue` — **except** a `BudgetExhaustedError`
with `usage.cost > 0`, which is re-raised. Absorbing that would end the attack
quietly and report a run the target never actually defended. The budget-less
noop client raises the same error with nothing spent and still degrades.

### 4. A target that emits no post-call event gets one probe

The next probe is written from the target's reply, which arrives as a
`ControllablePostCallEvent`. Targets are not obliged to emit one (superred's
own `RAGTarget`, for instance, does not). Rather than decline every surface
until the run budget is spent, the attack stops after the opening question when
a run ends without a post-call. Multi-turn interrogation therefore requires a
target that emits post-call events; against one that does not, this degrades to
a single goal-adjacent opening question.

The post-call response is a `ControllableNoInjection`: the channel type-checks
that event's response, and this attack reads the answer but never rewrites it.

### 5. Injection-surface selection (superred adaptation)

Follows the house pattern (`muzzle`, `chord_xthp`, `eia_agent`, `gepa_agentic`,
`poisonedrag`, `agentvigil_websentinel`): a byte-identical copy of the shared
`surface_llm.py`, with `classify_controllables` run once at `initialize`. The
interrogation is pinned to the surface that received the opening question, so
later probes continue the same conversation.

Note: the shared `surface_llm.py` pins `temperature=0.0`, which the repo's
`test_no_temperature.py` guard forbids. The six merged modules shipping the
classifier omit the guard; this module keeps it and skips only that one pinned
shared file.

## Known limitations

- Each turn costs two to three attacker-LLM calls (refusal check, judge, next
  probe) on top of the one-off network generation.
- Only `actors[0]` is used, as upstream. The remaining candidates are generated
  and discarded; nothing retries with a different actor.
