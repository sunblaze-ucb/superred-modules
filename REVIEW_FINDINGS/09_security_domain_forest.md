# Security-Domain Forest Review

Scope: how the `SecurityDomain` forest used by `AgentDojoTarget` is constructed in `security_tags.py`, wired into `Controllable`s (`controllables.py`), `Observable`s (`observables.py`), `ConfigSpec`s (`config_specs.py`), `RegistryEntry`-derived per-call ctrls (`runtime_wrapper.py`), and `Score`s in the claim package (`layer1_task.py`, `layer2_task.py`). Cross-checks against the rationale-doc test (`tests/test_quadrant_rationale.py`), the topology test (`tests/test_security_tags.py`), and the framework primitives (`superred/core/types/security_domain.py`, `controller.py`).

## Summary

The forest is built coherently: three trees (system / user / tools) with 17 module-level singleton tags, an explicit subsumption hierarchy on `system` (prompt, tool_catalogue, agent_trace), a flat `user` tag, and a 2x2 quadrant grid on `tools`. All 47 read Controllables receive a quadrant tag matched to a corresponding rationale entry, and read-mirror Observables reuse the controllable's tag identity (`is`, not `==`). The forest builder declares parent-child links correctly and `scope_includes` works against this graph live (verified). Three structural concerns: (1) `SecurityDomainTag` is value-equal because it is a `@dataclass(frozen=True)`, which contradicts the brief's "identity-based" framing and could mask an accidental tag rebuild; (2) Layer-1 and Layer-2 evaluations tag every sub_score with `USER_TAG`, so scopes that exclude `USER_TAG` silently lose all diagnostic sub-scores; (3) the `seed_yaml_override__{suite}` ConfigSpec is tagged `CONTENT_1P_DATA_1P_TAG`, which doesn't fit the slot's actual content semantics (it commonly carries 3p-authored seed records). Otherwise the design holds up.

## Findings

### F1. [MEDIUM] `SecurityDomainTag` equality is value-based, not identity-based

**Location**: `/Users/simonsure/research/superred/superred/src/superred/core/types/security_domain.py:31-50`

```python
@dataclass(frozen=True)
class SecurityDomainTag:
    name: str
    parent: SecurityDomainTag | None = None
```

A `@dataclass(frozen=True)` generates `__eq__` and `__hash__` from field values. Live check:

```
SecurityDomainTag('foo') == SecurityDomainTag('foo')  # True
SecurityDomainTag('foo') is SecurityDomainTag('foo')   # False
hash(a) == hash(b)                                      # True
b in frozenset({a})                                     # True
```

But the membership test that actually authorizes events is identity-based: `tag.includes(other)` walks `current is self` (security_domain.py:48), and `scope_includes` calls `s.includes(tag)`. So:

- `frozenset({USER_TAG_v1, USER_TAG_v2})` collapses to ONE element by hash (value-equal).
- `scope_includes(frozenset({USER_TAG_v1}), controllable_tagged_with_USER_TAG_v2)` returns **False** because `s.includes(other)` cannot reach `other` via parent links if `other` is a separately-allocated object.

The current port does **not** trigger this bug: every tag has a single construction site in `security_tags.py` and consumers import that exact singleton (verified live: `USER_PROMPT_CTRL.security_domain is USER_TAG`, `read_data_field_observable(...).security_domain is READ_CTRLS[...].security_domain`). But the framework's mixed semantics are a landmine — if a future module accidentally re-creates a tag with the same name, equality says "same" and includes says "different", producing silent unauthorized blocks. Fix space:
- Override `__eq__` / `__hash__` in `SecurityDomainTag` to use `id(self)`.
- Or document the singleton contract in `SecurityDomainTag`'s docstring and add a `SecurityDomain.__init__` check that all tags are not just unique by name but unique by identity in the module's tag registry.

This is a framework-level finding, not a port-level finding, but it should be raised because the AgentDojo port has 17 tags and one accidental shadow construction would be invisible.

### F2. [MEDIUM] All Layer-1 / Layer-2 sub_scores are tagged `USER_TAG`; scopes without `USER_TAG` lose diagnostic feedback

**Locations**:
- `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-claim/security_claims/agentdojo/src/security_claim_agentdojo/layer1_task.py:150-176`
- `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-claim/security_claims/agentdojo/src/security_claim_agentdojo/layer2_task.py:138-159`

Every score (primary `attack_succeeded`, `category_*`, `suite_*`, `utility`, `targeted`, plus Layer-2's `env_mutated`, `violated_via_write`) is constructed with `security_domain=USER_TAG`. The controller filters `sub_scores` at `controller.py:736-740`:

```python
filtered_sub = {
    k: v for k, v in evaluation.sub_scores.items()
    if v.security_domain is None or scope_includes(scope, v.security_domain)
}
```

Consequence: a threat-model sweep with `scope=frozenset({TOOLS_TAG})` or `scope=frozenset({CONTENT_3P_DATA_3P_TAG})` (no `USER_TAG`) will see the primary score (always preserved) but **lose every sub_score** because none of them is in scope. That kills the utility/category/suite diagnostics, even though such a scope is semantically the most natural for AgentDojo's prompt-injection-from-tool-output threat model.

The brief's framing supports `tools.*`-only scopes as primary: AgentDojo attackers inject through tool outputs, not the user channel. The smoke and integration test scopes happen to bundle `USER_TAG` (see `test_integration.py:138-140`), masking the issue. A clean fix is one of:
- Set `security_domain=None` on diagnostic sub_scores (they become always-visible).
- Set the score's tag to match the actual surface (e.g. `category_prompt_injection` → the quadrant the injection arrived through; `utility` → no scope; `env_mutated` → the tool catalogue or trace tag, since it reflects agent writes).
- Use the score-level scope to surface only those scores whose tag the attacker actually controls; declare this contract in `ASSUMPTIONS.md`.

This is the strongest semantic gap in the port-level wiring of the forest.

### F3. [LOW] `seed_yaml_override__{suite}` ConfigSpec tag is `CONTENT_1P_DATA_1P_TAG`

**Location**: `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-port/targets/agentdojo/src/agentdojo_target/config_specs.py:67-76`

```python
def _seed_yaml_override_spec(suite: str) -> ConfigSpec:
    return ConfigSpec(
        name=f"seed_yaml_override__{suite}",
        security_domain=CONTENT_1P_DATA_1P_TAG,
        ...
```

`CONTENT_1P_DATA_1P_TAG` represents "1p-authored content in 1p storage" — the rationale doc gives `workspace.get_current_day` (system clock) as the canonical example. But the actual seed overlays applied via this slot routinely carry 3p-authored content: an `init_environment` mutation for a workspace task may insert emails from external senders, calendar invites authored by 3p users, or hotel review records authored by 3p users — content that maps to `content_3p_data_3p` or `content_1p_data_3p`, not `1p/1p`.

The seat for this tag is a Task-side concern: ConfigSpec slots are NOT filtered by `scope_includes` in the controller (see `controller.py:489-507` — only Controllables and Observables are filtered). So the tag is documentation, not access control. Still, the documented "trust boundary" for the slot is inconsistent with what flows through it. Two cleaner options:

- Use `TOOLS_TAG` (the broadest tools root) — covers all four quadrants because the overlay can populate any of them.
- Drop the tag concept here and document the slot as "Task-owned, not attacker-reachable" without a forest position.

This is informational severity because the slot is not authorization-gated, but the lone-leaf assignment is semantically misleading.

### F4. [LOW] No test pins identity-equality of `Controllable.security_domain` to the `security_tags.py` singletons

**Locations**:
- `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-port/targets/agentdojo/tests/test_security_tags.py`
- `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-port/targets/agentdojo/tests/test_quadrant_rationale.py`

The topology tests verify tree shape and subsumption; the rationale-doc test verifies `READ_QUADRANT_MAP` per-tool quadrants. Neither test pins:

- `Controllable.security_domain is <expected-canonical-singleton>` for the system_prompt / user_prompt / catalogue / per-read controllables.
- `Observable.security_domain is <expected-canonical-singleton>` for the static observable specs and the read-mirror observables.
- `ConfigSpec.security_domain is <expected-canonical-singleton>` for the config slots.

Today these all hold (verified live), but absent a guard test, a refactor that introduces a stray construction or import-cycle workaround would not fail tests until a downstream scope mismatch happens. Adding 1-3 `assert <thing>.security_domain is <singleton>` tests per category would be cheap. Concretely, the value-equality issue (F1) means a refactor could pass `==` tests and still silently mis-authorize.

### F5. [LOW] `RunEndEvent.security_domain` is `next(iter(scope))` which is non-deterministic for multi-tag scopes

**Location**: `/Users/simonsure/research/superred/superred/src/superred/core/controller.py:751-754`

```python
run_end = RunEndEvent(
    evaluation=run_end_eval,
    security_domain=next(iter(scope)),
)
```

For scopes like `frozenset({USER_TAG, PROMPT_TAG, TOOL_CATALOGUE_TAG, CONTENT_3P_DATA_3P_TAG})` (the integration test scope), the tag that ends up on the persisted `RunEndEvent` is set-iteration order, which is hash-randomized per Python invocation. This is a framework-level concern (not introduced by the port), but it does mean trajectory snapshots are not bit-stable across runs. Worth either picking a deterministic choice (e.g. the lexically smallest tag name) or documenting the non-determinism. Surfaces in the port's persistence outputs but doesn't change semantic correctness.

### F6. [INFO] Tag count matches: 17 declared, 3 roots, 9214 distinct combinations

Live verification:

```
Roots: ['system', 'tools', 'user']
READ_FUNCTION_NAMES count: 47
READ_CTRLS count: 47
READ_QUADRANT_MAP count: 47
Quadrant distribution: {content_1p_data_3p: 11, content_3p_data_3p: 34, content_3p_data_1p: 1, content_1p_data_1p: 1}
Distinct combinations: 9214
```

47 read tools all present; quadrant distribution matches `test_quadrant_rationale.py:347-353` exactly (`{1p_1p: 1, 1p_3p: 11, 3p_1p: 1, 3p_3p: 34}`). The rationale doc is consistent with the live `READ_QUADRANT_MAP`.

The 9214 distinct combinations is the Cartesian product across trees: the system tree (10 tags) has ~232 antichains, the tools tree (4 leaves siblings + root) has 6 antichains (any subset of the 4 leaves OR just the root), the user tree has 2 antichains (empty or {user}). 232 * 6 * 2 = 2784 — actually the system tree antichain count is larger when prompt and tool_catalogue and agent_trace each have children. The takeaway is that the controller's sweep space (one threat model per distinct combination) is well-defined and not absurdly small.

### F7. [INFO] Read-mirror observables share Controllable identity (verified by live `is` check)

**Location**: `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-port/targets/agentdojo/src/agentdojo_target/observables.py:152-177`

```python
def read_data_field_observable(tool_prefixed_name: str) -> Observable:
    ctrl = READ_CTRLS.get(tool_prefixed_name)
    ...
    return Observable(
        name=f"read_data_field:{tool_prefixed_name}",
        security_domain=ctrl.security_domain,
        ...
```

Mirror observables reuse the controllable's `security_domain` object directly (verified live: `obs.security_domain is ctrl.security_domain` for `banking__get_iban`). This means a scope that includes the per-quadrant tag sees both the controllable event and the mirror observable, and a `tools.<leaf>`-only scope reading via the observable channel can see the legitimate value without injection rights. The pattern is consistent and well-realised.

### F8. [INFO] Forest topology and subsumption assertions are well-covered in tests

`tests/test_security_tags.py` covers:
- Three roots are exactly `{system, user, tools}`.
- `SYSTEM_TAG.includes(<every system descendant>)`.
- `TOOL_CATALOGUE_TAG.includes(TOOL_CATALOGUE_READABLE_TAG | TOOL_CATALOGUE_ADDABLE_TAG)`.
- `PROMPT_TAG.includes(PROMPT_READABLE_TAG)` and the converse fails.
- `AGENT_TRACE_TAG.includes(*messages | tool_calls | tool_responses)`.
- 2x2 grid leaves are siblings (none includes another, all included by `TOOLS_TAG`).
- Trees do not cross-include.
- `scope_includes` agrees with `tag.includes`.
- `DOMAIN` size is 17 (via distinct_combinations enumeration).

`tests/test_quadrant_rationale.py` covers:
- Every read tool has a rationale row (47).
- No stale rationale entries.
- Each tool's `READ_QUADRANT_MAP` entry matches its rationale row (parametrized 47x).
- No typoed quadrant name.
- Read-tool count is pinned at 47.
- Per-quadrant distribution is pinned at `{1, 11, 1, 34}`.

These tests pin every dimension of the forest the brief calls out, plus the per-tool rationale. The drift surface is small and the test failure messages name the file to update.

## Strengths

- **Single source of truth**: `security_tags.py` is the only construction site for the 17 forest tags; all consumers (controllables, observables, config_specs, runtime_wrapper, claim Tasks) import the singletons. Verified live with `is` checks across the boundary.
- **Documented subsumption**: the docstring for each non-leaf tag spells out what it includes (e.g. `TOOL_CATALOGUE_TAG` -> "implies replace, unregister, rewrite-description, and addable"). Future readers don't need to derive the tree from imports.
- **Per-tool rationale**: `test_quadrant_rationale.py` couples each of the 47 read tools to a one-line justification grounded in the content/data convention. A divergence from the brief's Section 4.a table would surface as a test failure with the named tool.
- **Conservative-broader-quadrant rule**: when a read can span multiple quadrants (e.g. `workspace.search_emails` covers both received and sent), the broader leaf is chosen. This means a narrow scope cannot accidentally reach a tool whose returns include third-party content, which is the safer default.
- **Read-mirror observable reuses controllable identity**: this is the right pattern — a `tools.<leaf>` scope can read the pre-injection legitimate value via the observable channel even when the controllable is filtered out. The `is` reuse guarantees the two surfaces are co-scoped.
- **Catalog-editing controllables partitioned across two write tags**: `register` -> `TOOL_CATALOGUE_ADDABLE_TAG` (weakest), `replace/unregister/rewrite_doc` -> `TOOL_CATALOGUE_TAG` (broader). The per-call ctrl built in `runtime_wrapper._attacker_call_ctrl` inherits the right tag based on `entry.kind`. Models the malicious-MCP-can-only-add threat cleanly.
- **Forest construction is validated at import**: `SecurityDomain.__init__` rejects duplicate names and dangling parents; `_build_read_controllables` raises if `READ_QUADRANT_MAP` misses any read tool or has stale entries. Import-time failure modes are loud.

## Open questions

1. **Should `SecurityDomainTag` be identity-equal at the framework level?** F1 documents the mixed semantics. Picking one consistent rule (identity-only, or value-equal with `includes` updated to use `__eq__`) would close the landmine. Out of scope for this port, but worth raising in the framework.
2. **Is the choice of `USER_TAG` on every sub_score deliberate?** F2 is the most consequential port-level finding. If the threat model intends `tools.*`-only scopes to be valid sweeps (which the brief's threat model suggests), the sub_score tagging needs rework. If the intent is "AgentDojo sub_scores only flow when the attacker has user-channel access", that needs to be documented in `ASSUMPTIONS.md` so users know to include `USER_TAG` in production scopes.
3. **Is `seed_yaml_override__{suite}` better tagged `TOOLS_TAG` or none?** F3. The slot accepts arbitrary YAML/JSON that becomes part of the suite sub-env at run start, but it is a Task-owned configuration channel, not an attacker-reachable surface. The current tag misrepresents the slot's content.
4. **Should the rationale doc test additionally pin Controllable / Observable identity?** F4. A small addition would prevent value-equality silently passing if a future refactor reintroduces a tag.
5. **Should the controller pick a deterministic tag for `RunEndEvent.security_domain`?** F5. Persisted trajectories vary across runs for multi-tag scopes today. Picking the lexically smallest tag name (or the unique-deepest-common-ancestor) would stabilise serialised outputs.
