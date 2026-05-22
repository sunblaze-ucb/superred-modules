# Review: `env.py` + `seed_loader.py`

Scope:
- `targets/agentdojo/src/agentdojo_target/env.py`
- `targets/agentdojo/src/agentdojo_target/seed_loader.py`
- `targets/agentdojo/tests/test_env.py`
- `targets/agentdojo/tests/test_env_sync.py`
- `targets/agentdojo/tests/test_seed_loader.py`

Reviewer cross-checked behaviour against upstream AgentDojo at
`/Users/simonsure/research/superred/.venv/lib/python3.13/site-packages/agentdojo/`.

ASSUMPTIONS.md sections C.4 + C.5 cover the env-sync and composite-loading
decisions. Findings below are organised by severity.

---

## Summary

The composite root and the round-trip sync helper are well-factored: field
order is deterministic across reloads, no `inbox` / `calendar` name collision
exists on the composite (those live under `workspace.*` / `travel.*`), and the
audited set of `initial_*` shadow lists (inbox/calendar/cloud_drive) is in
fact the complete set in `agentdojo/default_suites/v1/tools/`. The seed loader
correctly delegates `{slot}` substitution to upstream's
`load_and_inject_default_environment({})`, version-pins to canonical `"v1"`,
and is repeatable per-call. The main gaps are an undocumented side-effect of
sync on `Inbox.contact_list` (the second model-validator), three missing
defensive tests, and a few error-path UX issues. No CRITICAL or HIGH issues.

---

## CRITICAL

None.

---

## HIGH

None.

---

## MEDIUM

### M1. `Inbox` has a second `@model_validator` that `sync_initial_fields` interacts with implicitly

`agentdojo/default_suites/v1/tools/email_client.py:32` defines
`_create_contact_list`, a second `@model_validator(mode="after")` on `Inbox`
that walks `initial_emails` and appends any sender/recipient/cc/bcc address
not already in `contact_list`.

`sync_initial_fields` rewrites `initial_emails` to mirror the **current**
`emails` dict. When the agent has called `send_email(...)` to a previously-unseen
recipient and then we serialise -> deserialise, the validator's "append
missing contacts" branch fires on re-validation and adds those new
recipients to `contact_list`. Reproduced manually:

```python
env = load_composite_seed()
contacts_pre = len(env.workspace.inbox.contact_list)              # 18
env.workspace.inbox.send_email(['attacker@evil.org'], 's', 'b')   # no contact added inline
sync_initial_fields(env)
rt = CompositeEnvironment.model_validate_json(env.model_dump_json())
len(rt.workspace.inbox.contact_list)                              # 19
'attacker@evil.org' in {c.email for c in rt.workspace.inbox.contact_list}  # True
```

Notes:

- The behaviour does **not** drift further on repeated round-trips
  (`_create_contact_list` checks membership before appending, verified at 3
  cycles). So this is a one-shot semantic shift, not unbounded growth.
- Upstream `Inbox.send_email` does **not** add the contact inline; AgentDojo's
  benchmark never re-validates a live `Inbox`, so this re-derivation never
  fires upstream.
- Impact: SecurityClaim predicates that inspect `contact_list` post-run will
  see a contact that AgentDojo upstream would not have populated. None of the
  upstream injection tasks I scanned look at `contact_list`, but the port
  doesn't promise to constrain future tasks.

Fix options (no code change requested):
1. Document the deviation in `env.py`'s `sync_initial_fields` docstring and
   in `ASSUMPTIONS.md` C.4.
2. Capture `contact_list` before sync and restore it afterwards.
3. Drop `contact_list` from the dump and let the validator rebuild from the
   synced `initial_emails` (cleanest, mirrors upstream behaviour for emails).

### M2. No explicit round-trip fixed-point test

`test_sync_idempotent` asserts that running `sync` twice on the same env is a
no-op, but no test pins the stronger property: *after sync, dump -> validate
-> sync -> dump returns the same JSON bytes*. I verified this holds (single
cycle) but a regression here would silently change downstream evaluation.

Suggested addition (one test in `test_env_sync.py`):

```python
def test_sync_then_round_trip_is_fixed_point() -> None:
    env = load_composite_seed()
    sync_initial_fields(env)
    dump1 = env.model_dump_json()
    rt = CompositeEnvironment.model_validate_json(dump1)
    sync_initial_fields(rt)
    assert rt.model_dump_json() == dump1
```

### M3. Sync helper covers exactly the 3 known shadow fields but does not guard against upstream additions

`sync_initial_fields` hardcodes 5 attribute paths (workspace.{inbox,
calendar, cloud_drive}, travel.{inbox, calendar}). I confirmed by greppping
`grep -rln 'initial_' agentdojo/default_suites/v1/tools/` that ONLY those 3
classes carry `initial_*` shadows in v1, but the codebase has no assertion
that catches drift if a future AgentDojo version adds another shadow field
(e.g. `initial_transactions` on `BankAccount`).

Suggested mitigation: add a startup-time check in `env.py` that walks each
sub-env's `model_fields` for any `initial_*` name and warns / fails if it
finds one that `sync_initial_fields` does not handle.

### M4. `merge_yaml_overlay` errors leak pydantic's `ValidationError` repr unchanged

The docstring says `ValueError` is raised on validation failure (correct,
since `ValidationError` IS a `ValueError`). But the call site uses
`type(sub).model_validate(merged)` and never catches; the propagated message
is pydantic's multi-line repr with paths like `bank_account.balance`. For an
attacker-supplied overlay (config slot), this could echo back the input or
internal structure in an error message.

Suggested mitigation: wrap the validate in try/except, raise a sanitised
`ValueError(f"Overlay for suite {suite_name!r} failed validation")` plus
optional logging of the full error, so attacker-controlled overlay text
cannot pivot through error messages into the trajectory.

---

## LOW

### L1. `_BENCHMARK_VERSION` is module-private and undocumented

`seed_loader.py:25` pins `_BENCHMARK_VERSION = "v1"`. The leading underscore
hides it from consumers who might reasonably want to point at `"v1.1"` for a
different benchmark cohort. Not a bug; flag for future config-surface
expansion if you ever support v1.1+ pinning.

Cross-check: upstream `agentdojo/task_suite/load_suites.py:63` builds
`_SUITES` as a `defaultdict(dict, ...)` so an unknown version returns `{}`
and `suite_name` lookup raises `KeyError: 'banking'` — a confusing message
because the caller passed a version, not a suite. Worth defensive validation
if `_BENCHMARK_VERSION` ever becomes user-tunable.

### L2. `_deep_merge` description in docstring is mildly misleading

The docstring says "Lists, in particular, are NOT merged element-wise; the
overlay list replaces the base list entirely." Accurate, but the function
also replaces non-dict scalars wholesale and replaces dicts that are nested
inside lists. The "Lists" wording reads as if list elements get special
treatment when they actually get the same wholesale-replace as scalars.
Minor doc clarification.

### L3. `merge_yaml_overlay` silently accepts extra fields (pydantic default)

`CompositeEnvironment.model_config` is empty (no `extra="forbid"`). A
maliciously-crafted YAML overlay containing unknown top-level keys is
silently dropped, not flagged. Acceptable for the threat model (attacker
gets no leverage from extra-field injection), but worth noting for the
record.

### L4. Duplicate-ID round-trip failure mode

If an attacker (or a buggy tool) inserts an email with a duplicate ID into
`inbox.emails` and `sync_initial_fields` runs, the subsequent re-validation
raises `ValidationError: Email IDs must be unique`. The exception is correct
and clearly explained but propagates up out of the target's `query` path
with a stack trace into the trajectory. Probably benign (this can only
happen with a misbehaving tool), but worth a one-line catch + skipped-task
behaviour if it ever shows up in practice.

### L5. Composite env `extra="ignore"` means `merge_yaml_overlay` cannot detect typos

If a caller supplies `{"bank_acount": {...}}` (typo, missing 'c'), the
overlay is silently dropped because pydantic ignores extras. Combined with
the absence of a "Did you mean..." hint, this is a footgun for human
overlay-writers (not a security concern).

---

## INFO

### I1. Field ordering is deterministic across reloads (verified)

`CompositeEnvironment.model_fields` and the JSON-dump key order both follow
declaration order (banking, workspace, slack, travel). Reloading the seed
twice produces byte-identical JSON. No issue.

### I2. No cross-suite `inbox` / `calendar` collision at the composite level (verified)

The composite root has zero flat fields named `inbox` or `calendar`. The
collision exists nominally between `workspace.inbox` and `travel.inbox`
(and the same for calendar) but each is its own pydantic instance and
`test_workspace_and_travel_inboxes_are_distinct` /
`test_workspace_and_travel_calendars_are_distinct` pin that they don't
alias.

### I3. `injection_vectors.yaml` substitution path is upstream-owned

`load_and_inject_default_environment({})` reads `environment.yaml`, fetches
defaults from `injection_vectors.yaml`, and calls `text.format(**defaults)`
before `yaml.safe_load` — all upstream code. Our seed_loader trusts this
chain. Confirmed in
`agentdojo/task_suite/task_suite.py:139-146`.

### I4. Pre-import side-effect for circular import is documented and correct

`env.py:32` and `tool_registry.py:31` both pre-import
`agentdojo.task_suite.load_suites` with a `# noqa: F401` comment. The
comment block in `env.py:23-31` clearly explains why (suite-package init
order). I confirmed by removing the import in a scratch venv that imports
fail without it. No issue.

### I5. `model_copy(deep=True)` correctly isolates the snapshot (test pinned)

`test_deep_copy_isolates_subenvs` proves mutating the original after a
deep-copy doesn't propagate. Good.

### I6. Trash bucket survives round-trip (Inbox has no `initial_trash`)

`delete_email` moves the email from `emails` to `trash`. Verified the trash
bucket survives the round-trip after sync (no shadow field to overwrite).
Good.

### I7. Module-level YAML files are version-pinned to v1 only

`seed_loader.py:25` is `_BENCHMARK_VERSION = "v1"` and the upstream
`get_suite("v1", ...)` returns the v1 suites with benchmark_version
`(1,0,0)` — no leak into v1.1 / v1.2 / v1.2.x. All v1 YAML data files exist
on disk and load cleanly.

### I8. `merge_yaml_overlay` returns a defensive deep copy

Even for an empty overlay (`""`), the function returns
`env.model_copy(deep=True)` rather than `env` itself, so callers cannot
accidentally mutate the input. Good defensive style.

### I9. `_create_files` validator on `CloudDrive` also recomputes file `size`

`CloudDriveFile` has its own `@model_validator(mode="after")` that recomputes
`size` from `len(content)`. This re-fires on round-trip. If a caller has
manually set a size that disagrees with `content`, the round-trip will
correct it. Benign and arguably desirable.

---

## Verification commands run during review

```bash
# Round-trip fixed-point manual check (passes)
python3 -c "
from agentdojo_target.env import sync_initial_fields, CompositeEnvironment
from agentdojo_target.seed_loader import load_composite_seed
env = load_composite_seed()
sync_initial_fields(env)
d1 = env.model_dump_json()
rt = CompositeEnvironment.model_validate_json(d1)
sync_initial_fields(rt)
assert rt.model_dump_json() == d1
"

# Determinism across reloads (passes)
python3 -c "
from agentdojo_target.env import sync_initial_fields
from agentdojo_target.seed_loader import load_composite_seed
a = load_composite_seed(); sync_initial_fields(a)
b = load_composite_seed(); sync_initial_fields(b)
assert a.model_dump_json() == b.model_dump_json()
"

# Confirm only inbox/calendar/cloud_drive have initial_*
grep -rln 'initial_' agentdojo/default_suites/v1/tools/

# Run existing tests
pytest tests/test_env.py tests/test_env_sync.py tests/test_seed_loader.py -q
# -> 21 passed
```
