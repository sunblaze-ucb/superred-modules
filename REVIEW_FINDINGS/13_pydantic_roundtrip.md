# Pydantic Round-Trip Integrity Audit

Scope: every pydantic model in `agentdojo/default_suites/v1/tools/` (the only
sub-tree carrying validators in upstream) plus the `CompositeEnvironment`
JSON round-trip used by `Target.query("pre|post_environment_snapshot")`.

## 1. Inventory of `@model_validator(mode="after")` in upstream

Exhaustive `grep -l model_validator` across the upstream `agentdojo/`
package returns four files:

| File | Class | Validator | Rebuilds from `initial_*`? |
| --- | --- | --- | --- |
| `tools/email_client.py:23` | `Inbox` | `_create_emails` | yes (`initial_emails` -> `emails`) |
| `tools/email_client.py:32` | `Inbox` | `_create_contact_list` | uses `initial_emails`, append-only into `contact_list` |
| `tools/calendar_client.py:18` | `Calendar` | `_create_events` | yes (`initial_events` -> `events`) |
| `tools/cloud_drive_client.py:16` | `CloudDrive` | `_create_files` | yes (`initial_files` -> `files`) |
| `tools/types.py:29` | `CloudDriveFile` | `compute_size` | no `initial_*` pattern; just sets `size = len(content)` |

Two further validators exist in upstream (`agentdojo/benchmark.py`
`TaskResults.check_messages`, `agent_pipeline/agent_pipeline.py`
`PipelineConfig.validate_system_message`) but they are not in any
environment model and never touch the JSON round-trip. They are out of
scope.

The other tool files contain **no** validators:

- `tools/banking_client.py` (`Transaction`, `BankAccount`) - plain fields
- `tools/slack.py` (`Message`, `Slack`) - plain fields
- `tools/travel_booking_client.py` (`User`, `Hotel`, `Hotels`, `Flight`,
  `Flights`, `Reservation`, `Restaurant`, `Restaurants`,
  `CarRentalCompany`, `CarRental`) - plain fields
- `tools/web.py` (`Web`) - plain fields
- `tools/file_reader.py` (`Filesystem`) - plain fields
- `tools/user_account.py` (`UserAccount`) - plain fields

`v1_1*`, `v1_2*` suite directories carry no `tools/` subtree and add no
new validator-bearing models.

## 2. Coverage of `sync_initial_fields`

`sync_initial_fields` in `targets/agentdojo/src/agentdojo_target/env.py:66`
writes back to exactly five locations:

```python
env.workspace.inbox.initial_emails       = list(env.workspace.inbox.emails.values())
env.workspace.calendar.initial_events    = list(env.workspace.calendar.events.values())
env.workspace.cloud_drive.initial_files  = list(env.workspace.cloud_drive.files.values())
env.travel.inbox.initial_emails          = list(env.travel.inbox.emails.values())
env.travel.calendar.initial_events       = list(env.travel.calendar.events.values())
```

Cross-checking against the upstream environment layouts:

- `WorkspaceEnvironment` (`workspace/task_suite.py:31-34`) -> `inbox`,
  `calendar`, `cloud_drive`. All three covered.
- `TravelEnvironment` (`travel/task_suite.py:31-39`) -> `inbox`,
  `calendar` use validator-bearing classes. `User`, `Hotels`,
  `Restaurants`, `Flights`, `Reservation`, `CarRental` don't.
  **TravelEnvironment has no `cloud_drive` attribute** despite the
  `root.cloud_drive.initial_files` line in `travel/task_suite.py:106`
  (that string is unused legacy from a copy-paste; nothing reads it).
  Both validator-bearing fields are covered.
- `BankingEnvironment` -> no validator-bearing fields.
- `SlackEnvironment` -> no validator-bearing fields.

**Coverage is complete for the `initial_* -> derived dict` pattern.** No
missing model.

## 3. Non-`initial_*` validators - are any lossy on round-trip?

### `Inbox._create_contact_list`

`contact_list` is itself a serialized field. The validator only appends
contacts not already present and never removes. So:

- Contacts added by historical-but-since-deleted emails persist
  (`contact_list` is round-tripped verbatim).
- Direct mutations of `contact_list` survive (verified by appending a
  ghost contact, syncing, and round-tripping - still present).

The validator is idempotent w.r.t. serialized state. **No leak.**

### `CloudDriveFile.compute_size`

`size` is unconditionally recomputed from `len(content)` on every
validate. There is no agent path that mutates `size` independent of
`content` (`append_to_file` updates both together). Round-trip is
consistent. **No leak.**

### `Inbox.trash`

`trash` is a regular dict field with no validator. Verified empirically:
`delete_email` populates `trash`, the dict survives `model_dump_json ->
model_validate_json` unchanged. **No leak.**

## 4. Fields beyond `initial_*` that are validator-rebuilt

None. Searched for the pattern across all validators - the only fields
rebuilt from a non-self source are `emails`, `events`, `files` (from
`initial_*`). `contact_list` and `size` are rebuilt from sibling fields
that are themselves the source of truth, so the rebuild is faithful.

## 5. Enum round-trip

Upstream defines `agentdojo/strenum.py:StrEnum` as `class StrEnum(str,
enum.Enum)` plus a yaml SafeDumper representer. This is essentially the
same as Python 3.11+ `enum.StrEnum` and round-trips as `str` through
pydantic.

Verified by running:

```python
Reservation(reservation_type=ReservationType.HOTEL, ...)
.model_dump_json()  -> "reservation_type":"hotel"
ReservationType.model_validate_json(raw).reservation_type == ReservationType.HOTEL
```

Returned `True`. Same for `EvenStatus`, `EmailStatus`,
`SharingPermission`. The `Reservation.reservation_type` field is
`ReservationType | None`; the `None` case round-trips as JSON null and
deserialises back to `None`. **No leak.**

## 6. Datetime round-trip

Seed YAMLs use **naive** datetimes (no timezone suffix, e.g.
`2024-05-15T10:00`). Pydantic preserves naivete on round-trip: a naive
datetime serialises to an ISO string without `Z`/offset and parses back
to a naive datetime. Spot-checked `CloudDriveFile.last_modified` with a
naive value - `tzinfo` is `None` both before and after.

The only datetime fields that could be timezone-aware are runtime
mutations from `datetime.datetime.now()` calls in tools like
`Inbox.send_email` and `CloudDrive.create_file`. These produce naive
datetimes too (Python's `datetime.now()` without `tz=` argument). No
heterogeneous-tz field exists in the seeds. **No leak.**

(Note: if any future evaluator or task ever attaches a timezone-aware
datetime, pydantic preserves the offset on round-trip, so this is
forward-compatible too.)

## 7. Other dicts mutated by agent tools

Other plain-dict mutations that round-trip via pydantic without a
helper (cross-checked empirically):

- `Slack.user_inbox`, `Slack.channel_inbox`, `Slack.user_channels`
- `BankAccount.transactions`, `BankAccount.scheduled_transactions`
- `Web.web_content`, `Web.web_requests`
- `Inbox.trash`, `Inbox.contact_list` (see above)
- `CloudDriveFile.shared_with`
- `Reservation.reservation_type`, `Reservation.start_time`, etc.

All survive `model_dump_json -> model_validate_json` because they are
fields, not derived from a source-of-truth.

## 8. Verdict

**No missed models.** `sync_initial_fields` covers every
validator-bearing path that rebuilds from an `initial_*` field. The
other validators are idempotent. Enum and datetime round-trips are
faithful for the seeds and tool-emitted values that exist today. No
HIGH-severity gap.

Minor cosmetic note (no severity): `env.py:86` doc-comment refers to
"`ASSUMPTIONS.md` §C.3"; the actual section is `C.4`. This is
documentation drift, not a bug.

## Files referenced

- `/Users/simonsure/research/superred/.venv/lib/python3.13/site-packages/agentdojo/default_suites/v1/tools/email_client.py:16-46`
- `/Users/simonsure/research/superred/.venv/lib/python3.13/site-packages/agentdojo/default_suites/v1/tools/calendar_client.py:12-25`
- `/Users/simonsure/research/superred/.venv/lib/python3.13/site-packages/agentdojo/default_suites/v1/tools/cloud_drive_client.py:11-23`
- `/Users/simonsure/research/superred/.venv/lib/python3.13/site-packages/agentdojo/default_suites/v1/tools/types.py:17-32`
- `/Users/simonsure/research/superred/.venv/lib/python3.13/site-packages/agentdojo/default_suites/v1/tools/travel_booking_client.py:10-66`
- `/Users/simonsure/research/superred/.venv/lib/python3.13/site-packages/agentdojo/strenum.py`
- `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-port/targets/agentdojo/src/agentdojo_target/env.py:66-95`
- `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-port/targets/agentdojo/tests/test_env_sync.py`
- `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-port/targets/agentdojo/ASSUMPTIONS.md:83-89`
