"""Round-trip tests for :func:`sync_initial_fields`.

Without this sync, an in-memory mutation to ``inbox.emails`` /
``calendar.events`` / ``cloud_drive.files`` is discarded by the
pydantic ``@model_validator`` that rebuilds the derived dict from the
``initial_*`` list on ``model_validate``.  These tests pin the
expected behaviour.
"""

from __future__ import annotations

from agentdojo_target.env import CompositeEnvironment, sync_initial_fields
from agentdojo_target.seed_loader import load_composite_seed


def test_unsynced_email_delete_is_lost_on_round_trip() -> None:
    """Establishes the bug that motivates sync_initial_fields: mutating
    only ``emails`` (not ``initial_emails``) is lost on round-trip."""
    env = load_composite_seed()
    pre_count = len(env.workspace.inbox.emails)
    assert pre_count >= 3
    # Mutate emails dict only (simulating an agent tool call) BUT skip
    # the sync helper.
    victim = next(iter(env.workspace.inbox.emails))
    del env.workspace.inbox.emails[victim]
    # Round-trip without sync -> emails is rebuilt from initial_emails.
    raw = env.model_dump_json()
    rt = CompositeEnvironment.model_validate_json(raw)
    assert len(rt.workspace.inbox.emails) == pre_count  # mutation lost


def test_synced_email_delete_survives_round_trip() -> None:
    """With sync_initial_fields, the round-trip preserves the deletion."""
    env = load_composite_seed()
    pre_count = len(env.workspace.inbox.emails)
    victim = next(iter(env.workspace.inbox.emails))
    del env.workspace.inbox.emails[victim]
    sync_initial_fields(env)
    raw = env.model_dump_json()
    rt = CompositeEnvironment.model_validate_json(raw)
    assert len(rt.workspace.inbox.emails) == pre_count - 1
    assert victim not in rt.workspace.inbox.emails


def test_synced_calendar_event_cancellation_survives() -> None:
    """A canceled-status flip on a calendar event must survive round-trip."""
    env = load_composite_seed()
    # Pick an event that is currently confirmed.
    eid, evt = next(
        (eid, ev)
        for eid, ev in env.workspace.calendar.events.items()
        if str(ev.status) != "canceled"
    )
    # Mutate the event in-place to mark it canceled.
    # Pydantic doesn't enforce immutability here; field assignment works.
    from agentdojo.default_suites.v1.tools.calendar_client import EvenStatus

    env.workspace.calendar.events[eid].status = EvenStatus.canceled
    sync_initial_fields(env)
    raw = env.model_dump_json()
    rt = CompositeEnvironment.model_validate_json(raw)
    assert str(rt.workspace.calendar.events[eid].status) == "canceled"


def test_synced_cloud_drive_file_create_survives() -> None:
    """Adding a new file to ``files`` (then syncing) survives round-trip."""
    env = load_composite_seed()
    from agentdojo.default_suites.v1.tools.types import (
        CloudDriveFile,
        CloudDriveFileID,
    )

    new_id = CloudDriveFileID("99999")
    env.workspace.cloud_drive.files[new_id] = CloudDriveFile(
        id_=new_id,
        filename="attacker_synth.txt",
        content="hello",
        size=5,
        owner=env.workspace.cloud_drive.account_email,
        last_modified="2024-05-20T12:00:00",
    )
    sync_initial_fields(env)
    raw = env.model_dump_json()
    rt = CompositeEnvironment.model_validate_json(raw)
    assert "99999" in rt.workspace.cloud_drive.files


def test_sync_returns_same_env_for_chaining() -> None:
    """Convenience: sync returns the same instance so it can chain."""
    env = load_composite_seed()
    assert sync_initial_fields(env) is env


def test_sync_idempotent() -> None:
    """Running sync twice in a row is a no-op."""
    env = load_composite_seed()
    sync_initial_fields(env)
    before = env.model_dump_json()
    sync_initial_fields(env)
    after = env.model_dump_json()
    assert before == after
