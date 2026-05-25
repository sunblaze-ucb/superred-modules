"""Tests for the seed loader and per-suite YAML overlay merger."""

from __future__ import annotations

import json

import pytest

from agentdojo_target.env import CompositeEnvironment
from agentdojo_target.seed_loader import load_composite_seed, merge_yaml_overlay


def test_seed_returns_composite() -> None:
    """The loader returns a fully-constructed CompositeEnvironment."""
    env = load_composite_seed()
    assert isinstance(env, CompositeEnvironment)


def test_seed_substitutes_injection_defaults() -> None:
    """Injection vector defaults are substituted; the parsed env has no
    raw ``{slot}`` text in the banking transaction subject."""
    env = load_composite_seed()
    subjects = [t.subject for t in env.banking.bank_account.transactions]
    for s in subjects:
        assert "{injection_" not in s, f"unparsed slot leaked into {s!r}"


def test_seed_loads_all_four_suites_populated() -> None:
    """Each sub-env has its expected populated fields."""
    env = load_composite_seed()
    # banking
    assert env.banking.bank_account.iban
    assert env.banking.user_account.first_name
    assert env.banking.filesystem.files  # at least one file
    # workspace
    assert env.workspace.inbox.emails
    assert env.workspace.calendar.events
    assert env.workspace.cloud_drive.files
    # slack
    assert env.slack.slack.users
    assert env.slack.web.web_content
    # travel
    assert env.travel.user.first_name
    assert env.travel.hotels.hotel_list
    assert env.travel.flights.flight_list


def test_merge_overlay_empty_returns_copy() -> None:
    """Empty overlay returns a deep copy unchanged."""
    env = load_composite_seed()
    out = merge_yaml_overlay(env, "banking", "")
    assert out is not env
    assert out.banking.bank_account.iban == env.banking.bank_account.iban


def test_merge_overlay_json_partial_override() -> None:
    """JSON overlay overrides a single nested field; others retained."""
    env = load_composite_seed()
    overlay = json.dumps({"bank_account": {"balance": 0.0}})
    out = merge_yaml_overlay(env, "banking", overlay)
    assert out.banking.bank_account.balance == 0.0
    # iban (not in overlay) is preserved
    assert out.banking.bank_account.iban == env.banking.bank_account.iban
    # original is untouched
    assert env.banking.bank_account.balance != 0.0


def test_merge_overlay_yaml_partial_override() -> None:
    """YAML overlay (no leading {) works the same way."""
    env = load_composite_seed()
    overlay = "bank_account:\n  balance: 42.0\n"
    out = merge_yaml_overlay(env, "banking", overlay)
    assert out.banking.bank_account.balance == 42.0


def test_merge_overlay_unknown_suite_rejected() -> None:
    """An unrecognised suite name raises ValueError."""
    env = load_composite_seed()
    with pytest.raises(ValueError, match="Unknown suite"):
        merge_yaml_overlay(env, "nonsense", "{}")


def test_merge_overlay_invalid_json_rejected() -> None:
    """Malformed JSON-looking overlay raises ValueError."""
    env = load_composite_seed()
    with pytest.raises(ValueError, match="JSON"):
        merge_yaml_overlay(env, "banking", "{not valid json}")


def test_merge_overlay_non_dict_rejected() -> None:
    """An overlay that parses to a non-dict is rejected."""
    env = load_composite_seed()
    with pytest.raises(ValueError, match="must encode a dict"):
        merge_yaml_overlay(env, "banking", "[1, 2, 3]")


def test_merge_overlay_validation_failure_rejected() -> None:
    """An overlay that fails pydantic validation propagates ValidationError as ValueError."""
    env = load_composite_seed()
    # balance must be a number; passing a non-numeric should fail
    with pytest.raises(Exception):  # noqa: BLE001 - pydantic raises ValidationError
        merge_yaml_overlay(env, "banking", '{"bank_account": {"balance": "not a number"}}')
