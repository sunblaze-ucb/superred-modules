"""Unit tests for AsbTarget lifecycle and surface (offline)."""

from __future__ import annotations

import asyncio
import json

import pytest

from asb_target import DOMAIN, AsbTarget
from asb_target.controllables import CONTROLLABLES


def _fake_embed(texts):  # type: ignore[no-untyped-def]
    return [[float(len(t))] for t in texts]


def _target() -> AsbTarget:
    return AsbTarget(
        api_base="http://unused", api_key="unused", request_delay_seconds=0, embed=_fake_embed
    )


def test_instantiates_against_framework() -> None:
    # The reset hook is named reset_ephemeral_state (framework abstractmethod);
    # a wrongly-named cleanup() would make the class non-instantiable.
    t = _target()
    assert not type(t).__abstractmethods__
    assert hasattr(t, "reset_ephemeral_state")
    assert not hasattr(t, "cleanup")


def test_security_domain_and_controllables() -> None:
    t = _target()
    assert t.security_domain is DOMAIN
    assert t.get_controllables() == CONTROLLABLES


def test_config_slots_set_and_validate() -> None:
    t = _target()
    t.set_config("agent_name", "system_admin_agent")
    t.set_config("user_prompt", "check the logs")
    t.set_config("attacker_tool", json.dumps({"Attacker Tool": "X", "Attack goal": "G"}))
    t.set_config("memory_mode", "on")
    with pytest.raises(ValueError):
        t.set_config("nonexistent", "x")
    # removed slots must raise (no longer exposed)
    for removed in ("defense_type", "pot_trigger", "llm_model"):
        with pytest.raises(ValueError):
            t.set_config(removed, "x")


def test_memory_mode_parsing() -> None:
    t = _target()
    for truthy in ("on", "1", "true", "yes"):
        t.set_config("memory_mode", truthy)
        assert t._memory_mode is True
    for falsey in ("", "off", "false", "0"):
        t.set_config("memory_mode", falsey)
        assert t._memory_mode is False


def test_query_slots_and_validation() -> None:
    t = _target()
    assert t.query("workflow_failure") == "0"
    assert t.query("messages") == "[]"
    assert t.query("memory_found") == ""
    with pytest.raises(ValueError):
        t.query("nope")


def test_reset_ephemeral_state_preserves_durable_memory() -> None:
    t = _target()
    t._memory.add("Agent: a; Task: t; Workflow: [w]; Tools: []", {"source": "a"})
    t._messages = [{"role": "user", "content": "x"}]
    asyncio.run(t.reset_ephemeral_state())
    assert t.query("messages") == "[]"  # ephemeral state cleared
    assert len(t._memory) == 1  # durable memory preserved


def test_get_observables() -> None:
    t = _target()
    t.set_config("agent_name", "system_admin_agent")
    t.set_config(
        "attacker_tool",
        json.dumps({"Attacker Tool": "Evil", "Description": "d", "Attack goal": "g"}),
    )
    obs = t.get_observables()
    names = {o.observable.name for o in obs}
    assert names == {"system_prompt", "model_identity", "tool_catalog_listing"}
    assert not any("attack_reference" in n for n in names)
    model = next(o.content for o in obs if o.observable.name == "model_identity")
    assert model == t._model
    catalog = next(o.content for o in obs if o.observable.name == "tool_catalog_listing")
    catalog_names = {entry["name"] for entry in catalog}
    assert "Evil" in catalog_names  # attacker tool listed once registered
