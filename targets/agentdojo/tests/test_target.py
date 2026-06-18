"""Unit tests for :class:`AgentDojoTarget`.

These tests do NOT invoke a real LLM; they verify the lifecycle
contract, config/query slot dispatch, observable enumeration,
reset_ephemeral_state semantics, and security-domain forest exposure.
"""

from __future__ import annotations

import json

import pytest

from agentdojo_target.config_specs import CONFIG_SPECS
from agentdojo_target.controllables import CONTROLLABLES
from agentdojo_target.query_specs import QUERY_SPECS
from agentdojo_target.security_tags import DOMAIN
from agentdojo_target.target import AgentDojoTarget


@pytest.fixture
def target() -> AgentDojoTarget:
    return AgentDojoTarget(
        pipeline_model="openai/gpt-4o-2024-05-13",
        api_key="sk-dummy",
    )


# ----- Property surfaces -----


def test_security_domain_is_full_forest(target: AgentDojoTarget) -> None:
    assert target.security_domain is DOMAIN


def test_config_specs_exposed(target: AgentDojoTarget) -> None:
    names = [s.name for s in target.config_specs]
    assert names == [s.name for s in CONFIG_SPECS]


def test_query_specs_exposed(target: AgentDojoTarget) -> None:
    names = [s.name for s in target.query_specs]
    assert names == [s.name for s in QUERY_SPECS]


def test_controllables_exposed(target: AgentDojoTarget) -> None:
    names = [c.name for c in target.get_controllables()]
    assert names == [c.name for c in CONTROLLABLES]


def test_observables_exposed_with_pre_run_content(target: AgentDojoTarget) -> None:
    """Static observables have non-None content (model id, catalog).

    The system prompt is NOT a static observable: it is carried exactly
    once, on the Phase-1 system-prompt ControllablePreCallEvent.  The
    composite environment is NOT exposed as an observable; its full state
    is available post-run to the scorer via the query specs."""
    observables = target.get_observables()
    names = {o.observable.name for o in observables}
    assert names == {
        "model_identity",
        "detailed_system_specification",
        "tool_catalog_listing",
    }
    by_name = {o.observable.name: o for o in observables}
    assert by_name["model_identity"].content == "openai/gpt-4o-2024-05-13"
    assert isinstance(by_name["tool_catalog_listing"].content, list)
    # The leaked system-spec brief is non-empty free text covering the four items.
    spec = by_name["detailed_system_specification"].content
    assert isinstance(spec, str)
    assert "## 1. Purpose" in spec
    assert "## 4. Code and Hardcoded Prompts" in spec


# ----- set_config dispatch -----


def test_set_config_system_prompt(target: AgentDojoTarget) -> None:
    target.set_config("system_prompt", "be very helpful")
    # The prompt surfaces only on the Phase-1 controllable event at run
    # time, so verify the stored value directly.
    assert target._system_prompt == "be very helpful"


def test_set_config_user_prompt(target: AgentDojoTarget) -> None:
    target.set_config("user_prompt", "What is my balance?")
    # User prompt is internal-only until run() consumes it; ensure it's
    # stored via the public set_config contract.
    assert target._user_prompt == "What is my balance?"


def test_set_config_pipeline_model_rejected(target: AgentDojoTarget) -> None:
    """pipeline_model is a build-time construction choice, not a per-run
    config slot, so set_config must reject it."""
    with pytest.raises(ValueError, match="build-time"):
        target.set_config("pipeline_model", "openai/whatever")


def test_set_config_seed_override(target: AgentDojoTarget) -> None:
    overlay = json.dumps({"bank_account": {"balance": 0.0}})
    target.set_config("seed_yaml_override__banking", overlay)
    # The composite env is no longer exposed as an observable; the overlay
    # is applied at run-start by _build_seed_env_with_overrides().  Verify
    # the overlaid value lands on the freshly built per-run env.
    env = target._build_seed_env_with_overrides()
    assert env.banking.bank_account.balance == 0.0


def test_set_config_unknown_slot_raises(target: AgentDojoTarget) -> None:
    with pytest.raises(ValueError, match="Unknown config slot"):
        target.set_config("nonsense", "x")


# ----- query dispatch -----


def test_query_unknown_slot_raises(target: AgentDojoTarget) -> None:
    with pytest.raises(ValueError, match="Unknown query slot"):
        target.query("nonsense")


def test_query_returns_string_for_each_slot(target: AgentDojoTarget) -> None:
    """Pre-run queries return JSON-encoded empty/default values."""
    assert target.query("last_response") == ""
    assert json.loads(target.query("function_call_trace")) == []
    assert json.loads(target.query("pre_environment_snapshot")) == {}
    assert json.loads(target.query("post_environment_snapshot")) == {}
    assert json.loads(target.query("conversation_history")) == []
    assert json.loads(target.query("tool_catalog_snapshot")) == []
    assert json.loads(target.query("write_calls_made")) == []


# ----- reset_ephemeral_state -----


@pytest.mark.asyncio
async def test_reset_ephemeral_state_resets_per_run_state(
    target: AgentDojoTarget,
) -> None:
    """reset_ephemeral_state() must zero per-run state but preserve config slots."""
    target.set_config("user_prompt", "hello")
    target._last_response = "leftover"
    target._function_call_trace.append(  # type: ignore[arg-type]
        type(
            "FC",
            (),
            {"function": "x", "args": {}, "id": None, "placeholder_args": None},
        )()
    )
    await target.reset_ephemeral_state()
    assert target.query("last_response") == ""
    assert json.loads(target.query("function_call_trace")) == []
    # Config slot was NOT reset.
    assert target._user_prompt == "hello"


@pytest.mark.asyncio
async def test_teardown_is_noop(target: AgentDojoTarget) -> None:
    """teardown() should never raise even when no run has happened."""
    await target.teardown()
