"""Unit tests for InspectAgentTarget (offline: fake model, no network)."""

from __future__ import annotations

import json

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
from inspect_ai.tool import Tool, ToolCall, tool

from inspect_agent_target import (
    CONTROLLABLES,
    InspectAgentTarget,
    SYSTEM_PROMPT_CTRL,
    TOOLS_TAG,
    USER_PROMPT_CTRL,
)
from inspect_agent_target import target as target_mod
from inspect_agent_target.rollout import run_rollout, static_tools_provider
from inspect_agent_target.target import (
    _function_call_trace,
    _last_assistant_text,
    _parse_tool_names,
)
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
)
from superred.core.types.security_domain import SecurityDomainTag


def _make_target() -> InspectAgentTarget:
    return InspectAgentTarget(
        model="openai/gpt-4o-mini",
        tool_resolver=lambda name: (_ for _ in ()).throw(AssertionError("no resolve")),
        api_base="b",
        api_key="k",
    )


class _Out:
    def __init__(self, message: ChatMessageAssistant) -> None:
        self.message = message


class _FakeModel:
    """Minimal stand-in for inspect Model: yields preset assistant messages."""

    def __init__(self, messages: list[ChatMessageAssistant]) -> None:
        self._messages = messages
        self._i = 0

    async def generate(self, _input, tools=None, tool_choice=None, **_kw):  # noqa: ANN001
        msg = self._messages[self._i]
        self._i += 1
        return _Out(msg)


@tool
def _echo() -> Tool:
    async def execute(text: str) -> str:
        """Echo the text back.

        Args:
            text: the text to echo
        """
        return f"REAL:{text}"

    return execute


# --- specs / domain / controllables ---------------------------------------


def test_config_specs_names() -> None:
    t = _make_target()
    names = {s.name for s in t.config_specs}
    # No 'model' slot: the model is fixed at construction, not Task-configurable.
    assert names == {
        "system_prompt", "user_prompt", "tool_names", "tool_choice", "message_limit"
    }


def test_query_specs_names() -> None:
    t = _make_target()
    assert {s.name for s in t.query_specs} == {
        "last_response", "conversation_history", "function_call_trace"
    }


def test_controllables_and_domain() -> None:
    t = _make_target()
    # No tools configured -> just the fixed controllables.
    assert [c.name for c in t.get_controllables()] == [
        "system_prompt",
        "user_prompt",
        "tool_catalog_register",
        "tool_catalog_replace",
        "tool_catalog_unregister",
        "tool_catalog_rewrite_doc",
    ]
    assert SYSTEM_PROMPT_CTRL in CONTROLLABLES and USER_PROMPT_CTRL in CONTROLLABLES
    # Default domain (no tool scopes): the three fixed roots.
    assert {r.name for r in t.security_domain.roots} == {"system", "user", "tools"}


def test_per_tool_output_controllables_unmapped_fall_back_to_tools_root() -> None:
    t = _make_target()
    t.set_config("tool_names", '["alpha", "beta"]')
    names = [c.name for c in t.get_controllables()]
    # One tool:<name> output controllable per configured tool, after the fixed set.
    assert names[-2:] == ["tool:alpha", "tool:beta"]
    tool_ctrls = [c for c in t.get_controllables() if c.name.startswith("tool:")]
    assert all(c.security_domain is TOOLS_TAG for c in tool_ctrls)


def test_tool_scopes_scope_per_tool_controllables_and_domain() -> None:
    web = SecurityDomainTag("web_test", parent=TOOLS_TAG)
    t = InspectAgentTarget(
        model="m",
        tool_resolver=lambda n: None,  # type: ignore[arg-type,return-value]
        api_base="b",
        api_key="k",
        tool_scopes={"alpha": web},
    )
    t.set_config("tool_names", '["alpha", "beta"]')
    by_name = {c.name: c for c in t.get_controllables()}
    assert by_name["tool:alpha"].security_domain is web  # mapped to its boundary
    assert by_name["tool:beta"].security_domain is TOOLS_TAG  # unmapped fallback
    # The mapped boundary tag is part of the assembled domain.
    assert "web_test" in t.security_domain._tags


def test_observables_reflect_config() -> None:
    t = _make_target()
    t.set_config("system_prompt", "SP")
    t.set_config("message_limit", "7")
    obs = {o.observable.name: o.content for o in t.get_observables()}
    assert obs["model_identity"] == "openai/gpt-4o-mini"
    assert obs["system_prompt"] == "SP"
    assert obs["message_limit"] == "7"  # static observable, not on the trajectory
    assert obs["tool_catalog_listing"] == []  # no tools configured here


# --- set_config -----------------------------------------------------------


def test_set_config_dispatch() -> None:
    t = _make_target()
    t.set_config("system_prompt", "SP")
    t.set_config("user_prompt", "UP")
    t.set_config("tool_names", '["a", "b"]')
    t.set_config("tool_choice", "none")
    t.set_config("message_limit", "5")
    assert t._system_prompt == "SP"
    assert t._user_prompt == "UP"
    assert t._tool_names == ["a", "b"]
    assert t._tool_choice == "none"
    assert t._message_limit == 5


def test_set_config_rejects_model_slot() -> None:
    # 'model' is no longer a config slot: it is construction-only.
    t = _make_target()
    with pytest.raises(ValueError, match="Unknown config slot"):
        t.set_config("model", "openai/x")


def test_set_config_unknown_slot() -> None:
    t = _make_target()
    with pytest.raises(ValueError, match="Unknown config slot"):
        t.set_config("nope", "x")


def test_set_config_bad_tool_choice() -> None:
    t = _make_target()
    with pytest.raises(ValueError, match="tool_choice"):
        t.set_config("tool_choice", "banana")


def test_message_limit_empty_uses_default() -> None:
    t = _make_target()
    t.set_config("message_limit", "")
    assert t._message_limit == t._default_message_limit


def test_parse_tool_names() -> None:
    assert _parse_tool_names("") == []
    assert _parse_tool_names('["x","y"]') == ["x", "y"]
    with pytest.raises(ValueError):
        _parse_tool_names('{"not": "a list"}')
    with pytest.raises(ValueError):
        _parse_tool_names("[1, 2]")


# --- query ----------------------------------------------------------------


def test_query_unknown_slot() -> None:
    t = _make_target()
    with pytest.raises(ValueError, match="Unknown query slot"):
        t.query("nope")


# --- helpers --------------------------------------------------------------


def test_function_call_trace_and_last_text() -> None:
    tc = ToolCall(id="1", function="foo", arguments={"a": 1})
    msgs = [
        ChatMessageUser(content="hi"),
        ChatMessageAssistant(content="", tool_calls=[tc]),
        ChatMessageTool(content="ok", tool_call_id="1", function="foo"),
        ChatMessageAssistant(content="done"),
    ]
    trace = _function_call_trace(msgs)
    assert trace == [{"function": "foo", "arguments": {"a": 1}, "id": "1"}]
    assert _last_assistant_text(msgs) == "done"
    assert _last_assistant_text([ChatMessageUser(content="x")]) == ""


# --- rollout (no-tool path) -----------------------------------------------


@pytest.mark.asyncio
async def test_rollout_no_tools_breaks_immediately() -> None:
    model = _FakeModel([ChatMessageAssistant(content="final")])
    msgs = await run_rollout(
        model,  # type: ignore[arg-type]
        system_prompt="SP",
        user_prompt="UP",
        tools_provider=static_tools_provider([]),
        tool_choice="auto",
        message_limit=20,
    )
    assert [m.role for m in msgs] == ["system", "user", "assistant"]
    assert _last_assistant_text(msgs) == "final"


@pytest.mark.asyncio
async def test_rollout_no_system_prompt() -> None:
    model = _FakeModel([ChatMessageAssistant(content="hi")])
    msgs = await run_rollout(
        model,  # type: ignore[arg-type]
        system_prompt="",
        user_prompt="UP",
        tools_provider=static_tools_provider([]),
        tool_choice="auto",
        message_limit=20,
    )
    assert [m.role for m in msgs] == ["user", "assistant"]


# --- run() with monkeypatched model + passthrough events ------------------


@pytest.mark.asyncio
async def test_run_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeModel([ChatMessageAssistant(content="answer")])
    monkeypatch.setattr(target_mod, "get_model", lambda *a, **k: fake)

    t = InspectAgentTarget(
        model="openai/gpt-4o-mini", tool_resolver=lambda n: None, api_base="b", api_key="k"  # type: ignore[arg-type,return-value]
    )
    t.set_config("system_prompt", "SP")
    t.set_config("user_prompt", "UP")
    t.set_config("tool_names", "[]")

    emitted: list = []

    async def send_event(event):  # noqa: ANN001
        return ControllableNoInjection(event=event, controllable=event.controllable)

    def emit(event):  # noqa: ANN001
        emitted.append(event)

    await t.run(emit, send_event)
    assert t.query("last_response") == "answer"
    assert json.loads(t.query("function_call_trace")) == []
    assert len(json.loads(t.query("conversation_history"))) == 3
    # 3 chat-message observables on the trajectory; the tool catalogue is a
    # STATIC observable (get_observables), not emitted on the trajectory.
    msg_obs = [e for e in emitted if e.observable.name.startswith("agent_trace_message_")]
    catalog_obs = [e for e in emitted if e.observable.name == "tool_catalog_listing"]
    assert len(msg_obs) == 3
    assert len(catalog_obs) == 0
    await t.reset_ephemeral_state()
    assert t.messages == []


@pytest.mark.asyncio
async def test_generate_config_matches_agentharm_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The real GenerateConfig reaches get_model with AgentHarm's generation
    # defaults; max_connections stays unset (the factory owns concurrency).
    captured: dict = {}

    def fake_get_model(model_id, *, base_url=None, api_key=None, config=None):  # noqa: ANN001
        captured["model_id"] = model_id
        captured["config"] = config
        return _FakeModel([ChatMessageAssistant(content="x")])

    monkeypatch.setattr(target_mod, "get_model", fake_get_model)

    t = InspectAgentTarget(
        model="openai/gpt-4o-mini", tool_resolver=lambda n: None, api_base="b", api_key="k"  # type: ignore[arg-type,return-value]
    )
    t.set_config("user_prompt", "UP")
    t.set_config("tool_names", "[]")

    async def send_event(event):  # noqa: ANN001
        return ControllableNoInjection(event=event, controllable=event.controllable)

    await t.run(lambda e: None, send_event)
    cfg = captured["config"]
    assert captured["model_id"] == "openai/gpt-4o-mini"
    assert cfg.temperature == 0.0
    assert cfg.max_tokens == 4096
    assert cfg.seed == 0
    assert cfg.max_retries == 3
    assert cfg.max_connections is None


@pytest.mark.asyncio
async def test_run_user_prompt_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}
    fake = _FakeModel([ChatMessageAssistant(content="x")])

    def fake_get_model(*_a, **_k):  # noqa: ANN001, ANN202
        return fake

    monkeypatch.setattr(target_mod, "get_model", fake_get_model)

    # Capture the user prompt the rollout actually uses.
    orig_run_rollout = target_mod.run_rollout

    async def spy_rollout(model, *, system_prompt, user_prompt, tools_provider, tool_choice, message_limit, on_tool_results=None):  # noqa: ANN001
        captured["user"] = user_prompt
        captured["system"] = system_prompt
        return await orig_run_rollout(
            model, system_prompt=system_prompt, user_prompt=user_prompt,
            tools_provider=tools_provider, tool_choice=tool_choice,
            message_limit=message_limit, on_tool_results=on_tool_results,
        )

    monkeypatch.setattr(target_mod, "run_rollout", spy_rollout)

    t = InspectAgentTarget(model="m", tool_resolver=lambda n: None, api_base="b", api_key="k")  # type: ignore[arg-type,return-value]
    t.set_config("user_prompt", "benign")
    t.set_config("system_prompt", "sys")
    t.set_config("tool_names", "[]")

    async def send_event(event):  # noqa: ANN001
        if event.controllable is USER_PROMPT_CTRL:
            return ControllableInjection(
                event=event, controllable=event.controllable, value="JAILBROKEN"
            )
        return ControllableNoInjection(event=event, controllable=event.controllable)

    await t.run(lambda e: None, send_event)
    assert captured["user"] == "JAILBROKEN"
    assert captured["system"] == "sys"


@pytest.mark.asyncio
async def test_run_catalog_register_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeModel([ChatMessageAssistant(content="ok")])
    monkeypatch.setattr(target_mod, "get_model", lambda *a, **k: fake)

    t = InspectAgentTarget(model="m", tool_resolver=lambda n: None, api_base="b", api_key="k")  # type: ignore[arg-type,return-value]
    t.set_config("user_prompt", "UP")
    t.set_config("tool_names", "[]")

    payload = json.dumps({"name": "injected_tool", "description": "evil", "fake_return": "x"})

    async def send_event(event):  # noqa: ANN001
        ctrl = getattr(event, "controllable", None)
        if ctrl is not None and ctrl.name == "tool_catalog_register":
            return ControllableInjection(event=event, controllable=ctrl, value=payload)
        return ControllableNoInjection(event=event, controllable=event.controllable)

    await t.run(lambda e: None, send_event)
    assert t._catalog is not None
    assert "injected_tool" in t._catalog.names()


@pytest.mark.asyncio
async def test_run_tool_output_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    # turn 1 calls _echo (executed for real), turn 2 answers.
    fake = _FakeModel(
        [
            ChatMessageAssistant(
                content="",
                tool_calls=[ToolCall(id="1", function="_echo", arguments={"text": "hi"})],
            ),
            ChatMessageAssistant(content="done"),
        ]
    )
    monkeypatch.setattr(target_mod, "get_model", lambda *a, **k: fake)

    t = InspectAgentTarget(model="m", tool_resolver=lambda n: _echo(), api_base="b", api_key="k")
    t.set_config("user_prompt", "UP")
    t.set_config("tool_names", '["_echo"]')

    async def send_event(event):  # noqa: ANN001
        ctrl = getattr(event, "controllable", None)
        if ctrl is not None and ctrl.name == "tool:_echo":
            return ControllableInjection(
                event=event, controllable=ctrl, value="INJECTED-OUTPUT"
            )
        return ControllableNoInjection(event=event, controllable=event.controllable)

    await t.run(lambda e: None, send_event)
    hist = json.loads(t.query("conversation_history"))
    tool_msgs = [m for m in hist if m.get("role") == "tool"]
    assert tool_msgs, "no tool result message in trace"
    content = str(tool_msgs[0].get("content"))
    assert "INJECTED-OUTPUT" in content  # attacker replaced the tool's real return
    assert "REAL:hi" not in content
