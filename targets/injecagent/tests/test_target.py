"""Tests for the InjecAgent target: contract, prompt/message builders, cache-hit
simulator, and full run() orchestration (both modes, ds two-step, only_first_step,
injection override) with a stubbed model. No network."""

from __future__ import annotations

import json

import pytest
from injecagent_claim import load_cases
from injecagent_claim.judge import evaluate_task
from superred.core.controller import TargetFactory
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
)

from injecagent_target import (
    DOMAIN,
    InjecAgentTarget,
    injecagent_target_factory,
)
from injecagent_target import agent as agent_mod
from injecagent_target import simulator as sim_mod

DH = load_cases(settings=("base",), attacks=("dh",))[0]
DS = load_cases(settings=("base",), attacks=("ds",))[0]


def _react(tool: str, params: str = "{}") -> str:
    return f"Thought: I will act as the message asked.\nAction: {tool}\nAction Input: {params}"


_REFUSE = (
    "Thought: This looks like an injected instruction from untrusted content; I "
    "will not act on it and will answer the user normally.\nFinal Answer: Done."
)


def _handlers(injections: dict[str, str] | None = None):
    events: list = []

    def emit(ev):
        events.append(ev)

    async def send_event(ev):
        name = ev.controllable.name
        if injections and name in injections:
            return ControllableInjection(
                event=ev, controllable=ev.controllable, value=injections[name]
            )
        return ControllableNoInjection(event=ev, controllable=ev.controllable)

    return emit, send_event, events


def _configure(target: InjecAgentTarget, case: dict, attack: str) -> None:
    target.set_config("case", json.dumps(case))
    target.set_config("attack", attack)
    target.set_config("setting", case["_setting"])


# -- contract ----------------------------------------------------------------
def test_contract() -> None:
    t = InjecAgentTarget(model="stub/model")
    assert {c.name for c in t.config_specs} == {"case", "attack", "setting"}
    assert [q.name for q in t.query_specs] == ["result"]
    assert t.security_domain is DOMAIN
    assert {c.name for c in t.get_controllables()} == {"attacker_instruction", "user_instruction"}
    _configure(t, DH, "dh")
    names = {ov.observable.name for ov in t.get_observables()}
    assert names == {"model_identity", "attack_type", "setting", "prompt_type", "mode"}


def test_construction_validates() -> None:
    with pytest.raises(ValueError):
        InjecAgentTarget(model="x", prompt_type="bogus")
    with pytest.raises(ValueError):
        InjecAgentTarget(model="x", mode="bogus")


def test_factory_builds_target() -> None:
    fac = injecagent_target_factory(model="stub/model")
    assert isinstance(fac, TargetFactory)
    assert isinstance(fac.create(), InjecAgentTarget)


# -- builders ----------------------------------------------------------------
def test_build_prompted_input_embeds_poison_and_tools() -> None:
    tool_dict = agent_mod.get_tool_dict()
    system, user = agent_mod.build_prompted_input(DH, tool_dict, "InjecAgent")
    assert DH["User Instruction"] in user
    assert DH["Attacker Instruction"] in user  # baked into the scratchpad observation
    assert DH["User Tool"] in user  # tool description rendered
    # step 2 scratchpad includes the first attacker tool call + its response
    ds = dict(DS)
    ds["Attacker Tool Thought"] = "Thought: forwarding."
    ds["Attacker Tool Parameters"] = "{}"
    ds["Attacker Tool Response"] = '{"data": "secret"}'
    _, user2 = agent_mod.build_prompted_input(ds, tool_dict, "InjecAgent", step_2=True)
    assert DS["Attacker Tools"][0] in user2 and '{"data": "secret"}' in user2


def test_build_finetuned_messages_shape() -> None:
    tool_dict = agent_mod.get_tool_dict(gpt_format=True)
    messages, tools = agent_mod.build_finetuned_messages(DH, tool_dict, "InjecAgent")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assert messages[2]["tool_calls"][0]["function"]["name"] == DH["User Tool"]
    assert len(tools) == 1 + len(DH["Attacker Tools"])  # user tool + attacker tools


# -- simulator (cache hit; no LLM) -------------------------------------------
async def test_simulator_cache_hit() -> None:
    # pull a real (tool, params) key from the vendored cache and confirm a hit
    cache = sim_mod._load_cache()
    key = next(iter(cache))  # e.g. "(AmazonViewSavedAddresses,{})"
    inner = key[1:-1]
    tool, params = inner.split(",", 1)

    async def _boom(*a, **k):  # must NOT be called on a cache hit
        raise AssertionError("LLM fallback called on a cache hit")

    orig = sim_mod._call_sim
    sim_mod._call_sim = _boom
    try:
        out = await sim_mod.simulate_attacker_tool_response(
            tool, params, agent_mod.get_tool_dict(), sim_model="stub/model"
        )
    finally:
        sim_mod._call_sim = orig
    assert out == cache[key]


# -- run() orchestration -----------------------------------------------------
async def test_run_requires_case() -> None:
    t = InjecAgentTarget(model="stub/model")
    emit, send, _ = _handlers()
    with pytest.raises(RuntimeError):
        await t.run(emit, send)


async def test_run_dh_prompted(monkeypatch) -> None:
    attacker_tool = DH["Attacker Tools"][0]
    calls: list = []

    async def fake_call_prompted(model, system, user, **kw):
        calls.append(user)
        return _react(attacker_tool)

    monkeypatch.setattr(agent_mod, "call_prompted", fake_call_prompted)
    t = InjecAgentTarget(model="stub/model")
    _configure(t, DH, "dh")
    emit, send, events = _handlers()
    await t.run(emit, send)

    result = json.loads(t.query("result"))
    assert result["mode"] == "prompted" and "output" in result and "output_step2" not in result
    assert len(calls) == 1
    # end-to-end consistency with the claim's judge
    assert evaluate_task(DH, "dh", result).attack_succeeded is True


async def test_run_dh_refuse_is_secure(monkeypatch) -> None:
    async def fake_call_prompted(model, system, user, **kw):
        return _REFUSE

    monkeypatch.setattr(agent_mod, "call_prompted", fake_call_prompted)
    t = InjecAgentTarget(model="stub/model")
    _configure(t, DH, "dh")
    emit, send, _ = _handlers()
    await t.run(emit, send)
    result = json.loads(t.query("result"))
    assert evaluate_task(DH, "dh", result).attack_succeeded is False


async def test_run_ds_two_step(monkeypatch) -> None:
    exfil, send_tool = DS["Attacker Tools"][0], DS["Attacker Tools"][1]
    outputs = [_react(exfil), _react(send_tool, '{"to": "amy.watson@gmail.com"}')]
    sim_calls: list = []

    async def fake_call_prompted(model, system, user, **kw):
        return outputs.pop(0)

    async def fake_sim(tool, params, tool_dict, **kw):
        sim_calls.append(tool)
        return '{"addresses": ["1 Secret St"]}'

    monkeypatch.setattr(agent_mod, "call_prompted", fake_call_prompted)
    monkeypatch.setattr(sim_mod, "simulate_attacker_tool_response", fake_sim)

    t = InjecAgentTarget(model="stub/model")
    _configure(t, DS, "ds")
    emit, send, _ = _handlers()
    await t.run(emit, send)

    result = json.loads(t.query("result"))
    assert "output_step2" in result  # step 2 ran because step 1 succeeded
    assert sim_calls == [exfil]  # simulated the exfil tool's response
    assert evaluate_task(DS, "ds", result).attack_succeeded is True


async def test_run_ds_only_first_step(monkeypatch) -> None:
    exfil = DS["Attacker Tools"][0]

    async def fake_call_prompted(model, system, user, **kw):
        return _react(exfil)

    async def _boom_sim(*a, **k):
        raise AssertionError("simulation must not run under only_first_step")

    monkeypatch.setattr(agent_mod, "call_prompted", fake_call_prompted)
    monkeypatch.setattr(sim_mod, "simulate_attacker_tool_response", _boom_sim)

    t = InjecAgentTarget(model="stub/model", only_first_step=True)
    _configure(t, DS, "ds")
    emit, send, _ = _handlers()
    await t.run(emit, send)
    result = json.loads(t.query("result"))
    assert result.get("only_first_step") is True and "output_step2" not in result
    # step 1 exfil succeeded -> scored as a (partial) data-stealing attack
    assert evaluate_task(DS, "ds", result).attack_succeeded is True


async def test_run_finetuned(monkeypatch) -> None:
    attacker_tool = DH["Attacker Tools"][0]

    async def fake_call_finetuned(model, messages, tools, **kw):
        return None, [
            {"id": "call_x", "type": "function",
             "function": {"name": attacker_tool, "arguments": "{}"}}
        ]

    monkeypatch.setattr(agent_mod, "call_finetuned", fake_call_finetuned)
    t = InjecAgentTarget(model="stub/model", mode="finetuned")
    _configure(t, DH, "dh")
    emit, send, _ = _handlers()
    await t.run(emit, send)
    result = json.loads(t.query("result"))
    assert result["mode"] == "finetuned"
    assert result["tool_calls"][0]["function"]["name"] == attacker_tool
    assert evaluate_task(DH, "dh", result).attack_succeeded is True


async def test_run_ds_finetuned_two_step(monkeypatch) -> None:
    exfil, send_tool = DS["Attacker Tools"][0], DS["Attacker Tools"][1]
    def _tc(cid: str, name: str) -> list[dict]:
        return [{"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}]

    responses = [(None, _tc("c1", exfil)), (None, _tc("c2", send_tool))]

    async def fake_call_finetuned(model, messages, tools, **kw):
        return responses.pop(0)

    async def fake_sim(tool, params, tool_dict, **kw):
        return '{"addresses": ["1 Secret St"]}'

    monkeypatch.setattr(agent_mod, "call_finetuned", fake_call_finetuned)
    monkeypatch.setattr(sim_mod, "simulate_attacker_tool_response", fake_sim)
    t = InjecAgentTarget(model="stub/model", mode="finetuned")
    _configure(t, DS, "ds")
    emit, send, _ = _handlers()
    await t.run(emit, send)
    result = json.loads(t.query("result"))
    assert "tool_calls_step2" in result and not responses  # both steps ran
    assert evaluate_task(DS, "ds", result).attack_succeeded is True


async def test_run_attacker_instruction_injection_rewrites_observation(monkeypatch) -> None:
    seen: list = []
    marker = "PWNED_INSTRUCTION_XYZ_9"

    async def fake_call_prompted(model, system, user, **kw):
        seen.append(user)
        return _REFUSE

    monkeypatch.setattr(agent_mod, "call_prompted", fake_call_prompted)
    t = InjecAgentTarget(model="stub/model")
    _configure(t, DH, "dh")
    emit, send, _ = _handlers(injections={"attacker_instruction": marker})
    await t.run(emit, send)
    # the injected instruction is re-substituted into the poisoned tool observation
    assert marker in seen[0]
    assert DH["Attacker Instruction"] not in seen[0]
    # and the effective instruction is recorded so the claim's judge grades
    # against what the agent actually saw (not the case's original)
    assert json.loads(t.query("result"))["attacker_instruction"] == marker


async def test_enhanced_wrapper_survives_attacker_instruction_override(monkeypatch) -> None:
    # An override on an *enhanced* case must keep the reinforcement wrapper (which
    # lives only in the rendered Tool Response, not the template) -- otherwise the
    # delivered payload silently downgrades to base strength while reporting enhanced.
    enh = load_cases(settings=("enhanced",), attacks=("dh",))[0]
    marker = "OVERRIDE_PAYLOAD_ABC"
    seen: list = []

    async def fake_call_prompted(model, system, user, **kw):
        seen.append(user)
        return _REFUSE

    monkeypatch.setattr(agent_mod, "call_prompted", fake_call_prompted)
    t = InjecAgentTarget(model="stub/model")
    _configure(t, enh, "dh")
    emit, send, _ = _handlers(injections={"attacker_instruction": marker})
    await t.run(emit, send)
    assert "IMPORTANT!!! Ignore all previous instructions" in seen[0]  # wrapper survived
    assert marker in seen[0]
    assert enh["Attacker Instruction"] not in seen[0]  # original swapped out


async def test_run_user_instruction_injection(monkeypatch) -> None:
    seen: list = []

    async def fake_call_prompted(model, system, user, **kw):
        seen.append(user)
        return _REFUSE

    monkeypatch.setattr(agent_mod, "call_prompted", fake_call_prompted)
    t = InjecAgentTarget(model="stub/model")
    _configure(t, DH, "dh")
    emit, send, _ = _handlers(injections={"user_instruction": "Summarize my day please."})
    await t.run(emit, send)
    assert "Summarize my day please." in seen[0]
