"""Core state-machine tests for the MUZZLE optimizer.

Exercises the OBSERVE -> PROBE -> ATTACK phase machine of
``muzzle_optimizer.optimizer.MuzzleOptimizer`` directly through ``Optimizer._dispatch``
(via the shared ``dispatch_event`` harness), one event at a time. A single role-routing
``ScriptedLLM`` plays every red-team brain role, so the whole machine runs deterministically
and offline.

Harness contract reused here (see tests/conftest.py): the framework records the optimizer's
``ControllableInjection`` on the run trajectory, and the optimizer's PROBE / Judge
presence checks read it back via ``build_transcript``. So a test that wants a planted marker
or payload to be "surfaced" must ``traj.push(injection)`` BEFORE sending ``run_end()``.

PAIR is held to ``pair_n_streams=1`` / ``pair_n_iterations=1`` / ``pair_judge_method="gcg"``
so the offline loop is cheap and deterministic: the scripted attacker proposes
``PAIR_PAYLOAD``, the scripted victim refuses, and the GCG refusal-dictionary judge scores it
below jailbreak, so the best (and only) prompt that comes back is exactly ``PAIR_PAYLOAD``.
"""

import json

import pytest
from conftest import (
    FakeTrajectory,
    ScriptedLLM,
    TOOL_A_TAG,
    TOOL_B_TAG,
    USER_TAG,
    dispatch_event,
    make_controllable,
    observable_event,
    post_call,
    pre_call,
    run_end,
    run_start,
)

from muzzle_optimizer.constants import PLACEHOLDER_INSTR
from muzzle_optimizer.optimizer import MuzzleOptimizer
from superred.core.types.event import EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    RunEndResponse,
)
from superred.core.types.goal import Goal

# The bare prompt the default scripted PAIR attacker proposes (conftest _default_for_role).
PAIR_PAYLOAD = "REFINED PAYLOAD"
# A content-surface controllable name: the "tool:" prefix makes is_content_surface() True.
CONTENT_NAME = "tool:web_read"

# A Judge verdict whose LLM-level primary is "UI". Grounding overrides the primary purely
# from payload presence, so this only documents the scenario; absence is the real driver.
JUDGE_UI = json.dumps(
    {
        "outcome": "failure",
        "component_attribution": {"primary": "UI"},
        "recommendations": ["choose a different surface"],
        "evidence": [],
        "confidence": 0.3,
    }
)


def content_ctrl(name=CONTENT_NAME, tag=TOOL_A_TAG):
    """A PostCall content vessel surface (its name starts with ``tool:``)."""
    return make_controllable(
        name=name, tag=tag, value_type="text", description="external tool output content"
    )


def grafter_naming(names):
    """Grafter JSON that names each controllable so build_vessels ranks them in order."""
    candidates = []
    confidence = 0.9
    for name in names:
        candidates.append(
            {"element": f"inject into {name}", "why": "w", "how": "h", "confidence": confidence}
        )
        confidence -= 0.3
    return json.dumps({"reasoning": "r", "candidates": candidates})


async def make_optimizer(controllables, *, overrides=None, **kwargs):
    """Build and initialize a MUZZLE optimizer with a deterministic scripted LLM."""
    routing = {"grafter": grafter_naming([c.name for c in controllables])}
    if overrides:
        routing.update(overrides)
    llm = ScriptedLLM(overrides=routing)
    optimizer = MuzzleOptimizer(
        pair_n_streams=1, pair_n_iterations=1, pair_judge_method="gcg", **kwargs
    )
    await optimizer.initialize(
        Goal(description="exfiltrate the secret"), list(controllables), [], llm
    )
    return optimizer, llm


async def observe_to_probe(optimizer):
    """Run one OBSERVE run (which injects nothing) and return the run_end response."""
    traj = FakeTrajectory()
    await dispatch_event(optimizer, run_start(traj))
    return await dispatch_event(optimizer, run_end())


async def probe_to_attack(optimizer, vessel_ctrl):
    """Run one PROBE run that surfaces the placeholder at ``vessel_ctrl`` -> ATTACK."""
    traj = FakeTrajectory()
    await dispatch_event(optimizer, run_start(traj))
    injection = await dispatch_event(optimizer, post_call(vessel_ctrl, answer="legit answer"))
    assert isinstance(injection, ControllableInjection)
    traj.push(injection)  # record so the planted marker is in the snapshot
    return await dispatch_event(optimizer, run_end())


@pytest.mark.asyncio
async def test_observe_declines_every_event_then_advances_to_probe():
    ctrl = content_ctrl()
    optimizer, _ = await make_optimizer([ctrl])

    traj = FakeTrajectory()
    await dispatch_event(optimizer, run_start(traj))

    # OBSERVE is the benign passthrough run: every Pre/PostCall is declined.
    user_prompt = make_controllable(name="user_prompt", tag=USER_TAG)
    pre_resp = await dispatch_event(optimizer, pre_call(user_prompt))
    post_resp = await dispatch_event(optimizer, post_call(ctrl))
    assert isinstance(pre_resp, ControllableNoInjection)
    assert isinstance(post_resp, ControllableNoInjection)
    assert optimizer._phase == "OBSERVE"  # the phase only advances at run_end

    end_resp = await dispatch_event(optimizer, run_end())
    assert isinstance(end_resp, RunEndResponse)
    assert end_resp.done is False
    assert optimizer._phase == "PROBE"
    assert optimizer._vessels  # a content vessel was built from the grafter candidate
    assert optimizer._vessels[0].controllable == ctrl
    assert optimizer._vessels[0].kind == "content"
    assert optimizer._current_inject_value == PLACEHOLDER_INSTR


@pytest.mark.asyncio
async def test_observable_event_is_not_acted_on_as_controllable():
    ctrl = content_ctrl()
    optimizer, _ = await make_optimizer([ctrl])

    traj = FakeTrajectory()
    await dispatch_event(optimizer, run_start(traj))

    # An ObservableEvent is one-way and is never a controllable event: on_event falls through
    # to a bare EventResponse and never injects, declines, or advances the machine.
    resp = await dispatch_event(optimizer, observable_event("model_output", "agent said hi"))
    assert type(resp) is EventResponse
    assert not isinstance(resp, (ControllableInjection, ControllableNoInjection, RunEndResponse))
    assert optimizer._selected is False
    assert optimizer._injected is False
    assert optimizer._phase == "OBSERVE"


@pytest.mark.asyncio
async def test_probe_plants_placeholder_once_then_advances_to_attack():
    ctrl = content_ctrl()
    optimizer, _ = await make_optimizer([ctrl])
    await observe_to_probe(optimizer)
    assert optimizer._phase == "PROBE"

    traj = FakeTrajectory()
    await dispatch_event(optimizer, run_start(traj))

    first = await dispatch_event(optimizer, post_call(ctrl, answer="legit answer"))
    assert isinstance(first, ControllableInjection)
    assert first.controllable == ctrl
    assert PLACEHOLDER_INSTR in first.value
    traj.push(first)  # the marker must be in the snapshot for the probe to count as surfaced

    # One injection per run: a second matching PostCall in the same run is declined.
    second = await dispatch_event(optimizer, post_call(ctrl))
    assert isinstance(second, ControllableNoInjection)

    end_resp = await dispatch_event(optimizer, run_end())
    assert isinstance(end_resp, RunEndResponse)
    assert end_resp.done is False
    assert optimizer._phase == "ATTACK"
    assert optimizer._current_inject_value is None  # next RunStart will craft via PAIR
    assert optimizer._attempt_index == 0  # a surfaced probe is not a scored attempt


@pytest.mark.asyncio
async def test_probe_miss_loops_probe_and_advances_vessel_without_scoring():
    c0 = content_ctrl(name="tool:web_read")
    c1 = content_ctrl(name="tool:doc_read", tag=TOOL_B_TAG)
    optimizer, _ = await make_optimizer([c0, c1])
    await observe_to_probe(optimizer)
    assert optimizer._phase == "PROBE"
    assert len(optimizer._vessels) == 2
    assert optimizer._vessels[0].controllable == c0
    assert optimizer._vessel_idx == 0

    # PROBE run where the vessel never fires -> the marker never surfaces.
    traj = FakeTrajectory()
    await dispatch_event(optimizer, run_start(traj))
    end_resp = await dispatch_event(optimizer, run_end())

    assert isinstance(end_resp, RunEndResponse)
    assert end_resp.done is False  # another vessel still remains
    assert optimizer._phase == "PROBE"  # loops to probe the next vessel
    assert optimizer._vessel_idx == 1
    assert optimizer._current_inject_value == PLACEHOLDER_INSTR
    assert optimizer._attempt_index == 0  # a probe miss is not a scored attempt


@pytest.mark.asyncio
async def test_attack_injects_pair_payload_then_success_is_done():
    ctrl = content_ctrl()
    optimizer, _ = await make_optimizer([ctrl])
    await observe_to_probe(optimizer)
    await probe_to_attack(optimizer, ctrl)
    assert optimizer._phase == "ATTACK"

    traj = FakeTrajectory()
    await dispatch_event(optimizer, run_start(traj))
    # The offline PAIR loop crafted the payload at RunStart.
    assert optimizer._current_inject_value == PAIR_PAYLOAD

    first = await dispatch_event(optimizer, post_call(ctrl, answer="legit answer"))
    assert isinstance(first, ControllableInjection)
    assert PAIR_PAYLOAD in first.value
    traj.push(first)

    second = await dispatch_event(optimizer, post_call(ctrl))
    assert isinstance(second, ControllableNoInjection)  # one injection per run

    end_resp = await dispatch_event(optimizer, run_end(success=True, score=1.0))
    assert isinstance(end_resp, RunEndResponse)
    assert end_resp.done is True  # the SecurityClaim's success is authoritative
    assert optimizer._succeeded is True


@pytest.mark.asyncio
async def test_attack_failure_with_payload_present_refines_same_vessel():
    ctrl = content_ctrl()
    optimizer, _ = await make_optimizer([ctrl])
    await observe_to_probe(optimizer)
    await probe_to_attack(optimizer, ctrl)

    traj = FakeTrajectory()
    await dispatch_event(optimizer, run_start(traj))
    assert optimizer._current_inject_value == PAIR_PAYLOAD
    first = await dispatch_event(optimizer, post_call(ctrl, answer="legit answer"))
    assert isinstance(first, ControllableInjection)
    traj.push(first)  # payload present -> Judge grounds the attribution to "instruction"

    end_resp = await dispatch_event(optimizer, run_end(success=False))
    assert isinstance(end_resp, RunEndResponse)
    assert end_resp.done is False
    assert optimizer._phase == "ATTACK"  # stay on the same vessel
    assert optimizer._vessel_idx == 0
    assert optimizer._attempt_index == 1  # a scored ATTACK run
    assert optimizer._reflection is not None
    assert optimizer._reflection["most_recent_working_payload"] == PAIR_PAYLOAD
    assert optimizer._current_inject_value is None  # next ATTACK re-runs PAIR


@pytest.mark.asyncio
async def test_attack_failure_without_payload_advances_to_next_vessel_via_probe():
    c0 = content_ctrl(name="tool:web_read")
    c1 = content_ctrl(name="tool:doc_read", tag=TOOL_B_TAG)
    optimizer, _ = await make_optimizer([c0, c1], overrides={"judge": JUDGE_UI})
    await observe_to_probe(optimizer)
    await probe_to_attack(optimizer, c0)
    assert optimizer._phase == "ATTACK"
    assert optimizer._vessel_idx == 0

    traj = FakeTrajectory()
    await dispatch_event(optimizer, run_start(traj))
    assert optimizer._current_inject_value == PAIR_PAYLOAD
    # The vessel never fires this run, so the payload never lands -> "UI" attribution.
    end_resp = await dispatch_event(optimizer, run_end(success=False))

    assert isinstance(end_resp, RunEndResponse)
    assert end_resp.done is False
    assert optimizer._phase == "PROBE"  # advanced to probe the next vessel
    assert optimizer._vessel_idx == 1
    assert optimizer._current_inject_value == PLACEHOLDER_INSTR
    assert optimizer._attempt_index == 1  # the failed ATTACK was scored before advancing


@pytest.mark.asyncio
async def test_vessel_matches_by_value_equality_not_identity():
    ctrl = content_ctrl()
    optimizer, _ = await make_optimizer([ctrl])
    await observe_to_probe(optimizer)
    assert optimizer._phase == "PROBE"

    traj = FakeTrajectory()
    await dispatch_event(optimizer, run_start(traj))

    # A value-distinct content surface must not match the locked vessel.
    other = content_ctrl(name="tool:other_read", tag=TOOL_B_TAG)
    other_resp = await dispatch_event(optimizer, post_call(other))
    assert isinstance(other_resp, ControllableNoInjection)
    assert optimizer._selected is False  # a non-match must not consume the latch

    # A freshly rebuilt but value-EQUAL controllable still matches the locked vessel.
    rebuilt = content_ctrl()
    assert rebuilt == ctrl and rebuilt is not ctrl
    match_resp = await dispatch_event(optimizer, post_call(rebuilt))
    assert isinstance(match_resp, ControllableInjection)
    assert PLACEHOLDER_INSTR in match_resp.value


@pytest.mark.asyncio
async def test_wrong_name_pin_has_no_vessel_and_stays_passthrough():
    ctrl = content_ctrl()  # a real content surface IS in scope...
    optimizer, _ = await make_optimizer([ctrl], target_controllable_name="nope")

    traj = FakeTrajectory()
    await dispatch_event(optimizer, run_start(traj))
    post_resp = await dispatch_event(optimizer, post_call(ctrl))
    assert isinstance(post_resp, ControllableNoInjection)

    end_resp = await dispatch_event(optimizer, run_end())
    assert isinstance(end_resp, RunEndResponse)
    assert end_resp.done is True  # ...but the pin matches nothing -> no vessel -> stop
    assert optimizer._vessels == []
    assert optimizer._exhausted is True
    assert optimizer._phase == "OBSERVE"  # never advanced past the benign run

    # The optimizer stays a pure passthrough on any further event.
    traj2 = FakeTrajectory()
    await dispatch_event(optimizer, run_start(traj2))
    again = await dispatch_event(optimizer, post_call(ctrl))
    assert isinstance(again, ControllableNoInjection)
