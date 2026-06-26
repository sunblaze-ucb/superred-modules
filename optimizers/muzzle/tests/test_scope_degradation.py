"""Vessel selection degrades gracefully as the in-scope surfaces shrink.

MUZZLE prefers high-salience CONTENT post-call surfaces (tool outputs, retrieved documents),
falls back to the user-prompt pre-call surface, and finally to the system prompt; with no
injectable surface at all it must reduce to the unattacked passthrough baseline. These tests
drive the real OBSERVE -> PROBE -> ATTACK loop one event at a time and assert which surface is
chosen, on which call polarity it injects, and that the SecurityClaim (not the optimizer) owns
the success verdict.

Driver note: the framework records the optimizer's ``ControllableInjection`` back onto the run
trajectory, and the PROBE's marker-presence check reads it back via ``build_transcript``. So a
test must ``traj.push(inject_response)`` before sending ``run_end`` for the planted marker /
payload to be visible in the snapshot.
"""

from __future__ import annotations

import json

import pytest

from muzzle_optimizer.constants import PLACEHOLDER_INSTR
from muzzle_optimizer.optimizer import MuzzleOptimizer
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    RunEndResponse,
)
from superred.core.types.goal import Goal

from conftest import (
    TOOL_A_TAG,
    USER_TAG,
    FakeTrajectory,
    ScriptedLLM,
    dispatch_event,
    make_controllable,
    post_call,
    pre_call,
    run_end,
    run_end_no_feedback,
    run_start,
)

# PAIR's default attacker proposal (conftest "pair_attacker" role): the bare refined payload
# the offline loop returns and the ATTACK run then plants on the chosen vessel.
_PAIR_PAYLOAD = "REFINED PAYLOAD"


def _grafter_naming(name: str) -> str:
    """A grafter response that names ``name`` so build_vessels ranks that content surface."""
    return json.dumps(
        {
            "reasoning": "r",
            "candidates": [
                {"element": f"the {name} result box", "why": "w", "how": "h", "confidence": 0.9}
            ],
        }
    )


async def _observe_run(optimizer: MuzzleOptimizer, ctrl: object, *, is_content: bool) -> object:
    """Drive one OBSERVE (benign) run: nothing is injected; vessels/instruction are built."""
    traj = FakeTrajectory()
    await dispatch_event(optimizer, run_start(traj))
    event = post_call(ctrl) if is_content else pre_call(ctrl)  # type: ignore[arg-type]
    traj.push(event)
    declined = await dispatch_event(optimizer, event)
    # OBSERVE always declines (passthrough), regardless of surface kind.
    assert isinstance(declined, ControllableNoInjection)
    traj.push(declined)
    return await dispatch_event(optimizer, run_end())


async def _active_run(
    optimizer: MuzzleOptimizer, ctrl: object, *, is_content: bool, end_event: object
) -> tuple[object, object, object]:
    """Drive one PROBE/ATTACK run and verify call-polarity of the chosen vessel.

    A content vessel injects on the POST-call (and declines a PRE-call decoy); a prompt vessel
    injects on the PRE-call (and declines a POST-call decoy). Returns
    ``(decoy_response, real_response, run_end_response)``.
    """
    traj = FakeTrajectory()
    await dispatch_event(optimizer, run_start(traj))
    decoy = pre_call(ctrl) if is_content else post_call(ctrl)  # type: ignore[arg-type]
    decoy_response = await dispatch_event(optimizer, decoy)
    real = post_call(ctrl) if is_content else pre_call(ctrl)  # type: ignore[arg-type]
    traj.push(real)
    real_response = await dispatch_event(optimizer, real)
    traj.push(real_response)  # record the injection so PROBE/Judge read it back
    end = await dispatch_event(optimizer, end_event)
    return decoy_response, real_response, end


@pytest.mark.asyncio
async def test_content_postcall_surface_is_chosen_and_injected_on_post() -> None:
    """(a) With only a content post-call surface, a content vessel is chosen and injected."""
    ctrl = make_controllable("tool:web", TOOL_A_TAG)
    llm = ScriptedLLM(overrides={"grafter": _grafter_naming("tool:web")})
    optimizer = MuzzleOptimizer(pair_judge_method="gcg", pair_n_streams=1, pair_n_iterations=1)
    await optimizer.initialize(Goal(description="leak the key"), [ctrl], [], llm)

    end_observe = await _observe_run(optimizer, ctrl, is_content=True)
    assert isinstance(end_observe, RunEndResponse)
    assert end_observe.done is False
    assert optimizer._phase == "PROBE"
    # The grafter-named content surface is the (only) vessel, carrying grafter confidence.
    assert [v.kind for v in optimizer._vessels] == ["content"]
    assert optimizer._vessels[0].controllable == ctrl
    assert optimizer._vessels[0].confidence == 0.9
    assert optimizer._current_inject_value == PLACEHOLDER_INSTR

    decoy, probe_inject, end_probe = await _active_run(
        optimizer, ctrl, is_content=True, end_event=run_end()
    )
    assert isinstance(decoy, ControllableNoInjection)  # pre-call decoy declined
    assert isinstance(probe_inject, ControllableInjection)  # post-call carries the marker
    assert PLACEHOLDER_INSTR in probe_inject.value
    assert optimizer._phase == "ATTACK"

    decoy2, attack_inject, end_attack = await _active_run(
        optimizer, ctrl, is_content=True, end_event=run_end(success=True)
    )
    assert isinstance(decoy2, ControllableNoInjection)
    assert isinstance(attack_inject, ControllableInjection)
    assert _PAIR_PAYLOAD in attack_inject.value  # the offline-PAIR payload was planted
    assert end_attack.done is True
    assert optimizer._succeeded is True


@pytest.mark.asyncio
async def test_user_prompt_fallback_injects_on_precall() -> None:
    """(b) Pre-call only: the user-prompt fallback vessel injects on a PRE-call, not a post."""
    ctrl = make_controllable("user_prompt", USER_TAG)
    optimizer = MuzzleOptimizer(pair_judge_method="gcg", pair_n_streams=1, pair_n_iterations=1)
    await optimizer.initialize(Goal(description="leak the key"), [ctrl], [], ScriptedLLM())

    end_observe = await _observe_run(optimizer, ctrl, is_content=False)
    assert end_observe.done is False
    # No content surface in scope -> the single vessel is the user-prompt fallback.
    assert [v.kind for v in optimizer._vessels] == ["user_prompt"]
    assert optimizer._vessels[0].controllable == ctrl
    assert optimizer._phase == "PROBE"

    decoy, probe_inject, end_probe = await _active_run(
        optimizer, ctrl, is_content=False, end_event=run_end()
    )
    assert isinstance(decoy, ControllableNoInjection)  # post-call decoy declined
    assert isinstance(probe_inject, ControllableInjection)  # pre-call carries the marker
    assert PLACEHOLDER_INSTR in probe_inject.value
    assert optimizer._phase == "ATTACK"

    _, attack_inject, end_attack = await _active_run(
        optimizer, ctrl, is_content=False, end_event=run_end(success=True)
    )
    assert isinstance(attack_inject, ControllableInjection)
    assert _PAIR_PAYLOAD in attack_inject.value
    assert end_attack.done is True


@pytest.mark.asyncio
async def test_no_injectable_surface_is_passthrough_baseline() -> None:
    """(c) No controllable in scope -> OBSERVE ends done with zero scored attacks."""
    llm = ScriptedLLM()
    optimizer = MuzzleOptimizer()
    await optimizer.initialize(Goal(description="leak the key"), [], [], llm)
    assert optimizer._can_inject is False

    traj = FakeTrajectory()
    await dispatch_event(optimizer, run_start(traj))
    end = await dispatch_event(optimizer, run_end())

    assert isinstance(end, RunEndResponse)
    assert end.done is True
    assert optimizer._exhausted is True
    assert optimizer._is_done() is True
    # Never advanced past OBSERVE and never scored an ATTACK run.
    assert optimizer._phase == "OBSERVE"
    assert optimizer._attempt_index == 0
    # OBSERVE still ran its analysis brain, but no PAIR/attack roles were ever invoked.
    roles = llm.roles_called()
    assert "summarizer" in roles and "grafter" in roles and "prompter" in roles
    assert "pair_attacker" not in roles
    assert "judge" not in roles


@pytest.mark.asyncio
async def test_no_feedback_never_self_certifies_success() -> None:
    """(d) With feedback withheld, the optimizer never self-certifies; it stops only on budget."""
    ctrl = make_controllable("tool:web", TOOL_A_TAG)
    llm = ScriptedLLM(overrides={"grafter": _grafter_naming("tool:web")})
    optimizer = MuzzleOptimizer(
        max_attempts=2, pair_judge_method="gcg", pair_n_streams=1, pair_n_iterations=1
    )
    await optimizer.initialize(Goal(description="leak the key"), [ctrl], [], llm)

    await _observe_run(optimizer, ctrl, is_content=True)
    await _active_run(optimizer, ctrl, is_content=True, end_event=run_end_no_feedback())
    assert optimizer._phase == "ATTACK"

    # ATTACK run 1: a feedback-less run_end carries evaluation=None. The optimizer must not
    # read success from it and must keep steering (refine the same vessel).
    _, _, end_attack1 = await _active_run(
        optimizer, ctrl, is_content=True, end_event=run_end_no_feedback()
    )
    assert end_attack1.done is False
    assert optimizer._succeeded is False
    assert optimizer._attempt_index == 1
    assert optimizer._reflection is not None  # steering, not stopping

    # ATTACK run 2: still no feedback -> it stops only because attempts are exhausted, and the
    # stop is NOT a self-certified success.
    _, _, end_attack2 = await _active_run(
        optimizer, ctrl, is_content=True, end_event=run_end_no_feedback()
    )
    assert end_attack2.done is True
    assert optimizer._succeeded is False
    assert optimizer._attempt_index == optimizer._max_attempts == 2
