"""Config parity: every upstream knob is a field on AttackAnythingConfig.

The table below is the upstream tunable surface (SEATSConfig core + SEATSDecon +
the separate main-method flags + the v2 frontier flags), with the upstream
default. If a field or default drifts, this test fails. ``seed_temperature`` is the
one deliberate omission (superred never sends a sampling temperature; documented
in config.py and ASSUMPTIONS.md), so it is not required here.
"""

from __future__ import annotations

import dataclasses

import pytest

from attack_anything_optimizer.config import AttackAnythingConfig

# knob -> upstream default
_UPSTREAM: dict[str, object] = {
    # SEATSConfig core (seats.py:52-94)
    "n_iterations": 30,
    "k_depth": 2,
    "k_breadth": 2,
    "k_cross": 1,
    "n_early_stop_successes": 3,
    "max_turns": 4,
    "max_consecutive_refusals": 3,
    "exploration_c": 1.414,
    "success_reward": 0.8,
    "archive_threshold": 0.3,
    "self_evolve_interval": 10,
    "n_cross_goal_seeds": 2,
    "n_seed_prompts": 3,
    "use_llm_judge": False,
    "max_target_queries_per_goal": 0,
    "seed": 42,
    "archive_max_size": 200,
    "archive_per_goal": 20,
    # SEATSDecon (seats_decon.py:457-470)
    "n_steps": 4,
    "n_decon_paths": 2,
    "use_persona": True,
    "max_recursion_depth": 2,
    # separate main-method (seats_feedback_decon_separate.py:151-218)
    "turn_independent": False,
    "recursive_leaf_attack": False,
    "recursive_max_depth": 2,
    "recursive_branch": 4,
    "recursive_wrappers_per_leaf": 3,
    "require_all_subtasks": True,
    "wrapper_selection": "ucb",
    "wrapper_priority": (
        "code_comment",
        "audit_ctx",
        "hypothetical",
        "json",
        "table",
        "reverse_engineer",
    ),
    "ucb_c": 1.4,
    "ucb_min_uses": 2,
    "validator_threshold": 6,
    "validator_max_retries": 2,
    # v2 frontier
    "goal_as_root": False,
    "fallback_enabled": False,
}


@pytest.mark.parametrize("knob,default", sorted(_UPSTREAM.items()))
def test_knob_present_with_upstream_default(knob: str, default: object) -> None:
    fields = {f.name for f in dataclasses.fields(AttackAnythingConfig)}
    assert knob in fields, f"upstream knob {knob!r} missing from AttackAnythingConfig"
    assert getattr(AttackAnythingConfig(), knob) == default, (
        f"{knob!r} default drifted from upstream {default!r}"
    )


def test_component_toggles_default_on() -> None:
    c = AttackAnythingConfig()
    assert (c.use_decomposition, c.use_feedback, c.use_tree_search, c.use_archive) == (
        True,
        True,
        True,
        True,
    )
