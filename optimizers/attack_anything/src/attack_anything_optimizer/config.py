"""Configuration for the Attack Anything optimizer.

One frozen dataclass carries the full upstream knob surface (mirroring the CLI of
``run_feedback_decon_separate.py`` and the ``SEATSConfig`` defaults) plus:

* **component toggles** (``use_decomposition`` / ``use_feedback`` / ``use_tree_search``
  / ``use_archive``) that switch the four building blocks on and off. All four
  default to on, which reproduces the paper's headline method (turn-fresh
  SEATS-FB-RDRT). Turning them off reproduces the ablation ladder:
  ``use_decomposition=False`` -> plain SEATS / SEATS-FB; ``use_tree_search=False``
  -> RDRT-style decompose-and-attack; etc.
* **v2 frontier flags** (``goal_as_root`` / ``fallback_enabled``), default off.

``as_seats_config()`` builds the vendored, byte-identical
:class:`~attack_anything_optimizer._vendor.seats.SEATSConfig` from these fields, so
the upstream engine helpers see exactly the values they expect.
"""

from __future__ import annotations

from dataclasses import dataclass

from superred.core.types.llm import LLMConfig

from attack_anything_optimizer._vendor.seats import SEATSConfig


@dataclass(frozen=True)
class AttackAnythingConfig:
    """Full knob surface for the optimizer. All defaults are the upstream defaults."""

    # --- SEATSConfig fields (upstream seats.py:53-94 defaults) ---------------
    n_iterations: int = 30
    k_depth: int = 2
    k_breadth: int = 2
    k_cross: int = 1
    n_early_stop_successes: int = 3
    max_turns: int = 4
    max_consecutive_refusals: int = 3
    exploration_c: float = 1.414
    success_reward: float = 0.8
    archive_threshold: float = 0.3
    self_evolve_interval: int = 10
    n_cross_goal_seeds: int = 2
    n_seed_prompts: int = 3
    use_llm_judge: bool = False
    max_target_queries_per_goal: int = 0
    seed: int = 42

    # --- EliteArchive sizing (upstream run_*.py defaults) --------------------
    archive_max_size: int = 200
    archive_per_goal: int = 20

    # --- SEATS-Decon fields (upstream seats_decon.py defaults) ---------------
    n_steps: int = 4
    n_decon_paths: int = 2
    use_persona: bool = True
    max_recursion_depth: int = 2

    # --- turn-fresh SEATS-FB-RDRT-separate flags (upstream main-method) ------
    turn_independent: bool = False
    recursive_leaf_attack: bool = False
    recursive_max_depth: int = 2
    recursive_branch: int = 4
    recursive_wrappers_per_leaf: int = 3
    require_all_subtasks: bool = True
    wrapper_selection: str = "ucb"  # "ucb" | "priority" | "random"
    #: Legacy fixed order for ``wrapper_selection="priority"`` (upstream sep:180-182).
    wrapper_priority: tuple[str, ...] = (
        "code_comment",
        "audit_ctx",
        "hypothetical",
        "json",
        "table",
        "reverse_engineer",
    )
    ucb_c: float = 1.4  # UCB exploration constant for wrapper selection (upstream sep:185)
    ucb_min_uses: int = 2  # force-explore wrappers used < this many times (upstream sep:187)
    validator_threshold: int = 6
    validator_max_retries: int = 2

    # NOTE: the upstream ``seed_temperature`` knob is intentionally NOT exposed.
    # superred never sends a sampling temperature to the provider (see
    # tests/test_no_temperature.py); the upstream temperature knobs are therefore
    # inert (the provider default rides and each call is an independent draw), so
    # exposing the field would only be a name that trips the guard for no effect.

    # --- judge mode (upstream JUDGE_MODE env; None -> env/permissive) --------
    judge_mode: str | None = None  # "permissive" | "strict" | "calibrated"

    # --- optional endpoint decoupling (upstream --judge_* / --validator_*) ----
    #: When set, the optimizer builds a separate LLMClient for the judge and/or
    #: the decomposition validator (upstream points these at a stronger model).
    #: Like the target's own inference, these are OUT of the controller's
    #: attacker cost cap. ``None`` reuses the attacker client (self.llm), and a
    #: ``None`` validator config leaves the validator gate off (accept first
    #: decomposition) -- the current default.
    judge_llm_config: LLMConfig | None = None
    validator_llm_config: LLMConfig | None = None

    # --- component toggles (all on = the paper's headline method) ------------
    use_decomposition: bool = True
    use_feedback: bool = True
    use_tree_search: bool = True
    use_archive: bool = True

    # --- v2 frontier extension (default off) ---------------------------------
    goal_as_root: bool = False
    fallback_enabled: bool = False

    def as_seats_config(self) -> SEATSConfig:
        """Build the vendored, byte-identical ``SEATSConfig`` from these fields."""
        return SEATSConfig(
            n_iterations=self.n_iterations,
            k_depth=self.k_depth,
            k_breadth=self.k_breadth,
            k_cross=self.k_cross,
            n_early_stop_successes=self.n_early_stop_successes,
            max_turns=self.max_turns,
            max_consecutive_refusals=self.max_consecutive_refusals,
            exploration_c=self.exploration_c,
            success_reward=self.success_reward,
            archive_threshold=self.archive_threshold,
            self_evolve_interval=self.self_evolve_interval,
            n_cross_goal_seeds=self.n_cross_goal_seeds,
            n_seed_prompts=self.n_seed_prompts,
            use_llm_judge=self.use_llm_judge,
            max_target_queries_per_goal=self.max_target_queries_per_goal,
            seed=self.seed,
        )


__all__ = ["AttackAnythingConfig"]
